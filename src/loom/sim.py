"""
Ground-truth plant simulator.

This module is NOT the digital twin. It is the *physical world* stand-in: it
produces the data the twin is allowed to see, and separately records the
ground truth the twin is never shown. Everything downstream is scored against
that hidden truth, which is what turns the prototype into a measurable claim
rather than a demo that always agrees with itself.

Kernel
------
A serial flow line with finite buffers under blocking-after-service. For a
tandem line this admits an exact recursion, so no general event queue is
needed:

    start[u][j]  = max(depart[u][j-1], depart[u-1][j])
    finish[u][j] = start[u][j] + p[u][j]
    depart[u][j] = max(finish[u][j], start[u-B_j][j+1])

which yields, for free, the three machine states the active-period method of
Roser et al. (2001, 2002) consumes:

    starved  = [depart[u-1][j], start[u][j])      inactive
    working  = [start[u][j],    finish[u][j])     ACTIVE
    blocked  = [finish[u][j],   depart[u][j])     inactive

That identity is the load-bearing idea in Loom: those three states are
recoverable from arrival/departure scans alone, so a station with no process
sensors is still legible to the bottleneck engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LineConfig, DEFECT_TYPES

SEC_PER_HOUR = 3600.0


# --------------------------------------------------------------------------
# Fault library (ground truth - hidden from the twin)
# --------------------------------------------------------------------------


@dataclass
class Fault:
    fault_id: str
    kind: str                 # PARAM_DRIFT | CYCLE_DEGRADE | LOT_CONTAM | ENV_EXCURSION | MICROSTOP
    target: str               # station id, lot id, or zone
    start_day: float
    end_day: float
    magnitude: float
    defect_type: str | None = None
    note: str = ""


def default_faults() -> list[Fault]:
    """Five faults chosen to exercise a different failure mode each."""
    return [
        Fault("F1", "PARAM_DRIFT", "S12", 9.0, 20.0, magnitude=0.30,
              defect_type="LOOSE_FASTENER",
              note="Torque tool drifts ~0.30 Nm/day below nominal. Defect is "
                   "created in the body shop but only surfaces at S42 final "
                   "inspection ~40 min and 30 stations later."),
        Fault("F2", "CYCLE_DEGRADE", "S28", 6.0, 20.0, magnitude=0.0075,
              note="Un-instrumented manual station degrades 0.75%/day. Invisible "
                   "to any sensor; must be inferred from scan timestamps."),
        Fault("F3", "LOT_CONTAM", "LOT-425", 0.0, 99.0, magnitude=6.2,
              defect_type="ELECTRICAL_FAULT",
              note="One supplier lot of connectors carries 6.2x baseline defect "
                   "propensity. Attribution must land on the lot, not the station."),
        Fault("F4", "ENV_EXCURSION", "PAINT", 15.0, 16.6, magnitude=14.0,
              defect_type="PAINT_FINISH",
              note="Booth humidity rises ~14 points for 1.5 days. Confounder: a "
                   "naive station-level test will blame whichever paint booth ran "
                   "most units in the window."),
        Fault("F6", "OPERATOR_VAR", "FINAL-OP2", 0.0, 99.0, magnitude=1.55,
              defect_type="FIT_MISALIGN",
              note="One operator in final assembly runs 55% wider cycle-time "
                   "variance and a raised miss rate on manual fits. Tests "
                   "whether attribution can separate a person from the shift "
                   "and the station they happened to be working."),
        Fault("F5", "MICROSTOP", "S07", 3.0, 20.0, magnitude=0.011,
              note="Intermittent 90-240 s micro-stoppages on an instrumented "
                   "robot. Tests SPC on a station that does have sensors."),
    ]


def adversarial_faults() -> list[Fault]:
    """Fault shapes the detectors were NOT designed against.

    The five default faults are drift, degradation, contamination, an
    environmental excursion and micro-stops - and every detector in Loom was
    built while looking at them. Scoring against your own fault library is how
    you end up with a system that works only on the failures you imagined.

    These three break a different assumption each:

    * STEP      an abrupt shift, not a ramp. The rolling-horizon twin
                extrapolates a *linear* trend, so a step is exactly what its
                forecast model cannot represent.
    * OSCILLATE a periodic swing (thermal cycling, day/night ambient). EWMA and
                CUSUM both assume a shift toward a new level; a signal that
                keeps returning to baseline defeats an accumulator.
    * BURST     tightly clustered failures with long quiet gaps. The
                2-block persistence rule that suppresses false alarms is
                precisely what makes a short burst invisible.

    Whether each is caught is reported, not assumed. Misses are the point.
    """
    return [
        Fault("A1", "STEP", "S30", 11.0, 20.0, magnitude=2.6,
              defect_type="LOOSE_FASTENER",
              note="Torque set-point steps 2.6 Nm after a tool change. No ramp."),
        Fault("A2", "OSCILLATE", "S19", 4.0, 20.0, magnitude=0.055,
              note="Booth cycle time swings +/-5.5% on a ~2 day period, "
                   "returning to baseline each time."),
        Fault("A3", "BURST", "S36", 7.0, 20.0, magnitude=0.16,
              note="Micro-stoppages arrive in short dense clusters separated "
                   "by quiet days, rather than at a steady rate."),
    ]


def beta_faults() -> list[Fault]:
    """The same five failure modes, relocated onto Plant Beta's layout.

    Kept structurally identical so that any difference in results between the
    two lines is attributable to the line - sparser sensors, longer takt, one
    shift - and not to a different set of problems.
    """
    return [
        Fault("F1", "PARAM_DRIFT", "B17", 8.0, 20.0, magnitude=0.32,
              defect_type="LOOSE_FASTENER",
              note="Torque tool drift in final assembly."),
        Fault("F2", "CYCLE_DEGRADE", "B14", 5.0, 20.0, magnitude=0.0080,
              note="Un-instrumented manual station degrades 0.80%/day."),
        Fault("F3", "LOT_CONTAM", "LOT-418", 0.0, 99.0, magnitude=5.8,
              defect_type="ELECTRICAL_FAULT",
              note="Contaminated connector lot."),
        Fault("F6", "OPERATOR_VAR", "FINAL-OP2", 0.0, 99.0, magnitude=1.55,
              defect_type="FIT_MISALIGN",
              note="Operator variation on manual fits."),
        Fault("F5", "MICROSTOP", "B03", 3.0, 20.0, magnitude=0.012,
              note="Intermittent micro-stoppages on an instrumented robot."),
    ]


FAULT_SETS = {"alpha": default_faults, "beta": beta_faults,
              "adversarial": adversarial_faults}


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class PlantData:
    """Everything the simulator produced. `truth` is withheld from the twin."""

    cfg: LineConfig
    units: pd.DataFrame                     # one row per unit
    start: np.ndarray                       # (n_units, n_stations)
    finish: np.ndarray
    depart: np.ndarray
    proc: np.ndarray
    signals: dict[str, dict[str, np.ndarray]]   # sid -> signal -> (n_units,)
    inspections: pd.DataFrame               # detected defects
    truth: dict = field(default_factory=dict)   # HIDDEN

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_stations(self) -> int:
        return self.cfg.n_stations

    def blocked(self) -> np.ndarray:
        return self.depart - self.finish

    def starved(self) -> np.ndarray:
        s = np.zeros_like(self.start)
        s[1:, :] = self.start[1:, :] - self.depart[:-1, :]
        s[0, :] = 0.0
        return np.maximum(s, 0.0)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


def simulate(cfg: LineConfig, faults: list[Fault] | None = None) -> PlantData:
    rng = np.random.default_rng(cfg.seed)
    faults = default_faults() if faults is None else faults

    units_per_shift = int(cfg.shift_hours * SEC_PER_HOUR / cfg.takt_s)
    units_per_day = units_per_shift * cfg.shifts_per_day
    n_units = units_per_day * cfg.days
    n_st = cfg.n_stations
    stations = cfg.stations

    # ---------------- unit-level attributes ----------------
    uidx = np.arange(n_units)
    day = uidx // units_per_day
    within_day = uidx % units_per_day
    shift = within_day // units_per_shift  # 0 = A, 1 = B
    release = (day * 24.0 + shift * cfg.shift_hours) * SEC_PER_HOUR \
        + (within_day % units_per_shift) * cfg.takt_s
    # small release jitter from upstream body-shop supply
    release = release + rng.normal(0, cfg.takt_s * 0.05, n_units)
    release = np.maximum.accumulate(release)

    variant = rng.choice(["SEDAN", "SUV", "SUV-LWB"], size=n_units,
                         p=[0.44, 0.38, 0.18])
    variant_factor = np.select(
        [variant == "SEDAN", variant == "SUV", variant == "SUV-LWB"],
        [0.97, 1.03, 1.09],
    )

    lot_no = 400 + uidx // cfg.lot_size_units
    lot_id = np.array([f"LOT-{n}" for n in lot_no])

    zones = sorted({s.zone for s in stations})
    # Operators are drawn per zone per shift rather than by a fixed rota. A
    # deterministic rota makes operator identity perfectly collinear with shift,
    # which would hand the attribution engine a free pass it would not get in a
    # real plant.
    operator = {}
    for k, z in enumerate(zones):
        pool = rng.integers(1, cfg.operators_per_zone + 1,
                            size=cfg.days * cfg.shifts_per_day + 2)
        slot = day * cfg.shifts_per_day + shift
        roll = (slot + k * 7) % len(pool)
        operator[z] = np.array([f"{z}-OP{pool[r]}" for r in roll])

    units = pd.DataFrame({
        "unit": uidx,
        "vin": [f"VIN{100000 + i}" for i in uidx],
        "release_s": release,
        "day": day,
        "shift": np.where(shift == 0, "A", "B"),
        "variant": variant,
        "lot_id": lot_id,
        **{f"operator_{z}": operator[z] for z in zones},
    })

    day_f = release / (24 * SEC_PER_HOUR)   # continuous day for fault ramps

    # ---------------- processing times ----------------
    proc = np.zeros((n_units, n_st))
    repair = np.zeros((n_units, n_st))
    for j, st in enumerate(stations):
        base = st.nominal_cycle_s
        cv = st.stype.cycle_cv
        # lognormal keeps times positive and right-skewed like real manual work
        sigma = np.sqrt(np.log(1 + cv ** 2))
        mu = np.log(base) - 0.5 * sigma ** 2
        p = rng.lognormal(mu, sigma, n_units)
        # inspection and paint are variant-sensitive; robots less so
        if st.type_name in ("manual_fit", "inspection_gate", "electrical_fit"):
            p = p * variant_factor
        proc[:, j] = p

    truth_cycle = proc.copy()          # pre-fault baseline, for soft-sensor scoring

    # ---------------- apply faults ----------------
    truth: dict = {"faults": [], "defect_origin": {}}
    sid_to_j = {s.sid: j for j, s in enumerate(stations)}

    for f in faults:
        rec = {"fault_id": f.fault_id, "kind": f.kind, "target": f.target,
               "start_day": f.start_day, "end_day": f.end_day,
               "magnitude": f.magnitude, "defect_type": f.defect_type,
               "note": f.note}

        if f.kind == "CYCLE_DEGRADE":
            j = sid_to_j[f.target]
            ramp = np.clip(day_f - f.start_day, 0, f.end_day - f.start_day)
            proc[:, j] *= (1.0 + f.magnitude * ramp)

        elif f.kind == "OPERATOR_VAR":
            zone = f.target.split("-")[0]
            hit = operator[zone] == f.target
            for k, s in enumerate(stations):
                if s.zone != zone or s.type_name != "manual_fit":
                    continue
                extra = rng.normal(0, proc[:, k] * 0.10 * (f.magnitude - 1), n_units)
                proc[:, k] += hit * np.abs(extra) * 2.0
            rec["n_units_affected"] = int(hit.sum())

        elif f.kind == "STEP":
            j = sid_to_j.get(f.target)
            if j is not None:
                pass          # applied to the signal, not the cycle time

        elif f.kind == "OSCILLATE":
            j = sid_to_j[f.target]
            act = (day_f >= f.start_day) & (day_f <= f.end_day)
            proc[:, j] *= 1.0 + f.magnitude * np.sin(
                2 * np.pi * (day_f - f.start_day) / 2.0) * act

        elif f.kind == "BURST":
            j = sid_to_j[f.target]
            # dense clusters separated by quiet gaps
            phase = ((day_f - f.start_day) % 4.0) < 0.6
            act = (day_f >= f.start_day) & (day_f <= f.end_day) & phase
            hit = act & (rng.random(n_units) < f.magnitude)
            dur = rng.uniform(80, 260, n_units) * hit
            repair[:, j] += dur
            proc[:, j] += dur
            rec["n_events"] = int(hit.sum())

        elif f.kind == "MICROSTOP":
            j = sid_to_j[f.target]
            active = (day_f >= f.start_day) & (day_f <= f.end_day)
            hit = active & (rng.random(n_units) < f.magnitude)
            dur = rng.uniform(90, 240, n_units) * hit
            repair[:, j] += dur
            proc[:, j] += dur
            rec["n_events"] = int(hit.sum())

        truth["faults"].append(rec)

    # ---------------- line recursion ----------------
    start = np.zeros((n_units, n_st))
    finish = np.zeros((n_units, n_st))
    depart = np.zeros((n_units, n_st))
    buf = np.array([s.buffer_capacity for s in stations])
    trans = np.array([s.transport_s for s in stations])

    # Unit-by-unit forward pass. Vectorising across stations is not possible
    # because of the downstream-start dependency, but 19k x 42 in pure numpy
    # scalars is a couple of seconds, which is fine for a proof of concept.
    par = np.array([s.parallel for s in stations])

    def _segment(j0: int, j1: int, order: np.ndarray,
                 entry_time: np.ndarray) -> None:
        """Run stations [j0, j1) for units in the given processing order.

        `order` is the sequence in which units are worked at these stations,
        which is NOT the unit index once a rework loop exists. Everything is
        indexed through `order`, so a reworked unit that re-enters behind
        thirty other vehicles is handled correctly.
        """
        for pos, u in enumerate(order):
            for j in range(j0, j1):
                if j == j0:
                    avail = entry_time[u]
                else:
                    avail = depart[u, j - 1] + trans[j - 1]
                # a station with p machines is free when the unit p positions
                # earlier in this station's own sequence has left
                p = par[j]
                free = depart[order[pos - p], j] if pos - p >= 0 else -np.inf
                s = avail if avail > free else free
                start[u, j] = s
                fin = s + proc[u, j]
                finish[u, j] = fin
                if j == n_st - 1:
                    depart[u, j] = fin
                else:
                    # effective downstream capacity is the buffer plus the
                    # extra slots the parallel machines themselves provide
                    cap = buf[j] + (par[j + 1] - 1)
                    if pos - cap >= 0:
                        blk = start[order[pos - cap], j + 1]
                        depart[u, j] = fin if fin > blk else blk
                    else:
                        depart[u, j] = fin

    # A rework gate splits the line: upstream of it every unit is in release
    # order, downstream the order is whatever the rework cell hands back.
    rework_j = next((j for j, s in enumerate(stations) if s.rework_rate > 0), None)
    reworked = np.zeros(n_units, dtype=bool)

    if rework_j is None:
        _segment(0, n_st, np.arange(n_units), release)
    else:
        _segment(0, rework_j + 1, np.arange(n_units), release)
        reworked = rng.random(n_units) < stations[rework_j].rework_rate
        rw_time = np.abs(rng.normal(stations[rework_j].rework_minutes,
                                    stations[rework_j].rework_minutes * 0.3,
                                    n_units)) * 60.0
        reentry = depart[:, rework_j] + np.where(reworked, rw_time, 0.0) \
            + trans[rework_j]
        # units re-join in the order they actually become available again
        order2 = np.argsort(reentry, kind="stable")
        _segment(rework_j + 1, n_st, order2, reentry)
        # A rework cell has its own read point in any plant that tracks WIP, so
        # the re-entry timestamp is observable. Without it, a unit returning to
        # an idle station is indistinguishable from a slow station - see
        # `reconstruct_states`. This is a deployment requirement, not an
        # assumption we can infer our way around.
        truth["reentry_scan_s"] = np.where(reworked, reentry, np.nan)

    truth.setdefault("reentry_scan_s", np.full(n_units, np.nan))
    truth["reworked"] = reworked
    truth["n_reworked"] = int(reworked.sum())

    # ---------------- signals + latent defects ----------------
    signals: dict[str, dict[str, np.ndarray]] = {}
    latent = {d: np.zeros(n_units, dtype=bool) for d in DEFECT_TYPES}
    origin_station = {d: np.full(n_units, "", dtype=object) for d in DEFECT_TYPES}

    torque_drift = {f.target: f for f in faults if f.kind == "PARAM_DRIFT"}
    lot_faults = [f for f in faults if f.kind == "LOT_CONTAM"]
    env_faults = [f for f in faults if f.kind == "ENV_EXCURSION"]

    for j, st in enumerate(stations):
        sid = st.sid
        sig: dict[str, np.ndarray] = {}
        stype = st.stype

        # Cycle time is always derivable (scanners), even at dark stations.
        sig["cycle_time_s"] = proc[:, j].copy()

        excursion = np.zeros(n_units)   # 0..1 severity driving defect creation

        if "torque_nm" in stype.signals:
            nominal, spec_low = 42.0, 39.5
            tq = rng.normal(nominal, 1.15, n_units)
            for fs in faults:
                if fs.kind == "STEP" and fs.target == sid:
                    tq = tq - fs.magnitude * ((day_f >= fs.start_day)
                                              & (day_f <= fs.end_day))
            f = torque_drift.get(sid)
            if f is not None:
                ramp = np.clip(day_f - f.start_day, 0, f.end_day - f.start_day)
                tq = tq - f.magnitude * ramp
            sig["torque_nm"] = tq
            sig["angle_deg"] = rng.normal(88.0, 3.4, n_units) + (nominal - tq) * 0.9
            excursion = np.clip((spec_low - tq) / 3.0, 0, 1.5)

        if "weld_current_a" in stype.signals:
            wc = rng.normal(255.0, 7.5, n_units)
            sig["weld_current_a"] = wc
            sig["servo_load_pct"] = rng.normal(62.0, 6.0, n_units) \
                + (proc[:, j] - st.nominal_cycle_s) * 0.25
            excursion = np.clip((238.0 - wc) / 12.0, 0, 1.5)

        if "flow_rate_mlps" in stype.signals and st.type_name == "sealer":
            fr = rng.normal(14.6, 0.9, n_units)
            sig["flow_rate_mlps"] = fr
            sig["nozzle_pressure_bar"] = rng.normal(4.3, 0.28, n_units) + (fr - 14.6) * 0.1
            excursion = np.clip((13.1 - fr) / 1.4, 0, 1.5)

        if st.type_name == "paint_booth":
            hum = rng.normal(47.5, 2.6, n_units)
            for f in env_faults:
                if f.target == st.zone:
                    win = (day_f >= f.start_day) & (day_f <= f.end_day)
                    # trapezoidal ramp in/out over ~3 h
                    hum = hum + f.magnitude * win
            sig["booth_humidity_pct"] = hum
            sig["booth_temp_c"] = rng.normal(23.4, 0.7, n_units) - (hum - 47.5) * 0.04
            sig["flow_rate_mlps"] = rng.normal(9.8, 0.5, n_units)
            excursion = np.clip((hum - 58.0) / 14.0, 0, 1.5)

        if st.type_name == "oven":
            zt = rng.normal(163.0, 2.9, n_units)
            sig["zone_temp_c"] = zt
            excursion = np.clip((155.0 - zt) / 6.0, 0, 1.5)

        if st.type_name == "electrical_fit":
            force = rng.normal(31.0, 2.4, n_units)
            ohm = rng.normal(0.042, 0.006, n_units)
            bad_lot = np.zeros(n_units, dtype=bool)
            for f in lot_faults:
                # the contaminated lot only bites at the station that fits that
                # part; on Plant Beta that is a different station id
                if sid in ("S33", "B20"):
                    bad_lot |= (lot_id == f.target)
            ohm = ohm + bad_lot * rng.normal(0.028, 0.010, n_units)
            sig["insertion_force_n"] = force + bad_lot * rng.normal(4.1, 1.6, n_units)
            sig["continuity_ohm"] = ohm
            excursion = np.clip((ohm - 0.055) / 0.02, 0, 1.5)

        # latent defect creation
        dtype = stype.defect_type
        if dtype is not None:
            base_rate = stype.defect_sensitivity * 0.09
            op_lift = np.zeros(n_units)
            for f in faults:
                if f.kind == "OPERATOR_VAR" and st.type_name == "manual_fit" \
                        and st.zone == f.target.split("-")[0]:
                    op_lift = (operator[st.zone] == f.target) * 0.055
            p_def = np.clip(base_rate + 0.42 * excursion + op_lift, 0, 0.85)
            hit = rng.random(n_units) < p_def
            newly = hit & (~latent[dtype])
            latent[dtype] |= hit
            origin_station[dtype][newly] = sid

        # A dark station emits nothing but its scan timestamps. What it *would*
        # have emitted is kept in `truth` so the sensor-ROI planner can ask the
        # counterfactual question offline. It is never exposed to the twin.
        signals[sid] = sig if st.instrumented else {}
        if not st.instrumented:
            truth.setdefault("hidden_signals", {})[sid] = sig
        # keep true cycle for soft-sensor scoring, never exposed to the twin
        truth.setdefault("true_cycle", {})[sid] = proc[:, j].copy()

    # ---------------- inspection gates ----------------
    rows = []
    remaining = {d: latent[d].copy() for d in DEFECT_TYPES}
    for j, st in enumerate(stations):
        if not st.inspects:
            continue
        for dtype in st.inspects:
            can = remaining[dtype]
            caught = can & (rng.random(n_units) < st.detect_prob)
            remaining[dtype] = can & ~caught
            idx = np.flatnonzero(caught)
            for u in idx:
                rows.append((int(u), st.sid, dtype, float(depart[u, j]),
                             origin_station[dtype][u]))

    inspections = pd.DataFrame(
        rows, columns=["unit", "gate", "defect_type", "detected_s", "_true_origin"]
    ).sort_values("detected_s").reset_index(drop=True)

    escapes = {d: int(remaining[d].sum()) for d in DEFECT_TYPES}

    truth["defect_origin"] = {d: origin_station[d] for d in DEFECT_TYPES}
    truth["latent"] = latent
    truth["escapes"] = escapes
    truth["repair_s"] = repair
    truth["units_per_day"] = units_per_day
    truth["units_per_shift"] = units_per_shift

    return PlantData(cfg=cfg, units=units, start=start, finish=finish,
                     depart=depart, proc=proc, signals=signals,
                     inspections=inspections, truth=truth)


def observable_view(pd_: PlantData) -> dict:
    """What the twin is actually allowed to read.

    Instrumented stations expose their process signals. Dark stations expose
    only the arrival and departure scans - which is exactly what an RFID or
    barcode reader at the station boundary would give you in a real plant.
    """
    cfg = pd_.cfg
    scans_in = np.zeros_like(pd_.start)
    scans_out = np.zeros_like(pd_.depart)
    scans_in[:] = pd_.start
    scans_out[:] = pd_.depart

    exposed_signals = {}
    for st in cfg.stations:
        if st.instrumented:
            exposed_signals[st.sid] = dict(pd_.signals[st.sid])
        else:
            exposed_signals[st.sid] = {}

    return {
        "units": pd_.units,
        # observable: the rework cell's own read point
        "reentry_scan_s": pd_.truth.get("reentry_scan_s"),
        "rework_station_index": next(
            (s.index - 1 for s in cfg.stations if s.rework_rate > 0), None),
        "scan_in_s": scans_in,
        "scan_out_s": scans_out,
        "signals": exposed_signals,
        "inspections": pd_.inspections.drop(columns=["_true_origin"]),
        "cfg": cfg,
    }
