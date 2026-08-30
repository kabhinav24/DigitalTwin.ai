"""
Station-type library and line configuration.

Kattenstroth et al. (2024) identify "high initial creation effort", "hardcoded
models need manual adaption effort" and "modularisation and parameterisation"
as the central obstacles to reusing a DES model as a digital factory twin.
Loom answers that by never hard-coding a line. A line is a list of station
*instances*, each pointing at a reusable *station type*. Porting the twin to a
new plant is a config change, not a code change.

Station types map to ISO 23247 Observable Manufacturing Elements (OMEs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Station type library (reusable across plants)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StationType:
    """A reusable archetype. `signals` are what a *fully* instrumented
    instance of this type would emit. An instance may emit fewer."""

    name: str
    nominal_cycle_s: float
    cycle_cv: float  # coefficient of variation of processing time
    signals: tuple[str, ...]
    # probability an unattended process excursion turns into a latent defect
    defect_sensitivity: float = 0.0
    defect_type: str | None = None
    retrofit_cost_inr: int = 0  # cost to instrument one dark station of this type


STATION_TYPES: dict[str, StationType] = {
    "weld_robot": StationType(
        "weld_robot", 46.0, 0.06,
        ("cycle_time_s", "weld_current_a", "servo_load_pct"),
        defect_sensitivity=0.030, defect_type="WELD_POROSITY",
        retrofit_cost_inr=310_000,
    ),
    "torque_station": StationType(
        "torque_station", 52.0, 0.08,
        ("cycle_time_s", "torque_nm", "angle_deg"),
        defect_sensitivity=0.040, defect_type="LOOSE_FASTENER",
        retrofit_cost_inr=185_000,
    ),
    "manual_fit": StationType(
        "manual_fit", 53.0, 0.16,
        ("cycle_time_s",),
        defect_sensitivity=0.012, defect_type="FIT_MISALIGN",
        retrofit_cost_inr=95_000,
    ),
    "sealer": StationType(
        "sealer", 50.0, 0.09,
        ("cycle_time_s", "flow_rate_mlps", "nozzle_pressure_bar"),
        defect_sensitivity=0.018, defect_type="SEALANT_VOID",
        retrofit_cost_inr=240_000,
    ),
    "paint_booth": StationType(
        "paint_booth", 58.0, 0.07,
        ("cycle_time_s", "booth_temp_c", "booth_humidity_pct", "flow_rate_mlps"),
        defect_sensitivity=0.035, defect_type="PAINT_FINISH",
        retrofit_cost_inr=420_000,
    ),
    "oven": StationType(
        "oven", 56.0, 0.03,
        ("cycle_time_s", "zone_temp_c"),
        defect_sensitivity=0.010, defect_type="PAINT_FINISH",
        retrofit_cost_inr=265_000,
    ),
    "electrical_fit": StationType(
        "electrical_fit", 55.0, 0.11,
        ("cycle_time_s", "insertion_force_n", "continuity_ohm"),
        defect_sensitivity=0.028, defect_type="ELECTRICAL_FAULT",
        retrofit_cost_inr=205_000,
    ),
    "inspection_gate": StationType(
        "inspection_gate", 41.0, 0.12,
        ("cycle_time_s",),
        defect_sensitivity=0.0, defect_type=None,
        retrofit_cost_inr=120_000,
    ),
}


@dataclass
class Station:
    """A physical station instance on a specific line."""

    sid: str
    type_name: str
    zone: str
    index: int
    buffer_capacity: int = 3
    instrumented: bool = True
    # per-instance multiplier on the type's nominal cycle
    cycle_scale: float = 1.0
    # seconds on the conveyor between this station and the next. Real plants
    # scan at the station, so transport sits *between* two scans and inflates
    # the apparent dwell of the downstream station unless it is calibrated out.
    transport_s: float = 4.5
    # Number of identical machines working in parallel at this position. A
    # real body shop runs redundant robots so one can be serviced without
    # stopping the line; that breaks any model assuming one unit at a time.
    parallel: int = 1
    # Fraction of units this gate diverts to an offline rework cell. A
    # reworked unit re-enters the line later and OUT OF SEQUENCE, which is the
    # assumption most flow-line models quietly depend on.
    rework_rate: float = 0.0
    rework_minutes: float = 0.0
    # inspection gates only
    inspects: tuple[str, ...] = ()
    detect_prob: float = 0.0

    @property
    def stype(self) -> StationType:
        return STATION_TYPES[self.type_name]

    @property
    def nominal_cycle_s(self) -> float:
        return self.stype.nominal_cycle_s * self.cycle_scale

    @property
    def observed_signals(self) -> tuple[str, ...]:
        """A dark station still gives arrival/departure scans, so cycle time is
        *derivable* but not *measured*. Nothing else is available."""
        return self.stype.signals if self.instrumented else ()


@dataclass
class LineConfig:
    name: str
    stations: list[Station]
    takt_s: float = 60.0
    shift_hours: float = 8.0
    shifts_per_day: int = 2
    days: int = 20
    seed: int = 7
    # supplier lots rotate through the line
    lot_size_units: int = 420
    operators_per_zone: int = 4

    @property
    def has_rework(self) -> bool:
        return any(s.rework_rate > 0 for s in self.stations)

    @property
    def has_parallel(self) -> bool:
        return any(s.parallel > 1 for s in self.stations)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def dark_stations(self) -> list[str]:
        return [s.sid for s in self.stations if not s.instrumented]

    @property
    def coverage(self) -> float:
        return sum(s.instrumented for s in self.stations) / len(self.stations)

    def by_id(self, sid: str) -> Station:
        return next(s for s in self.stations if s.sid == sid)

    def to_json(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "takt_s": self.takt_s,
            "shift_hours": self.shift_hours,
            "shifts_per_day": self.shifts_per_day,
            "days": self.days,
            "seed": self.seed,
            "stations": [asdict(s) for s in self.stations],
        }
        Path(path).write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------
# The reference line: "Plant Alpha, Line 2" - mixed-model vehicle assembly
# --------------------------------------------------------------------------

# Layout follows the Accenture reference parameters: 30-50 stations across
# body construction, paint and final assembly, with uneven sensor coverage
# (majority instrumented, a meaningful minority on manual checks only).

_LAYOUT: list[tuple[str, str, bool, float]] = [
    # (station_type, zone, instrumented, cycle_scale)
    ("weld_robot",     "BODY",  True,  1.00),   # S01
    ("weld_robot",     "BODY",  True,  1.05),   # S02
    ("manual_fit",     "BODY",  False, 0.98),   # S03  dark
    ("weld_robot",     "BODY",  True,  1.02),   # S04
    ("sealer",         "BODY",  True,  1.00),   # S05
    ("manual_fit",     "BODY",  False, 1.00),   # S06  dark
    ("weld_robot",     "BODY",  True,  1.10),   # S07  <- intermittent fault
    ("torque_station", "BODY",  True,  0.97),   # S08
    ("manual_fit",     "BODY",  False, 1.01),   # S09  dark
    ("weld_robot",     "BODY",  True,  1.03),   # S10
    ("sealer",         "BODY",  False, 1.00),   # S11  dark
    ("torque_station", "BODY",  True,  1.04),   # S12  <- torque drift fault
    ("manual_fit",     "BODY",  False, 1.02),   # S13  dark
    ("inspection_gate","BODY",  True,  1.00),   # S14  BIW gate
    ("paint_booth",    "PAINT", True,  1.00),   # S15
    ("paint_booth",    "PAINT", True,  1.00),   # S16
    ("oven",           "PAINT", True,  1.03),   # S17  structural bottleneck (days 0-12)
    ("manual_fit",     "PAINT", False, 0.95),   # S18  dark
    ("paint_booth",    "PAINT", True,  1.00),   # S19
    ("oven",           "PAINT", True,  1.01),   # S20
    ("manual_fit",     "PAINT", False, 1.00),   # S21  dark
    ("paint_booth",    "PAINT", True,  0.99),   # S22
    ("oven",           "PAINT", True,  1.00),   # S23
    ("inspection_gate","PAINT", True,  1.00),   # S24  paint gate
    ("manual_fit",     "FINAL", False, 1.00),   # S25  dark
    ("electrical_fit", "FINAL", True,  1.00),   # S26
    ("torque_station", "FINAL", True,  1.01),   # S27
    ("manual_fit",     "FINAL", False, 1.04),   # S28  dark <- slow degradation
    ("electrical_fit", "FINAL", True,  0.98),   # S29
    ("torque_station", "FINAL", True,  1.03),   # S30
    ("manual_fit",     "FINAL", False, 0.98),   # S31  dark
    ("sealer",         "FINAL", True,  1.02),   # S32
    ("electrical_fit", "FINAL", True,  1.05),   # S33  <- lot contamination
    ("torque_station", "FINAL", True,  0.99),   # S34
    ("manual_fit",     "FINAL", False, 1.03),   # S35  dark
    ("electrical_fit", "FINAL", True,  1.00),   # S36
    ("torque_station", "FINAL", True,  1.02),   # S37
    ("manual_fit",     "FINAL", False, 0.96),   # S38  dark
    ("sealer",         "FINAL", True,  1.01),   # S39
    ("electrical_fit", "FINAL", True,  1.03),   # S40
    ("manual_fit",     "FINAL", False, 1.00),   # S41  dark
    ("inspection_gate","FINAL", True,  1.15),   # S42  final inspection
]

_GATES = {
    "S14": (("WELD_POROSITY", "SEALANT_VOID"), 0.55),
    "S24": (("PAINT_FINISH",), 0.62),
    "S42": (("WELD_POROSITY", "SEALANT_VOID", "PAINT_FINISH",
             "LOOSE_FASTENER", "ELECTRICAL_FAULT", "FIT_MISALIGN"), 0.86),
}


def build_reference_line(days: int = 20, seed: int = 7) -> LineConfig:
    stations: list[Station] = []
    for i, (tname, zone, instr, scale) in enumerate(_LAYOUT, start=1):
        sid = f"S{i:02d}"
        inspects, dprob = _GATES.get(sid, ((), 0.0))
        # S04 and S10 run redundant weld robots so one can be serviced without
        # stopping the line. S14 diverts failures to an offline rework cell,
        # which returns them to the line OUT OF SEQUENCE.
        par = 2 if sid in ("S04", "S10") else 1
        rw_rate = 0.075 if sid == "S14" else 0.0
        rw_min = 38.0 if sid == "S14" else 0.0
        stations.append(
            Station(
                sid=sid,
                type_name=tname,
                zone=zone,
                index=i,
                # Final assembly runs tight buffers; body and paint have room.
                buffer_capacity=(5 if tname == "oven"
                                 else 1 if zone == "FINAL"
                                 else 3),
                transport_s=round(3.2 + 2.6 * ((i * 37) % 11) / 10.0, 2),
                instrumented=instr,
                cycle_scale=scale,
                parallel=par,
                rework_rate=rw_rate,
                rework_minutes=rw_min,
                inspects=inspects,
                detect_prob=dprob,
            )
        )
    return LineConfig(name="Plant Alpha / Line 2", stations=stations,
                      days=days, seed=seed)


DEFECT_TYPES = (
    "WELD_POROSITY", "SEALANT_VOID", "PAINT_FINISH",
    "LOOSE_FASTENER", "ELECTRICAL_FAULT", "FIT_MISALIGN",
)

# Cost model inputs (illustrative, stated as assumptions in the proposal)
ECONOMICS = {
    "contribution_margin_per_unit_inr": 62_000,
    "rework_cost_per_defect_inr": 4_800,
    "scrap_cost_per_defect_inr": 21_000,
    "warranty_cost_per_escape_inr": 34_000,
    "unplanned_downtime_cost_per_min_inr": 9_400,
    "engineer_hours_per_rca": 22,
    "engineer_cost_per_hour_inr": 1_450,
}


# --------------------------------------------------------------------------
# A second line, to demonstrate rather than assert portability
# --------------------------------------------------------------------------

# "Plant Beta, Line 1" is an older, smaller, more sparsely instrumented line:
# 28 stations, no dedicated paint zone (paint is outsourced), 54% sensor
# coverage against Plant Alpha's 69%, a slower takt and single-shift running.
# It shares nothing with Plant Alpha except the station-type library. Porting
# the twin to it is this block of data and no code at all, which is the
# specific claim Kattenstroth et al. say most simulation models cannot make.

_LAYOUT_BETA: list[tuple[str, str, bool, float]] = [
    ("weld_robot",     "BODY",  True,  1.00),   # B01
    ("manual_fit",     "BODY",  False, 1.04),   # B02  dark
    ("weld_robot",     "BODY",  True,  1.06),   # B03
    ("manual_fit",     "BODY",  False, 1.02),   # B04  dark
    ("sealer",         "BODY",  False, 1.00),   # B05  dark
    ("torque_station", "BODY",  True,  1.02),   # B06
    ("manual_fit",     "BODY",  False, 1.08),   # B07  dark
    ("weld_robot",     "BODY",  True,  1.01),   # B08
    ("manual_fit",     "BODY",  False, 1.00),   # B09  dark
    ("inspection_gate","BODY",  True,  1.00),   # B10  BIW gate
    ("manual_fit",     "FINAL", False, 1.03),   # B11  dark
    ("electrical_fit", "FINAL", True,  1.02),   # B12
    ("torque_station", "FINAL", True,  1.00),   # B13
    ("manual_fit",     "FINAL", False, 1.10),   # B14  dark  <- degrades
    ("electrical_fit", "FINAL", True,  0.99),   # B15
    ("manual_fit",     "FINAL", False, 1.01),   # B16  dark
    ("torque_station", "FINAL", True,  1.05),   # B17  <- torque drift
    ("sealer",         "FINAL", True,  1.02),   # B18
    ("manual_fit",     "FINAL", False, 1.00),   # B19  dark
    ("electrical_fit", "FINAL", True,  1.07),   # B20  <- lot contamination
    ("torque_station", "FINAL", True,  1.01),   # B21
    ("manual_fit",     "FINAL", False, 0.98),   # B22  dark
    ("sealer",         "FINAL", True,  1.00),   # B23
    ("electrical_fit", "FINAL", True,  1.03),   # B24
    ("manual_fit",     "FINAL", False, 1.02),   # B25  dark
    ("torque_station", "FINAL", True,  0.99),   # B26
    ("manual_fit",     "FINAL", False, 1.05),   # B27  dark
    ("inspection_gate","FINAL", True,  1.12),   # B28  final inspection
]

_GATES_BETA = {
    "B10": (("WELD_POROSITY", "SEALANT_VOID"), 0.52),
    "B28": (("WELD_POROSITY", "SEALANT_VOID", "LOOSE_FASTENER",
             "ELECTRICAL_FAULT", "FIT_MISALIGN"), 0.84),
}


def build_compact_line(days: int = 20, seed: int = 11) -> LineConfig:
    """Plant Beta / Line 1. Same code path, different data."""
    stations: list[Station] = []
    for i, (tname, zone, instr, scale) in enumerate(_LAYOUT_BETA, start=1):
        sid = f"B{i:02d}"
        inspects, dprob = _GATES_BETA.get(sid, ((), 0.0))
        stations.append(Station(
            sid=sid, type_name=tname, zone=zone, index=i,
            buffer_capacity=2 if zone == "FINAL" else 3,
            instrumented=instr, cycle_scale=scale,
            transport_s=round(3.6 + 2.2 * ((i * 29) % 9) / 10.0, 2),
            inspects=inspects, detect_prob=dprob,
        ))
    return LineConfig(name="Plant Beta / Line 1", stations=stations,
                      takt_s=72.0, shift_hours=8.0, shifts_per_day=1,
                      days=days, seed=seed, lot_size_units=300)


LINES = {"alpha": build_reference_line, "beta": build_compact_line}
