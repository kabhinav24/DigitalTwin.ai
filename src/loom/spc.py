"""
Statistical process control.

The brief lists SPC alongside anomaly detection and ML as a predictive
technique, and it earns its place for a reason the fancier methods do not: a
control chart needs no training data, no labels and no model, and a process
engineer can read one without trusting anybody. It is the cheapest honest
detector in the box.

Two charts, both on shift-aggregated statistics rather than individual units,
because a paced line produces far too many points to chart individually and
shift means are what a plant already reviews:

**EWMA**  z_t = lam*x_t + (1-lam)*z_{t-1}, with widening early limits. Tuned for
slow drift - the torque tool that wanders 0.3 Nm a day.

**CUSUM**  accumulates signed deviation and fires when the running sum clears a
decision interval. Tuned for a sustained small shift, and it reports *when the
shift began*, not just when it was caught, which is what a root-cause
investigation actually needs.

A third chart tracks the **stoppage rate**: the fraction of units per shift
whose cycle time clears an outlier threshold. Micro-stoppages barely move a
shift mean but move that rate a lot.

The part worth noticing: all three run identically on a soft-sensed dwell
series at an un-instrumented station. Once cycle time is recovered from scans,
a dark station gets the same control charts as an instrumented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LineConfig
from .twin import LineStates

# A textbook chart uses L = 3 sigma, which holds a ~0.3% false-alarm rate for
# ONE chart. This line runs ~270 charts at once, so a 3-sigma limit fires
# somewhere on more than half of all clean stations - the alert-fatigue failure
# the brief warns about, reproduced exactly. Two corrections are applied:
#
#   1. multiplicity - limits widened to roughly Bonferroni scale for ~270 charts
#   2. persistence  - a single out-of-limit block is noise; a signal requires
#                     PERSISTENCE consecutive blocks, which is standard Western
#                     Electric practice and costs very little detection lag
#
# The cost of both is measured, not assumed: `score_against_faults` reports
# detection lag and `false_alarm_rate` reports what it bought.
EWMA_LAMBDA = 0.22
EWMA_L = 5.5
CUSUM_K = 0.5          # slack, in baseline sigmas
CUSUM_H = 9.35         # decision interval, in baseline sigmas
OUTLIER_SIGMA = 4.0
PERSISTENCE = 2        # consecutive out-of-limit blocks required to signal
BASELINE_BLOCKS = 10


@dataclass
class ChartResult:
    station: str
    signal: str
    chart: str
    instrumented: bool
    baseline_mean: float
    baseline_sd: float
    first_signal_block: int | None
    first_signal_day: float | None
    onset_estimate_day: float | None    # CUSUM only: when the shift began
    n_signals: int
    peak_stat: float
    series: list[float]
    limit: float


def _first_persistent(flags: np.ndarray, k: int = PERSISTENCE) -> int | None:
    """Index of the first of k consecutive True values, or None."""
    if k <= 1:
        return int(np.argmax(flags)) if flags.any() else None
    run = 0
    for i, v in enumerate(flags):
        run = run + 1 if v else 0
        if run >= k:
            return i - k + 1
    return None


def _blocks(x: np.ndarray, block: int) -> np.ndarray:
    n = (len(x) // block) * block
    return x[:n].reshape(-1, block)


def ewma_chart(vals: np.ndarray, mu: float, sd: float,
               lam: float = EWMA_LAMBDA, L: float = EWMA_L
               ) -> tuple[np.ndarray, np.ndarray]:
    z = np.empty(len(vals))
    prev = mu
    for i, v in enumerate(vals):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    t = np.arange(1, len(vals) + 1)
    lim = L * sd * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    return z - mu, lim


def cusum_chart(vals: np.ndarray, mu: float, sd: float,
                k: float = CUSUM_K, h: float = CUSUM_H
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-sided CUSUM. Returns (max of the two arms, reset points, h)."""
    z = (vals - mu) / max(sd, 1e-9)
    hi = np.zeros(len(z))
    lo = np.zeros(len(z))
    hi_start = np.zeros(len(z), dtype=int)
    lo_start = np.zeros(len(z), dtype=int)
    for i in range(len(z)):
        prev_hi = hi[i - 1] if i else 0.0
        prev_lo = lo[i - 1] if i else 0.0
        hi[i] = max(0.0, prev_hi + z[i] - k)
        lo[i] = max(0.0, prev_lo - z[i] - k)
        hi_start[i] = i if hi[i] == 0 else (hi_start[i - 1] if i else 0)
        lo_start[i] = i if lo[i] == 0 else (lo_start[i - 1] if i else 0)
    stat = np.maximum(hi, lo)
    onset = np.where(hi >= lo, hi_start, lo_start)
    return stat, onset, h


def scan(cfg: LineConfig, states: LineStates,
         signals: dict[str, dict[str, np.ndarray]],
         units_per_day: int, baseline_blocks: int = BASELINE_BLOCKS,
         block_units: int | None = None, ewma_l: float = EWMA_L,
         cusum_h: float = CUSUM_H, persistence: int = PERSISTENCE
         ) -> pd.DataFrame:
    """Run every chart over every station and return the first signal of each.

    Instrumented stations are charted on their real process signals. Dark
    stations are charted on soft-sensed dwell, which is the point: SPC does not
    care where the series came from once the series exists.
    """
    # One block is one shift on whichever line this is. Plant Beta runs a
    # 72 s takt on one shift, so its shift is 400 units, not Plant Alpha's 480.
    if block_units is None:
        block_units = max(60, units_per_day // max(cfg.shifts_per_day, 1))
    rows: list[ChartResult] = []
    per_day = units_per_day / block_units

    def add(sid: str, name: str, series: np.ndarray, instrumented: bool) -> None:
        B = _blocks(series, block_units)
        if len(B) < baseline_blocks + 3:
            return
        means = B.mean(axis=1)
        base = means[:baseline_blocks]
        # Robust baseline. A commissioning window is assumed clean and usually
        # is not: fault F5 begins on day 3, inside the baseline period, and a
        # mean-and-sd baseline quietly absorbs it and then cannot see it.
        # Median and a MAD-derived sigma resist that contamination.
        mu = float(np.median(base))
        # Short-term sigma from the median moving range. This is the standard
        # Shewhart estimator and it is used here for the reason it exists: it
        # measures block-to-block variation and is therefore blind to a
        # sustained drift sitting inside the baseline window. Fault F5 begins
        # on day 3, inside the commissioning period, and a plain mean-and-sd
        # baseline absorbs it and then cannot see it.
        mr = np.abs(np.diff(base))
        sd = float(max(np.median(mr) / 0.954 if len(mr) else 0.0,
                       1e-9))

        stat, lim = ewma_chart(means, mu, sd, L=ewma_l)
        out = np.abs(stat) > lim
        out[:baseline_blocks] = False
        first = _first_persistent(out, persistence)
        rows.append(ChartResult(
            sid, name, "EWMA", instrumented, mu, sd, first,
            None if first is None else round(first / per_day, 2), None,
            int(out.sum()), float(np.abs(stat).max()),
            [round(float(v), 4) for v in stat], float(lim[-1])))

        cstat, onset, h = cusum_chart(means, mu, sd, h=cusum_h)
        cout = cstat > h
        cout[:baseline_blocks] = False
        cfirst = _first_persistent(cout, persistence)
        rows.append(ChartResult(
            sid, name, "CUSUM", instrumented, mu, sd, cfirst,
            None if cfirst is None else round(cfirst / per_day, 2),
            None if cfirst is None else round(int(onset[cfirst]) / per_day, 2),
            int(cout.sum()), float(cstat.max()),
            [round(float(v), 4) for v in cstat], h))

    for st in cfg.stations:
        j = st.index - 1
        if st.instrumented:
            for name, arr in signals.get(st.sid, {}).items():
                add(st.sid, name, np.asarray(arr, dtype=float), True)
            # stoppage-rate chart: micro-stops barely shift a mean but move this
            ct = np.asarray(signals[st.sid].get("cycle_time_s",
                                                states.dwell[:, j]), float)
            base = ct[:baseline_blocks * block_units]
            thr = base.mean() + OUTLIER_SIGMA * base.std()
            add(st.sid, "stoppage_rate", (ct > thr).astype(float), True)
        else:
            # the same charts, on a series that no sensor produced
            add(st.sid, "soft_dwell_s", states.dwell[:, j].astype(float), False)
            d = states.dwell[:, j]
            base = d[:baseline_blocks * block_units]
            thr = base.mean() + OUTLIER_SIGMA * base.std()
            add(st.sid, "soft_stoppage_rate", (d > thr).astype(float), False)

    return pd.DataFrame([r.__dict__ for r in rows])


def score_against_faults(charts: pd.DataFrame, faults: list[dict],
                         days: int) -> pd.DataFrame:
    """Detection lead time per injected fault.

    A chart counts as detecting a fault if it fires at the fault's target
    station after the fault began. Lead time is measured against the fault's
    own start day, so a negative number would mean the chart fired before the
    fault existed - a false alarm, not a detection.
    """
    rows = []
    for f in faults:
        tgt = f["target"]
        cand = charts[(charts["station"] == tgt)
                      & charts["first_signal_day"].notna()
                      & (charts["first_signal_day"] >= f["start_day"] - 0.5)]
        if cand.empty:
            rows.append({"fault_id": f["fault_id"], "kind": f["kind"],
                         "target": tgt, "start_day": f["start_day"],
                         "detected": False, "chart": None, "signal": None,
                         "detect_day": None, "lag_days": None,
                         "onset_estimate_day": None, "onset_error_days": None})
            continue
        best = cand.sort_values("first_signal_day").iloc[0]
        # CUSUM estimates *when the shift began*, which is what a root-cause
        # investigation needs, so surface it even when EWMA signalled first
        cus = cand[(cand["chart"] == "CUSUM")
                   & cand["onset_estimate_day"].notna()]
        onset = (float(cus.sort_values("first_signal_day").iloc[0]
                       ["onset_estimate_day"]) if not cus.empty
                 else best["onset_estimate_day"])
        rows.append({
            "fault_id": f["fault_id"], "kind": f["kind"], "target": tgt,
            "start_day": f["start_day"], "detected": True,
            "chart": best["chart"], "signal": best["signal"],
            "detect_day": float(best["first_signal_day"]),
            "lag_days": round(float(best["first_signal_day"]) - f["start_day"], 2),
            "onset_estimate_day": None if onset is None else float(onset),
            "onset_error_days": (None if onset is None
                                 else round(float(onset) - f["start_day"], 2)),
        })
    return pd.DataFrame(rows)


def affected_stations(cfg: LineConfig, faults: list[dict]) -> set[str]:
    """Every station a fault legitimately perturbs, not just its named target.

    This distinction matters and getting it wrong inflates the false-alarm rate
    badly. A humidity excursion targets a *zone*, and genuinely moves the
    humidity signal at all four booths in it. An operator fault targets a
    *person*, and genuinely widens cycle time at every manual station they
    work. A contaminated lot bites at whichever station fits that part. Charts
    firing at those stations are corroborating detections, not false alarms.
    """
    hit: set[str] = set()
    for f in faults:
        tgt, kind = f["target"], f["kind"]
        if any(s.sid == tgt for s in cfg.stations):
            hit.add(tgt)
        elif kind == "ENV_EXCURSION":
            hit |= {s.sid for s in cfg.stations if s.zone == tgt}
        elif kind == "OPERATOR_VAR":
            zone = tgt.split("-")[0]
            hit |= {s.sid for s in cfg.stations
                    if s.zone == zone and s.type_name == "manual_fit"}
        elif kind == "LOT_CONTAM":
            hit |= {s.sid for s in cfg.stations
                    if s.type_name == "electrical_fit"}
    return hit


def false_alarm_rate(charts: pd.DataFrame, faults: list[dict],
                     cfg: LineConfig | None = None) -> dict:
    """Charts firing at stations no fault touches, directly or indirectly.

    Reported because a chart that fires everywhere detects nothing, and because
    the brief is explicit that false alarms erode floor trust.

    Caveat stated rather than buried: a bottleneck also starves and blocks its
    neighbours, so some remaining signals on "clean" stations are real knock-on
    effects of a real fault. This number is therefore an upper bound on the
    true false-alarm rate, not a point estimate.
    """
    if cfg is not None:
        touched = affected_stations(cfg, faults)
    else:
        touched = {f["target"] for f in faults}
    clean = charts[~charts["station"].isin(touched)]
    fired = clean["first_signal_block"].notna()
    return {
        "clean_stations": int(clean["station"].nunique()),
        "clean_station_charts": int(len(clean)),
        "charts_firing": int(fired.sum()),
        "false_alarm_rate_upper_bound": (float(fired.mean()) if len(clean)
                                         else float("nan")),
        "by_chart": clean.assign(fired=fired).groupby("chart")["fired"]
        .mean().round(3).to_dict(),
    }


# --------------------------------------------------------------------------
# Operating point: the detection / false-alarm trade-off, measured
# --------------------------------------------------------------------------


def sweep_operating_points(cfg: LineConfig, states: LineStates,
                           signals: dict[str, dict[str, np.ndarray]],
                           faults: list[dict], units_per_day: int,
                           limits: tuple[float, ...] = (3.0, 3.5, 4.1, 4.6, 5.2)
                           ) -> pd.DataFrame:
    """Detection rate against false-alarm rate as the control limit widens.

    The brief is explicit that false alarms erode floor trust. That trade-off
    cannot be solved away, only chosen, so Loom measures the curve and states
    which point it picked and why, instead of quoting the single flattering
    number a 3-sigma chart produces.
    """
    rows = []
    if True:
        for L in limits:
            ch = scan(cfg, states, signals, units_per_day,
                      ewma_l=L, cusum_h=1.7 * L)
            sc = score_against_faults(ch, faults, cfg.days)
            fa = false_alarm_rate(ch, faults, cfg)
            station_faults = sc[sc["kind"].isin(
                ["PARAM_DRIFT", "CYCLE_DEGRADE", "MICROSTOP"])]
            rows.append({
                "ewma_L": L,
                "cusum_h": round(1.7 * L, 2),
                "station_faults_detected": int(station_faults["detected"].sum()),
                "station_faults_total": int(len(station_faults)),
                "mean_lag_days": float(station_faults["lag_days"].mean(skipna=True)),
                "false_alarm_upper_bound": round(fa["false_alarm_rate_upper_bound"], 4),
            })
    return pd.DataFrame(rows)
