"""
Bottleneck detection and prediction.

Detection follows the active-period method (Roser, Nakano & Tanaka 2001, 2002),
which Ragazzini et al. (2024) select for three reasons that also apply here: it
is data-driven, it works equally well on real and synthetic state logs, and it
depends only on machine state, never on the machine's physics. That last
property is what lets Loom apply it to stations that have no sensors at all -
their states are reconstructed from scans.

Prediction reuses the same detector, run over the *synthetic future* state log
produced by the rolling-horizon twin. Where Ragazzini's DT is a commercial
plant-simulation model fed by PLC data, Loom's is a config-generated kernel fed
by soft-sensor estimates, so sensor-poor stations are represented rather than
silently dropped.

Active vs inactive
------------------
Active   : working, or stopped in a way that makes others wait (micro-stop).
Inactive : starved (waiting for a part) or blocked (waiting for a slot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import LineConfig
from .twin import LineStates, SoftSensor

MERGE_GAP_S = 2.0    # gaps shorter than this do not break an active period


# --------------------------------------------------------------------------
# Processing-time imputation (needed to place the working interval)
# --------------------------------------------------------------------------


def impute_processing(cfg: LineConfig, states: LineStates, sensor: SoftSensor,
                      lo: int, hi: int) -> np.ndarray:
    """Per-unit processing time over a window, for every station.

    Uncensored units give processing time exactly. For blocked units the twin
    imputes E[p | p < theta] under the station's fitted lognormal, which is the
    conditional mean given everything that was actually observed.
    """
    n_st = cfg.n_stations
    out = np.empty((hi - lo, n_st))
    dwell = states.dwell[lo:hi]
    theta = states.theta[lo:hi]
    cens = states.blocked_flag[lo:hi]

    for j, st in enumerate(cfg.stations):
        p = dwell[:, j].copy()
        c = cens[:, j]
        if c.any():
            e = sensor.estimate(st.sid, lo, hi)
            if np.isfinite(e.mean_s) and np.isfinite(e.sd_s) and e.sd_s > 0:
                s2 = np.log(1 + (e.sd_s / e.mean_s) ** 2)
                s = np.sqrt(s2)
                mu = np.log(e.mean_s) - 0.5 * s2
                th = np.clip(theta[c, j], 1e-6, None)
                z = (np.log(th) - mu) / s
                denom = np.clip(stats.norm.cdf(z), 1e-9, 1.0)
                # E[X | X < th] for lognormal
                p[c] = np.exp(mu + 0.5 * s2) * stats.norm.cdf(z - s) / denom
            else:
                p[c] = np.minimum(dwell[c, j], theta[c, j])
        out[:, j] = np.clip(p, 0.1, None)
    return out


# --------------------------------------------------------------------------
# Active periods
# --------------------------------------------------------------------------


def active_periods(start: np.ndarray, proc: np.ndarray,
                   merge_gap_s: float = MERGE_GAP_S) -> list[np.ndarray]:
    """Merge consecutive working intervals at each station into active periods.

    Returns one (n_periods, 2) array of [t_start, t_end] per station.
    """
    n_u, n_st = start.shape
    out = []
    for j in range(n_st):
        s = start[:, j]
        f = s + proc[:, j]
        order = np.argsort(s)
        s, f = s[order], f[order]
        # a new period begins when the next unit starts after the previous
        # finished (plus tolerance) - i.e. the station went idle in between
        brk = np.empty(n_u, dtype=bool)
        brk[0] = True
        brk[1:] = s[1:] > (np.maximum.accumulate(f)[:-1] + merge_gap_s)
        gid = np.cumsum(brk) - 1
        n_g = gid[-1] + 1
        p_start = np.full(n_g, np.inf)
        p_end = np.zeros(n_g)
        np.minimum.at(p_start, gid, s)
        np.maximum.at(p_end, gid, f)
        out.append(np.column_stack([p_start, p_end]))
    return out


def _covering_duration(periods: np.ndarray, t: np.ndarray) -> np.ndarray:
    """For each time in t, the length of the active period covering it (0 if none)."""
    if len(periods) == 0:
        return np.zeros_like(t)
    st, en = periods[:, 0], periods[:, 1]
    idx = np.searchsorted(st, t, side="right") - 1
    ok = (idx >= 0) & (en[np.clip(idx, 0, len(en) - 1)] >= t)
    dur = np.zeros_like(t)
    good = np.clip(idx, 0, len(en) - 1)
    dur[ok] = (en - st)[good][ok]
    return dur


@dataclass
class BottleneckResult:
    t_grid: np.ndarray
    bottleneck_idx: np.ndarray        # index into cfg.stations, -1 if idle line
    shifting: np.ndarray              # bool per grid point
    share: pd.DataFrame               # per-station sole / shifting / total share
    top: str                          # dominant bottleneck over the window


def detect_bottleneck(cfg: LineConfig, start: np.ndarray, proc: np.ndarray,
                      t0: float | None = None, t1: float | None = None,
                      resolution_s: float = 60.0) -> BottleneckResult:
    """Momentary bottleneck over a time grid, plus sole/shifting classification."""
    periods = active_periods(start, proc)
    t0 = float(start.min()) if t0 is None else t0
    t1 = float((start + proc).max()) if t1 is None else t1
    t = np.arange(t0, t1, resolution_s)
    if len(t) == 0:
        t = np.array([t0])

    dur = np.stack([_covering_duration(p, t) for p in periods], axis=1)
    best = dur.argmax(axis=1)
    best_val = dur.max(axis=1)
    idle = best_val <= 0
    best[idle] = -1

    # A moment is "shifting" when the runner-up station is also active and its
    # active period is within 15% of the leader's - i.e. the bottleneck is
    # handing over rather than sitting firmly on one station.
    part = np.partition(dur, -2, axis=1)
    second = part[:, -2]
    shifting = (~idle) & (second > 0) & (second >= 0.85 * best_val)

    rows = []
    for j, st in enumerate(cfg.stations):
        is_bn = best == j
        rows.append({
            "sid": st.sid, "zone": st.zone, "instrumented": st.instrumented,
            "sole_share": float((is_bn & ~shifting).mean()),
            "shifting_share": float((is_bn & shifting).mean()),
            "total_share": float(is_bn.mean()),
        })
    share = pd.DataFrame(rows).sort_values("total_share", ascending=False)
    top = share.iloc[0]["sid"] if share.iloc[0]["total_share"] > 0 else "NONE"

    return BottleneckResult(t_grid=t, bottleneck_idx=best, shifting=shifting,
                            share=share, top=top)


# --------------------------------------------------------------------------
# Prediction over a rolling horizon
# --------------------------------------------------------------------------


@dataclass
class BottleneckForecast:
    trigger_unit: int
    trigger_time_s: float
    horizon_units: int
    predicted: str
    predicted_confidence: float
    present: str            # bottleneck detected over the trailing window
    static: str             # long-run bottleneck from all history so far
    runtime_ms: float


def forecast_bottleneck(cfg: LineConfig, states: LineStates, sensor: SoftSensor,
                        trigger_unit: int, horizon_units: int = 480,
                        lookback_units: int = 480, replications: int = 5) -> BottleneckForecast:
    """One rolling-horizon cycle: sync, simulate forward, detect on the future.

    Mirrors the four blocks of Ragazzini et al. (2024) - data processing, DT,
    bottleneck prediction, decision support - with the DT block re-parameterised
    from soft sensors each cycle so it stays current without manual re-modelling
    (the "keeping the DES up to date" challenge in Kattenstroth et al. 2024).
    """
    import time
    from .twin import RollingHorizonTwin

    t_start = time.perf_counter()
    lo = max(0, trigger_unit - lookback_units)

    # Repeated replications: one stochastic run of a balanced line is noisy, so
    # the forecast is the modal bottleneck across replications and the vote
    # share doubles as the forecast's confidence.
    twin = RollingHorizonTwin(cfg, states, sensor)
    votes: dict[str, float] = {}
    for r in range(replications):
        twin.rng = np.random.default_rng(1000 + r)
        hz = twin.run(trigger_unit, horizon_units, lookback_units=lookback_units)
        fut = detect_bottleneck(cfg, hz.start, hz.finish - hz.start, resolution_s=30.0)
        for row in fut.share.head(3).itertuples():
            votes[row.sid] = votes.get(row.sid, 0.0) + row.total_share
    total = sum(votes.values()) or 1.0
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    pred_sid, pred_w = ranked[0]
    conf = pred_w / total

    past_proc = impute_processing(cfg, states, sensor, lo, trigger_unit)
    present = detect_bottleneck(cfg, states.start[lo:trigger_unit], past_proc,
                                resolution_s=60.0)
    static = detect_bottleneck(cfg, states.start[:trigger_unit],
                               impute_processing(cfg, states, sensor, 0, trigger_unit),
                               resolution_s=180.0)

    return BottleneckForecast(
        trigger_unit=trigger_unit,
        trigger_time_s=float(states.start[trigger_unit, 0]),
        horizon_units=horizon_units,
        predicted=pred_sid,
        predicted_confidence=float(conf),
        present=present.top,
        static=static.top,
        runtime_ms=(time.perf_counter() - t_start) * 1000,
    )


def evaluate_forecasts(cfg: LineConfig, states: LineStates, sensor: SoftSensor,
                       truth_proc: np.ndarray, triggers: list[int],
                       horizon_units: int = 480) -> pd.DataFrame:
    """Score prediction against what actually happened, alongside two baselines.

    This reproduces the None / Present / Future comparison in Ragazzini et al.
    on a line where a third of the stations have no sensors.
    """
    rows = []
    for u in triggers:
        fc = forecast_bottleneck(cfg, states, sensor, u, horizon_units)
        hi = min(u + horizon_units, states.start.shape[0])
        actual = detect_bottleneck(cfg, states.start[u:hi], truth_proc[u:hi],
                                   resolution_s=30.0)
        act_top2 = set(actual.share.head(2)["sid"])
        rows.append({
            "trigger_unit": u,
            "day": round(float(states.start[u, 0]) / 86400.0, 2),
            "actual": actual.top,
            "predicted": fc.predicted,
            "present": fc.present,
            "static": fc.static,
            "pred_correct": fc.predicted == actual.top,
            "present_correct": fc.present == actual.top,
            "static_correct": fc.static == actual.top,
            "pred_in_top2": fc.predicted in act_top2,
            "present_in_top2": fc.present in act_top2,
            "confidence": round(fc.predicted_confidence, 3),
            "runtime_ms": round(fc.runtime_ms, 1),
        })
    return pd.DataFrame(rows)
