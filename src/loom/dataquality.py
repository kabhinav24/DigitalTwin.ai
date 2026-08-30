"""
Data quality: what happens when the scans are not clean.

Every claim Loom makes at an un-instrumented station rests on scan timestamps.
That is the load-bearing assumption, so it deserves to be attacked rather than
assumed. Real read points fail in four specific ways:

    clock drift      each reader keeps its own time
    missed reads     a tag passes unseen; the station appears to have no event
    duplicate reads  a tag is seen twice within a second
    late manual scan an operator scans a unit some seconds after it moved

On clock drift, a negative result worth stating: **per-reader offsets are not
statistically identifiable from scan data alone.** The gap between two
consecutive scans is `transport + processing + blocked + starved`, and no
percentile of that distribution separates a reader's offset from the station's
minimum processing time. An estimator that tries accumulates error station by
station and ends up worse than doing nothing - we built one, measured it, and
deleted it. Clock synchronisation is therefore a **deployment prerequisite**
(NTP to within ~50 ms, which is routine on plant networks and negligible against
a 60 s takt), not something this layer pretends to fix. What it does instead is
*detect* drift and refuse to trust a station that shows it.

This module does two things. `corrupt_scans` injects all four so the pipeline
can be tested against dirty input. `assess_and_repair` detects and fixes what
it can, and — more importantly — reports what it could not fix, so downstream
confidence can be reduced rather than silently overstated.

The design rule: a repaired value is never presented with the same confidence
as a clean one. A station whose scans were 8% imputed carries that fact
forward into its soft-sensor confidence and into every alert built on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LineConfig


@dataclass
class DQProfile:
    """How badly to break the scan feed. Defaults are deliberately realistic
    rather than gentle: a plant with clean scans does not need this layer."""

    # residual offset on an NTP-synchronised plant network, not raw free-running
    # clocks. Sync is a prerequisite; see the module docstring.
    clock_drift_sd_s: float = 0.03
    clock_jitter_s: float = 0.35        # per-read noise
    missed_read_rate: float = 0.015
    duplicate_read_rate: float = 0.004
    late_manual_scan_rate: float = 0.02
    late_manual_delay_s: float = 9.0
    seed: int = 41


@dataclass
class DQReport:
    per_station: pd.DataFrame
    n_missing_detected: int
    n_missing_imputed: int
    n_duplicates_removed: int
    clock_offsets_s: np.ndarray
    clock_offset_mae_s: float
    overall_quality: float
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------


def corrupt_scans(cfg: LineConfig, scan_out: np.ndarray,
                  profile: DQProfile | None = None
                  ) -> tuple[np.ndarray, dict]:
    """Return a dirtied copy of the scan matrix plus the truth about the damage.

    NaN marks a missed read. Duplicates cannot be represented in a dense matrix,
    so they are modelled as their observable consequence: a read timestamped
    slightly early, which is what a de-duplicator that keeps the first event
    produces.
    """
    p = profile or DQProfile()
    rng = np.random.default_rng(p.seed)
    n_u, n_st = scan_out.shape
    dirty = scan_out.astype(float).copy()

    # 1. per-reader clock offset, constant for that station
    offsets = rng.normal(0.0, p.clock_drift_sd_s, n_st)
    manual = np.array([not s.instrumented for s in cfg.stations])
    # hand-held readers at manual stations drift more
    offsets = offsets * np.where(manual, 2.2, 1.0)
    dirty += offsets[None, :]

    # 2. per-read jitter
    dirty += rng.normal(0.0, p.clock_jitter_s, dirty.shape)

    # 3. late manual scans: an operator scans after the unit has moved on
    late = (rng.random(dirty.shape) < p.late_manual_scan_rate) & manual[None, :]
    dirty += late * np.abs(rng.normal(p.late_manual_delay_s, 3.0, dirty.shape))

    # 4. duplicate reads kept-first -> timestamp pulled early
    dup = rng.random(dirty.shape) < p.duplicate_read_rate
    dirty -= dup * np.abs(rng.normal(1.2, 0.6, dirty.shape))

    # 5. missed reads
    miss = rng.random(dirty.shape) < p.missed_read_rate
    miss[:, 0] = False              # release point is a schedule, not a scan
    dirty[miss] = np.nan

    return dirty, {
        "true_offsets_s": offsets,
        "missed_mask": miss,
        "duplicate_mask": dup,
        "late_mask": late,
        "n_missed": int(miss.sum()),
        "n_duplicate": int(dup.sum()),
        "n_late": int(late.sum()),
    }


# --------------------------------------------------------------------------
# Detection and repair
# --------------------------------------------------------------------------


def assess_and_repair(cfg: LineConfig, dirty: np.ndarray,
                      release_s: np.ndarray) -> tuple[np.ndarray, DQReport]:
    """Clean what is cleanable and quantify what is not.

    Repairs, in order:

    1. **Duplicates and monotonicity.** A departure earlier than the previous
       unit's departure at the same station is impossible on a serial line;
       those reads are pulled back into order.
    2. **Late scans.** An operator scanning a unit after it has physically moved
       inflates that station's dwell and makes the next station's dwell
       implausibly short - often shorter than any observed clean cycle. Where
       the downstream dwell falls below a robust floor, the upstream scan is
       pulled back to restore a feasible gap.
    3. **Missing reads.** Imputed by linear interpolation between the
       neighbouring stations the unit did register at, which is unbiased on a
       paced line but adds variance - and is recorded as such.

    Clock offsets are *diagnosed*, not corrected; see the module docstring.
    """
    n_u, n_st = dirty.shape
    x = dirty.copy()
    notes: list[str] = []

    missing = np.isnan(x)
    n_missing = int(missing.sum())

    # ---- 1. clock-drift DIAGNOSTIC (not a correction)
    # Compare each link's minimum feasible gap early in the run against late in
    # the run. A reader whose clock has walked shows up as a shifting floor.
    est_offsets = np.zeros(n_st)
    drift_flag = np.zeros(n_st, dtype=bool)
    half = n_u // 2
    for j in range(1, n_st):
        g = x[:, j] - x[:, j - 1]
        a, b = g[:half], g[half:]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 100 or len(b) < 100:
            continue
        shift = float(np.percentile(b, 1.0) - np.percentile(a, 1.0))
        est_offsets[j] = shift
        # A degrading station legitimately shifts this floor too, so the
        # diagnostic cannot separate clock drift from real process change. It
        # is deliberately set loose and reported as "check this reader", never
        # as a correction.
        drift_flag[j] = abs(shift) > 3.0     # seconds

    # ---- 2. duplicates, via per-unit causality
    #
    # An earlier version enforced monotonicity ACROSS UNITS at each station:
    # unit u must depart after unit u-1. That is true only on a strictly serial
    # line, and a rework loop violates it legitimately - a repaired unit
    # re-enters behind vehicles that were once behind it. Enforcing it anyway
    # corrupted every downstream timestamp and drove start-time error to
    # nineteen minutes. The sweep caught it; it is recorded here because the
    # failure is instructive.
    #
    # The repair now uses a constraint that holds regardless of sequencing: a
    # unit cannot leave a station before it has reached that station. That is
    # per-unit, so rework and parallel machines are irrelevant to it.
    n_dup_fixed = 0
    for j in range(1, n_st):
        prev = x[:, j - 1]
        cur = x[:, j]
        ok = np.isfinite(prev) & np.isfinite(cur)
        floor = prev + cfg.stations[j - 1].transport_s
        bad = ok & (cur < floor)
        n_dup_fixed += int(bad.sum())
        cur[bad] = floor[bad]
        x[:, j] = cur

    # ---- 2b. late scans: an infeasibly short downstream dwell means the
    # upstream scan landed late. Pull it back to the feasible floor.
    n_late_fixed = 0
    for j in range(1, n_st):
        gap = x[:, j] - x[:, j - 1]
        ok = np.isfinite(gap)
        if ok.sum() < 200:
            continue
        floor = float(np.percentile(gap[ok], 0.5))
        bad = ok & (gap < floor)
        if bad.any():
            x[bad, j - 1] = x[bad, j] - floor
            n_late_fixed += int(bad.sum())

    # ---- 3. impute missing reads across stations for each unit
    n_imputed = 0
    for u in np.flatnonzero(missing.any(axis=1)):
        row = x[u]
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        j = np.arange(n_st)
        row[~ok] = np.interp(j[~ok], j[ok], row[ok])
        n_imputed += int((~ok).sum())
        x[u] = row
    # any residual gaps: fall back to the station's median dwell
    still = ~np.isfinite(x)
    if still.any():
        for j in range(n_st):
            bad = still[:, j]
            if not bad.any():
                continue
            ref = x[~bad, j]
            x[bad, j] = np.median(ref) if len(ref) else release_s[bad]

    # ---- per-station quality
    rows = []
    for j, st in enumerate(cfg.stations):
        m = float(missing[:, j].mean())
        rows.append({
            "sid": st.sid, "instrumented": st.instrumented,
            "missing_rate": round(m, 5),
            "imputed_rate": round(m, 5),
            "clock_drift_observed_s": round(float(est_offsets[j]), 4),
            "clock_drift_flag": bool(drift_flag[j]),
            # a station is only as trustworthy as the fraction of real reads
            "quality": round(float(np.clip(1.0 - m * 6.0, 0.0, 1.0))
                             * (0.5 if drift_flag[j] else 1.0), 4),
        })
    per_station = pd.DataFrame(rows)

    if n_missing:
        notes.append(f"{n_missing} missed reads imputed; affected stations "
                     f"carry a reduced confidence downstream")
    if n_dup_fixed:
        notes.append(f"{n_dup_fixed} out-of-order or duplicate reads corrected")
    if n_late_fixed:
        notes.append(f"{n_late_fixed} late scans pulled back to a feasible gap")
    if drift_flag.any():
        notes.append(f"clock drift suspected at {int(drift_flag.sum())} readers; "
                     f"these are flagged, not corrected - check NTP sync")

    return x, DQReport(
        per_station=per_station,
        n_missing_detected=n_missing,
        n_missing_imputed=n_imputed,
        n_duplicates_removed=n_dup_fixed,
        clock_offsets_s=est_offsets,
        clock_offset_mae_s=float("nan"),
        overall_quality=float(per_station["quality"].mean()),
        notes=notes,
    )


def score_repair(report: DQReport, truth: dict) -> dict:
    """Did the repair find the damage that was actually done?"""
    return {
        "missed_injected": int(truth["n_missed"]),
        "missed_detected": int(report.n_missing_detected),
        "missed_detection_rate": round(
            report.n_missing_detected / max(truth["n_missed"], 1), 4),
        "duplicates_injected": int(truth["n_duplicate"]),
        "duplicates_corrected": int(report.n_duplicates_removed),
        "late_injected": int(truth["n_late"]),
        "drift_readers_flagged": int(
            report.per_station["clock_drift_flag"].sum()),
    }


def apply_quality_to_confidence(profile_df: pd.DataFrame,
                                dq: DQReport) -> pd.DataFrame:
    """Downgrade soft-sensor confidence by data quality.

    A cycle-time estimate built on 8% imputed scans is not as good as one built
    on clean scans, and saying so is the whole point of carrying a confidence
    number at all.
    """
    q = dq.per_station.set_index("sid")["quality"]
    out = profile_df.copy()
    out["dq_quality"] = out["sid"].map(q).fillna(1.0)
    out["confidence_raw"] = out["confidence"]
    out["confidence"] = (out["confidence"] * out["dq_quality"]).round(4)
    return out


# --------------------------------------------------------------------------
# Robustness sweeps
# --------------------------------------------------------------------------


def sweep_read_rate(build_states, score_fn,
                    rates: tuple[float, ...] = (0.0, 0.01, 0.03, 0.06, 0.12)
                    ) -> pd.DataFrame:
    """Accuracy as a function of missed-read rate.

    Industrial RFID read rates vary enormously with antenna placement, tag
    orientation and how much metal is nearby, and a single quoted figure would
    be invented. So Loom reports the whole curve instead: a plant measures its
    own read rate and looks up what to expect, rather than trusting a number
    from someone else's factory.
    """
    rows = []
    for r in rates:
        prof = DQProfile(missed_read_rate=r)
        states, dq = build_states(prof)
        m = score_fn(states)
        rows.append({"missed_read_rate": r, **m,
                     "feed_quality": round(dq.overall_quality, 4)
                     if dq is not None else 1.0})
    return pd.DataFrame(rows)


def sweep_clock_sync(build_states, score_fn,
                     drifts_s: tuple[float, ...] = (0.03, 0.25, 1.0, 3.0, 8.0)
                     ) -> pd.DataFrame:
    """Accuracy as clock synchronisation degrades.

    Loom requires NTP to roughly 50 ms. This quantifies what the requirement is
    actually worth, so a plant with free-running reader clocks can see the cost
    before deciding whether to fix it or to accept the wider error bars.
    """
    rows = []
    for d in drifts_s:
        prof = DQProfile(clock_drift_sd_s=d, missed_read_rate=0.0,
                         duplicate_read_rate=0.0, late_manual_scan_rate=0.0)
        states, dq = build_states(prof)
        m = score_fn(states)
        rows.append({"clock_drift_sd_s": d, **m})
    return pd.DataFrame(rows)
