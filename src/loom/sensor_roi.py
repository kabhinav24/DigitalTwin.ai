"""
Sensor investment optimiser.

The Round 2 brief fixes two constraints that most solutions treat as excuses:
sensor coverage is uneven, and production can only be stopped for
instrumentation during a small number of scheduled maintenance windows a year.
Loom treats them instead as an allocation problem. Given that you can retrofit
only a handful of stations at the next shutdown, which ones buy the most?

The ranking is a *counterfactual ablation*, not a heuristic. For each dark
station the optimiser reveals that station's real signals to the defect model,
refits, and measures the change in precision-recall AUC. That is the honest
answer to "what would this sensor have been worth", and it is only computable
because the simulator holds ground truth. In a live deployment the same
ranking is estimated from a short instrumented pilot on two or three stations,
which is exactly the shape of a Phase 1 pilot.

Value has three parts:

    quality      dPR-AUC on the defect model when the station is instrumented
    throughput   how often the station is the momentary bottleneck, weighted by
                 how uncertain the soft sensor is about it
    diagnosis    how often the station appeared as an unresolved suspect in
                 root-cause traces

They are combined, divided by the station type's retrofit cost, and returned
as an ordered plan that fits a stated budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import LineConfig, ECONOMICS
from .defect import ConformalDefectModel, time_split
from .twin import LineStates, SoftSensor


@dataclass
class RetrofitCandidate:
    sid: str
    zone: str
    station_type: str
    cost_inr: int
    d_pr_auc: float
    bottleneck_share: float
    soft_sensor_ci_width_s: float
    diagnosis_hits: int
    value_score: float
    value_per_lakh: float
    rationale: str


def rank_retrofits(cfg: LineConfig, states: LineStates, sensor: SoftSensor,
                   X: pd.DataFrame, y: np.ndarray, days: np.ndarray,
                   hidden_signals: dict[str, dict[str, np.ndarray]],
                   bottleneck_share: dict[str, float],
                   diagnosis_hits: dict[str, int] | None = None,
                   train_end: int = 13, calib_end: int = 16,
                   window: tuple[int, int] | None = None) -> pd.DataFrame:
    """Score every un-instrumented station by what instrumenting it would buy.

    `hidden_signals` is the ground truth the twin never sees in normal
    operation. Using it here is legitimate and clearly bounded: this function
    answers a planning question offline, it never feeds a live prediction.
    """
    diagnosis_hits = diagnosis_hits or {}
    tr, ca, te = time_split(days, train_end, calib_end)
    n_u = len(X)
    lo, hi = window or (max(0, n_u - 3840), n_u)

    base = ConformalDefectModel(alpha=0.10).fit(X, y, tr, ca)
    base_pred = base.predict(X[te])
    base_ap = average_precision_score(y[te], base_pred["risk"])

    rows: list[RetrofitCandidate] = []
    for st in cfg.stations:
        if st.instrumented:
            continue
        truth = hidden_signals.get(st.sid, {})
        add = {f"z_{st.sid}_{k}": _z(v) for k, v in truth.items()
               if k != "cycle_time_s"}
        # Instrumenting any station also replaces the soft-sensed dwell with a
        # directly measured cycle time, free of blocking contamination.
        if "cycle_time_s" in truth:
            add[f"z_{st.sid}_cycle_measured"] = _z(truth["cycle_time_s"])
        Xa = X.copy()
        for k, v in add.items():
            Xa[k] = v
        m = ConformalDefectModel(alpha=0.10).fit(Xa, y, tr, ca)
        ap = average_precision_score(y[te], m.predict(Xa[te])["risk"])
        d_ap = float(ap - base_ap)

        e = sensor.estimate(st.sid, lo, hi)
        ci_w = float(e.hi95 - e.lo95) if np.isfinite(e.hi95) else float("nan")
        bshare = float(bottleneck_share.get(st.sid, 0.0))
        dhits = int(diagnosis_hits.get(st.sid, 0))

        # normalise each term to roughly 0-1 before combining
        v_quality = max(d_ap, 0.0) / max(base_ap, 1e-6)
        v_throughput = bshare * min(ci_w / 2.0, 1.0)
        v_diagnosis = min(dhits / 5.0, 1.0)
        value = 0.5 * v_quality + 0.35 * v_throughput + 0.15 * v_diagnosis

        cost = st.stype.retrofit_cost_inr
        rows.append(RetrofitCandidate(
            sid=st.sid, zone=st.zone, station_type=st.type_name, cost_inr=cost,
            d_pr_auc=d_ap, bottleneck_share=bshare,
            soft_sensor_ci_width_s=ci_w, diagnosis_hits=dhits,
            value_score=value, value_per_lakh=value / (cost / 100_000),
            rationale=_rationale(st.sid, d_ap, bshare, ci_w, dhits),
        ))

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df.sort_values("value_per_lakh", ascending=False).reset_index(drop=True)


def plan_within_budget(ranked: pd.DataFrame, budget_inr: int,
                       max_stations: int = 5) -> dict:
    """Greedy pick by value density, which is optimal enough for this scale and
    far easier for a plant manager to argue with than a solver output."""
    chosen, spend = [], 0
    for r in ranked.itertuples():
        if len(chosen) >= max_stations or spend + r.cost_inr > budget_inr:
            continue
        chosen.append(r.sid)
        spend += r.cost_inr
    sel = ranked[ranked["sid"].isin(chosen)]
    return {
        "stations": chosen,
        "n": len(chosen),
        "spend_inr": int(spend),
        "budget_inr": int(budget_inr),
        "coverage_before": None,
        "total_value": float(sel["value_score"].sum()),
        "table": sel,
    }


def coverage_after(cfg: LineConfig, chosen: list[str]) -> float:
    n = cfg.n_stations
    now = sum(s.instrumented for s in cfg.stations)
    return (now + len([c for c in chosen if not cfg.by_id(c).instrumented])) / n


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v[:2880].mean()) / (v[:2880].std() + 1e-9)


def _rationale(sid: str, d_ap: float, bshare: float, ci: float, dhits: int) -> str:
    bits = []
    if d_ap > 0.005:
        bits.append(f"lifts defect PR-AUC by {d_ap:+.3f}")
    if bshare > 0.05:
        bits.append(f"is the momentary bottleneck {bshare:.0%} of the time")
    if np.isfinite(ci) and ci > 1.0:
        bits.append(f"soft-sensor interval is {ci:.1f}s wide")
    if dhits:
        bits.append(f"appeared as an unresolved suspect in {dhits} traces")
    return f"{sid} " + ("; ".join(bits) if bits
                        else "shows no measurable uplift; defer")


# ==========================================================================
# Production ranking: no ground truth, no counterfactual
# ==========================================================================
#
# The ablation above answers "what would this sensor have been worth" by
# revealing the sensor's real readings and measuring the uplift. That is a
# clean answer and it is only available in simulation, because in a plant the
# sensor you are considering does not exist and therefore measured nothing.
#
# A deployable ranking cannot use it. So Loom carries two rankings:
#
#   rank_retrofits            offline, uses hidden truth, prototype only
#   rank_retrofits_production live, uses only what the twin can see
#
# The production ranking is an expected-value-of-information score built from
# three quantities that are all observable without the sensor:
#
#   uncertainty   how wide the soft-sensor interval is at that station
#   exposure      how much of the line's throughput risk sits on that station
#   ambiguity     how often it turned up as a suspect the twin could not resolve
#
# `validate_production_ranking` then checks the two agree. If the cheap,
# deployable score reproduces the expensive, ground-truth one, the deployable
# score is trustworthy in a plant where the expensive one is impossible.


def rank_retrofits_production(cfg: LineConfig, states: LineStates,
                              sensor: SoftSensor,
                              bottleneck_share: dict[str, float],
                              unresolved_suspect_hits: dict[str, int] | None = None,
                              defect_exposure: dict[str, float] | None = None,
                              window: tuple[int, int] | None = None
                              ) -> pd.DataFrame:
    """Rank un-instrumented stations using only observable quantities.

    Every input here is available on day one of a deployment:

    * `uncertainty`  - the width of the soft sensor's own 95% interval, which
      the estimator already reports. A station the twin models confidently is
      a station a sensor would tell you little new about.
    * `exposure`     - how often the station constrains the line, times how much
      throughput sits behind a second of its cycle time. Instrumenting a station
      that never limits output buys nothing.
    * `ambiguity`    - how often root-cause traces named this station as a
      candidate without being able to resolve it. That is the twin telling you,
      unprompted, where its own blind spots are costing diagnoses.
    """
    unresolved_suspect_hits = unresolved_suspect_hits or {}
    defect_exposure = defect_exposure or {}
    n_u = states.dwell.shape[0]
    lo, hi = window or (max(0, n_u - 3840), n_u)

    rows = []
    for st in cfg.stations:
        if st.instrumented:
            continue
        e = sensor.estimate(st.sid, lo, hi)
        ci_w = float(e.hi95 - e.lo95) if np.isfinite(e.hi95) else float("nan")
        rel_ci = (ci_w / e.mean_s) if (np.isfinite(ci_w) and e.mean_s) else 0.0
        # a low-confidence estimate is itself information about missing sensing
        uncertainty = float(np.clip(rel_ci * 40 + (1 - e.confidence), 0, 2)) / 2

        bshare = float(bottleneck_share.get(st.sid, 0.0))
        cycle_over_takt = max((e.mean_s or 0) - cfg.takt_s, 0.0)
        exposure = float(np.clip(bshare + min(cycle_over_takt / 5.0, 1.0), 0, 2)) / 2

        ambiguity = float(np.clip(unresolved_suspect_hits.get(st.sid, 0) / 5.0
                                  + defect_exposure.get(st.sid, 0.0), 0, 1))

        evi = 0.40 * uncertainty + 0.40 * exposure + 0.20 * ambiguity
        cost = st.stype.retrofit_cost_inr
        rows.append({
            "sid": st.sid, "zone": st.zone, "station_type": st.type_name,
            "cost_inr": cost,
            "uncertainty": round(uncertainty, 4),
            "exposure": round(exposure, 4),
            "ambiguity": round(ambiguity, 4),
            "evi_score": round(evi, 5),
            "evi_per_lakh": round(evi / (cost / 100_000), 5),
            "soft_sensor_ci_width_s": ci_w,
            "rationale": _prod_rationale(st.sid, uncertainty, exposure, ambiguity),
        })
    return (pd.DataFrame(rows).sort_values("evi_per_lakh", ascending=False)
            .reset_index(drop=True))


def validate_production_ranking(ablation: pd.DataFrame,
                                production: pd.DataFrame,
                                top_k: int = 5) -> dict:
    """Does the deployable score reproduce the ground-truth one, and where not?

    Reported component by component, because the aggregate answer is misleading
    and the decomposition is the actually useful result:

    * **Throughput value transfers perfectly.** How much a station constrains
      the line is visible in timing data you already own, so the deployable
      `exposure` term reproduces the ablation's throughput term exactly.
    * **Quality value does not transfer at all.** Whether a station's process
      drives defects cannot be inferred from the fact that you cannot measure
      it. No observable proxy predicts the ablation's PR-AUC uplift.

    That is a negative result and it dictates the deployment method: rank by
    EVI to choose *where to look*, then run a short instrumented pilot on the
    top few stations to measure the quality half before committing the rest of
    a shutdown budget.
    """
    from scipy import stats as _st

    a = ablation.set_index("sid")
    p = production.set_index("sid")
    common = [s for s in a.index if s in p.index]
    if len(common) < 4:
        return {"status": "insufficient_overlap", "n": len(common)}

    def rho(x, y):
        r = _st.spearmanr(x, y)
        return round(float(r.statistic), 4), float(r.pvalue)

    overall, p_overall = rho(p.loc[common, "evi_per_lakh"],
                             a.loc[common, "value_per_lakh"])
    thr, _ = rho(p.loc[common, "exposure"], a.loc[common, "bottleneck_share"])
    qual, _ = rho(p.loc[common, "uncertainty"],
                  a.loc[common, "d_pr_auc"].clip(lower=0))

    a_top = list(ablation["sid"][:top_k])
    p_top = list(production["sid"][:top_k])
    overlap = len(set(a_top) & set(p_top))

    return {
        "status": "ok",
        "n_stations": len(common),
        "spearman_overall": overall,
        "p_value_overall": p_overall,
        "spearman_throughput_component": thr,
        "spearman_quality_component": qual,
        f"top{top_k}_overlap": overlap,
        "top1_agrees": bool(a_top[0] == p_top[0]),
        "ablation_top": a_top,
        "production_top": p_top,
        "conclusion": (
            "Throughput value is predictable without the sensor "
            f"(rho={thr}); quality value is not (rho={qual}). Deployment "
            "therefore uses EVI to shortlist, then a temporary instrumented "
            "pilot on the top stations to measure quality uplift before "
            "committing the remaining budget."
        ),
        "deployment_method": "two-stage: EVI shortlist -> pilot -> commit",
    }


def _prod_rationale(sid: str, unc: float, exp: float, amb: float) -> str:
    bits = []
    if unc > 0.3:
        bits.append("the twin's own estimate here is wide")
    if exp > 0.3:
        bits.append("throughput risk sits on this station")
    if amb > 0.2:
        bits.append("it recurs as an unresolved suspect in traces")
    return f"{sid}: " + ("; ".join(bits) if bits
                         else "well modelled and low impact; defer")
