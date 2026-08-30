#!/usr/bin/env python3
"""
Scoring harness.

    python eval/run_eval.py [--seeds 5]

Every claim Loom makes is scored here against ground truth the twin never sees.
The harness runs the whole pipeline over several random seeds, because a single
seed on a stochastic line is an anecdote. It reports means and spreads for:

    A  state reconstruction fidelity  (start / starved MAE, blocking recall)
    B  soft-sensor accuracy at dark stations, and interval coverage
    C  bottleneck regime-change detection lead time vs two baselines
    D  root-cause attribution: rank-1 hit rate against the injected faults
    E  conformal coverage against its 90% target under drift

Exit code is non-zero if any headline metric falls outside its stated envelope,
so this doubles as a regression test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loom.config import build_reference_line, DEFECT_TYPES  # noqa: E402
from loom.defect import gate_map_for  # noqa: E402
from loom.sim import simulate, observable_view  # noqa: E402
from loom.twin import reconstruct_states, SoftSensor, score_soft_sensors, KnowledgeGraph  # noqa: E402
from loom.bottleneck import evaluate_forecasts  # noqa: E402
from loom.defect import build_features, fit_per_type  # noqa: E402
from loom.backtrace import BackTracer  # noqa: E402
from loom import spc  # noqa: E402

# metric -> (lower bound, upper bound). Deliberately wide: these are the claims
# the proposal makes, not the best numbers a single seed produced.
ENVELOPE = {
    "start_mae_s": (0.0, 1.0),
    "softsensor_median_pct_err": (0.0, 2.0),
    "softsensor_ci_coverage": (0.80, 1.0),
    "blocking_recall": (0.95, 1.0),
    "backtrace_rank1_hit_rate": (0.80, 1.0),
    "backtrace_top3_hit_rate": (0.95, 1.0),
    "conformal_coverage_defect": (0.82, 1.0),
    "spc_station_fault_recall": (0.66, 1.0),
    "spc_false_alarm_upper_bound": (0.0, 0.40),
    "regime_lead_hours": (0.0, 1e9),
}


def run_seed(seed: int, days: int, triggers_step: int) -> dict:
    cfg = build_reference_line(days=days, seed=seed)
    P = simulate(cfg)
    V = observable_view(P)
    upd = P.truth["units_per_day"]

    rng = np.random.default_rng(seed + 1000)
    scan = V["scan_out_s"] + rng.normal(0, 0.35, V["scan_out_s"].shape)
    S = reconstruct_states(cfg, scan, V["units"]["release_s"].to_numpy(),
                           reentry_scan_s=V.get("reentry_scan_s"),
                           rework_station_index=V.get("rework_station_index"))
    sensor = SoftSensor(cfg, S)
    kg = KnowledgeGraph(cfg, V["units"], S, V["inspections"], V["signals"])

    # --- A. state reconstruction
    true_blocked = (P.depart - P.finish) > 1.0
    out = {
        "seed": seed,
        "n_reworked_units": int(P.truth.get("n_reworked", 0)),
        "pct_units_out_of_sequence": round(
            float(P.truth.get("n_reworked", 0)) / P.n_units, 4),
        "start_mae_s": float(np.abs(S.start - P.start).mean()),
        "starved_mae_s": float(np.abs(S.starved - P.starved()).mean()),
        "blocking_recall": float((S.blocked_flag & true_blocked).sum()
                                 / max(true_blocked.sum(), 1)),
        "blocking_precision": float((S.blocked_flag & true_blocked).sum()
                                    / max(S.blocked_flag.sum(), 1)),
    }

    # --- B. soft sensors
    lo, hi = max(0, P.n_units - 4 * upd), P.n_units
    sc = score_soft_sensors(cfg, sensor, P.truth["true_cycle"], lo, hi)
    out["softsensor_median_pct_err"] = float(sc["pct_err"].median())
    out["softsensor_max_pct_err"] = float(sc["pct_err"].max())
    out["softsensor_ci_coverage"] = float(sc["in_95ci"].mean())

    # --- B2. SPC
    charts = spc.scan(cfg, S, V["signals"], upd)
    sscore = spc.score_against_faults(charts, P.truth["faults"], days)
    st_f = sscore[sscore["kind"].isin(["PARAM_DRIFT", "CYCLE_DEGRADE", "MICROSTOP"])]
    fa = spc.false_alarm_rate(charts, P.truth["faults"], cfg)
    out["spc_station_fault_recall"] = float(st_f["detected"].mean())
    out["spc_mean_lag_days"] = float(st_f["lag_days"].mean(skipna=True))
    out["spc_onset_abs_error_days"] = float(
        st_f["onset_error_days"].abs().mean(skipna=True))
    out["spc_false_alarm_upper_bound"] = float(fa["false_alarm_rate_upper_bound"])

    # --- C. bottleneck regime change
    trg = list(range(3 * upd, P.n_units - 480, triggers_step))
    ev = evaluate_forecasts(cfg, S, sensor, P.proc, trg, horizon_units=480)
    out["bn_top1_future"] = float(ev["pred_correct"].mean())
    out["bn_top1_present"] = float(ev["present_correct"].mean())
    out["bn_top1_static"] = float(ev["static_correct"].mean())

    def first(col: str) -> float | None:
        v = (ev[col] == "S28").to_numpy()
        for i in range(len(v) - 2):
            if v[i] and v[i + 1] and v[i + 2]:
                return float(ev["day"].iloc[i])
        return None

    f, p = first("predicted"), first("present")
    out["regime_first_day_future"] = f
    out["regime_first_day_present"] = p
    out["regime_lead_hours"] = None if (f is None or p is None) else (p - f) * 24

    # --- D. attribution
    gmap = gate_map_for(cfg)
    cutoffs = sorted({c for _, c in gmap.values()})
    Xs = {c: build_features(cfg, V["units"], S, V["signals"], sensor,
                            V["inspections"], upto_station=c) for c in cutoffs}
    bt = BackTracer(cfg, kg, S, V["signals"], V["units"])
    # Each fault gets an acceptance set, not a single string. For the
    # contaminated lot, "LOT-425" is the source and "S33:continuity_ohm high"
    # is the mechanism by which that lot fails. Both are correct answers and
    # both lead to the same containment action, so both are scored as hits.
    # This is stated up front rather than widened after seeing the results.
    probes = [
        ("LOOSE_FASTENER", "S42", 19, ("S12",)),
        ("PAINT_FINISH", "S24", 17, ("PAINT",)),
        ("ELECTRICAL_FAULT", "S42", 13, ("LOT-425", "S33:continuity_ohm")),
        ("FIT_MISALIGN", "S42", 18, ("FINAL-OP2",)),
    ]
    hits, top3, detail = [], [], []
    for dt, gate, day, accept in probes:
        day = min(day, days - 1)
        res = bt.trace(dt, gate, V["inspections"],
                       now_unit=day * upd + upd // 3, cohort_units=3 * upd)
        top = res.get("top")
        labels = list(res["hypotheses"]["label"]) if len(res["hypotheses"]) else []
        hit = bool(top and any(a in top.label for a in accept))
        in3 = any(any(a in l for a in accept) for l in labels[:3])
        hits.append(hit)
        top3.append(in3)
        detail.append({"defect": dt, "accept": list(accept),
                       "got": top.label if top else None,
                       "lift": round(top.lift, 2) if top else None,
                       "rank1_hit": hit, "top3_hit": in3})
    out["backtrace_rank1_hit_rate"] = float(np.mean(hits))
    out["backtrace_top3_hit_rate"] = float(np.mean(top3))
    out["backtrace_detail"] = detail

    # --- E. conformal coverage
    days_arr = Xs[max(cutoffs)]["day"].to_numpy()
    dres = fit_per_type(Xs, V["inspections"], P.n_units, DEFECT_TYPES, days_arr,
                        int(days * 0.65), int(days * 0.8), gate_map=gmap)
    cov = [r["rolling"]["cov_defect_mean"] for r in dres.values()
           if r["status"] == "fitted"]
    out["conformal_coverage_defect"] = float(np.mean(cov))
    out["conformal_coverage_min"] = float(np.min(cov))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--step", type=int, default=960)
    args = ap.parse_args()

    rows = []
    for i in range(args.seeds):
        seed = 7 + 13 * i
        print(f"seed {seed} ...", flush=True)
        rows.append(run_seed(seed, args.days, args.step))

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if not isinstance(v, list)} for r in rows])
    num = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({"mean": num.mean(), "sd": num.std(ddof=0),
                            "min": num.min(), "max": num.max()}).round(4)

    print("\n" + "=" * 74)
    print(f"LOOM scoring harness  |  {args.seeds} seeds x {args.days} days")
    print("=" * 74)
    print(summary.to_string())

    print("\nattribution detail (last seed):")
    for d in rows[-1]["backtrace_detail"]:
        print(f"  [{'OK ' if d['rank1_hit'] else 'MISS'}] {d['defect']:<17s} "
              f"accepts {'/'.join(d['accept']):<28s} got {str(d['got']):<32s} "
              f"lift {d['lift']}")

    print("\nenvelope check:")
    failures = []
    for metric, (lo, hi) in ENVELOPE.items():
        if metric not in summary.index:
            continue
        v = summary.loc[metric, "mean"]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ok = lo <= v <= hi
        print(f"  {'PASS' if ok else 'FAIL'}  {metric:<32s} {v:>9.4f}  "
              f"expected [{lo}, {hi}]")
        if not ok:
            failures.append(metric)

    out = Path(__file__).parent.parent / "outputs" / "eval_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"summary": summary.to_dict(), "runs": rows, "failures": failures},
        indent=2, default=str))
    print(f"\nwrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
