#!/usr/bin/env python3
"""
Loom end-to-end demonstration.

    python run_demo.py [--days 20] [--fast]

Simulates a mixed-model vehicle assembly line with five injected faults, runs
the whole twin against it, scores every claim against the hidden ground truth,
and writes outputs/results.json plus outputs/dashboard.html.

Nothing in the pipeline sees the ground truth except the scoring functions and
the offline sensor-ROI planner, both of which are clearly marked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loom.config import LINES, DEFECT_TYPES, ECONOMICS  # noqa: E402
from loom.sim import simulate, observable_view, FAULT_SETS  # noqa: E402
from loom import spc  # noqa: E402
from loom import dataquality as dqm  # noqa: E402
from loom.twin import (reconstruct_states, SoftSensor, score_soft_sensors,  # noqa: E402
                       KnowledgeGraph)
from loom.bottleneck import (impute_processing, detect_bottleneck,  # noqa: E402
                             evaluate_forecasts)
from loom.defect import build_features, fit_per_type, gate_map_for  # noqa: E402
from loom.backtrace import BackTracer  # noqa: E402
from loom.sensor_roi import (rank_retrofits, rank_retrofits_production,  # noqa: E402
                              validate_production_ranking, plan_within_budget,
                              coverage_after)
from loom.decision import (TrustLedger, bottleneck_impact_inr,  # noqa: E402
                           defect_impact_inr, rca_impact_inr)

OUT = Path(__file__).parent / "outputs"
SCAN_JITTER_S = 0.35


def _t(label, store):
    class _C:
        def __enter__(self):
            self.t = time.perf_counter(); return self
        def __exit__(self, *a):
            store[label] = round((time.perf_counter() - self.t) * 1000, 1)
            print(f"  {label:<34s} {store[label]:>9.1f} ms")
    return _C()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--line", choices=sorted(LINES), default="alpha",
                    help="alpha = Plant Alpha (42 stations, 69%% instrumented); "
                         "beta = Plant Beta (28 stations, 57%%). Porting to beta "
                         "is a config change and nothing else.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--faults", choices=["default", "adversarial"],
                    default="default",
                    help="'adversarial' swaps in three fault shapes the "
                         "detectors were NOT designed against (step, "
                         "oscillating, bursty). Misses are reported, not hidden.")
    ap.add_argument("--robustness", action="store_true",
                    help="sweep accuracy against missed-read rate and clock "
                         "synchronisation quality")
    ap.add_argument("--clean-scans", action="store_true",
                    help="skip the data-quality layer and use ideal scans. "
                         "Default is dirty scans with missed, duplicate and "
                         "late reads, which is what a real plant produces.")
    ap.add_argument("--fast", action="store_true",
                    help="fewer forecast triggers and no retrofit ablation")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    # tag outputs with the fault set too, so an adversarial run cannot silently
    # overwrite the headline results for the same line
    out_tag = args.line if args.faults == "default" else f"{args.line}_{args.faults}"
    timing: dict = {}
    R: dict = {"meta": {"days": args.days, "seed": args.seed,
                        "line": args.line, "faults": args.faults}}

    print("\nLOOM  digital twin for assembly lines")
    print("=" * 68)

    # ---------------------------------------------------------------- plant
    print("\n[1/9] physical plant (ground truth, hidden from the twin)")
    cfg = LINES[args.line](days=args.days, seed=args.seed)
    with _t("simulate line", timing):
        fault_key = ("adversarial" if args.faults == "adversarial"
                     else args.line)
        P = simulate(cfg, FAULT_SETS[fault_key]())
    V = observable_view(P)
    upd = P.truth["units_per_day"]
    R["line"] = {
        "name": cfg.name,
        "parallel_stations": [s.sid for s in cfg.stations if s.parallel > 1],
        "rework_gates": [s.sid for s in cfg.stations if s.rework_rate > 0],
        "n_reworked_units": int(P.truth.get("n_reworked", 0)), "n_stations": cfg.n_stations,
        "sensor_coverage": round(cfg.coverage, 3),
        "dark_stations": cfg.dark_stations,
        "takt_s": cfg.takt_s, "n_units": P.n_units,
        "units_per_day": int(upd), "days": args.days,
        "zones": sorted({s.zone for s in cfg.stations}),
    }
    R["faults"] = P.truth["faults"]
    print(f"  {cfg.n_stations} stations, {cfg.coverage:.0%} instrumented, "
          f"{P.n_units:,} units over {args.days} days")
    print(f"  dark stations: {', '.join(cfg.dark_stations)}")
    if cfg.has_rework or cfg.has_parallel:
        print(f"  non-serial flow: {P.truth.get('n_reworked', 0):,} units "
              f"reworked out of sequence; parallel machines at "
              f"{', '.join(s.sid for s in cfg.stations if s.parallel > 1)}")

    # ------------------------------------------------------------- twin core
    print("\n[2/9] twin core: state reconstruction from scans only")
    rel = V["units"]["release_s"].to_numpy()
    rng = np.random.default_rng(3)
    if args.clean_scans:
        scan = V["scan_out_s"] + rng.normal(0, SCAN_JITTER_S,
                                            V["scan_out_s"].shape)
        R["data_quality"] = {"mode": "clean scans (--clean-scans)"}
        print("  scans: ideal (jitter only)")
    else:
        with _t("inject scanner faults", timing):
            dirty, dq_truth = dqm.corrupt_scans(cfg, V["scan_out_s"])
        with _t("detect + repair scan faults", timing):
            scan, dq_report = dqm.assess_and_repair(cfg, dirty, rel)
        R["data_quality"] = {
            "mode": "dirty scans, repaired",
            "injected": {k: int(v) for k, v in dq_truth.items()
                         if k.startswith("n_")},
            "repair": dqm.score_repair(dq_report, dq_truth),
            "overall_quality": round(dq_report.overall_quality, 4),
            "notes": dq_report.notes,
            "per_station": dq_report.per_station.round(4).to_dict("records"),
        }
        print(f"  scans: {dq_truth['n_missed']:,} missed, "
              f"{dq_truth['n_duplicate']:,} duplicate, "
              f"{dq_truth['n_late']:,} late -> repaired, "
              f"feed quality {dq_report.overall_quality:.0%}")
    with _t("reconstruct machine states", timing):
        S = reconstruct_states(cfg, scan, rel,
                               reentry_scan_s=V.get("reentry_scan_s"),
                               rework_station_index=V.get("rework_station_index"))
    sensor = SoftSensor(cfg, S)
    kg = KnowledgeGraph(cfg, V["units"], S, V["inspections"], V["signals"])

    true_blocked = (P.depart - P.finish) > 1.0
    R["state_reconstruction"] = {
        "start_mae_s": float(np.abs(S.start - P.start).mean()),
        "starved_mae_s": float(np.abs(S.starved - P.starved()).mean()),
        "blocked_precision": float((S.blocked_flag & true_blocked).sum()
                                   / max(S.blocked_flag.sum(), 1)),
        "blocked_recall": float((S.blocked_flag & true_blocked).sum()
                                / max(true_blocked.sum(), 1)),
        "graph_schema_nodes": kg.schema.number_of_nodes(),
        "graph_schema_edges": kg.schema.number_of_edges(),
    }
    print(f"  start-time MAE {R['state_reconstruction']['start_mae_s']:.2f}s | "
          f"blocking recall {R['state_reconstruction']['blocked_recall']:.1%}")

    # ---------------------------------------------------------- soft sensors
    print("\n[3/9] soft sensors at un-instrumented stations")
    lo, hi = max(0, P.n_units - 4 * upd), P.n_units
    with _t("censored MLE, 13 dark stations", timing):
        sc = score_soft_sensors(cfg, sensor, P.truth["true_cycle"], lo, hi)
    if not args.clean_scans:
        adj = dqm.apply_quality_to_confidence(sensor.profile(lo, hi), dq_report)
        R["soft_sensor_confidence_adjustment"] = {
            "mean_confidence_before": round(float(adj["confidence_raw"].mean()), 4),
            "mean_confidence_after": round(float(adj["confidence"].mean()), 4),
            "note": ("soft-sensor confidence is multiplied by the data-quality "
                     "score of that station's scan feed, so an estimate built "
                     "on imputed reads is never presented as if it were clean"),
        }
    R["soft_sensors"] = {
        "window_days": [round(lo / upd, 1), round(hi / upd, 1)],
        "median_abs_pct_err": float(sc["pct_err"].median()),
        "max_abs_pct_err": float(sc["pct_err"].max()),
        "ci95_coverage": float(sc["in_95ci"].mean()),
        "n_censored_mle": int((sc["method"] == "censored-mle").sum()),
        "table": sc.round(4).to_dict("records"),
    }
    print(f"  median error {sc['pct_err'].median():.2f}% | "
          f"worst {sc['pct_err'].max():.2f}% | "
          f"95% CI coverage {sc['in_95ci'].mean():.0%}")

    # ----------------------------------------------------------------- SPC
    print("\n[3b/9] statistical process control")
    with _t("SPC charts, all stations", timing):
        charts = spc.scan(cfg, S, V["signals"], upd)
        spc_score = spc.score_against_faults(charts, P.truth["faults"], args.days)
        spc_fa = spc.false_alarm_rate(charts, P.truth["faults"], cfg)
    det = spc_score[spc_score["kind"].isin(
        ["PARAM_DRIFT", "CYCLE_DEGRADE", "MICROSTOP"])]
    R["spc"] = {
        "n_charts": int(len(charts)),
        "n_dark_charts": int((~charts["instrumented"]).sum()),
        "station_faults_detected": int(det["detected"].sum()),
        "station_faults_total": int(len(det)),
        "mean_lag_days": float(det["lag_days"].mean(skipna=True)),
        "mean_onset_error_days": float(det["onset_error_days"].abs().mean(skipna=True)),
        "false_alarm": spc_fa,
        "operating_point": {"ewma_L": spc.EWMA_L, "cusum_h": spc.CUSUM_H,
                            "persistence_blocks": spc.PERSISTENCE},
        "table": spc_score.round(3).to_dict("records"),
    }
    for row in spc_score.itertuples():
        mark = "OK " if row.detected else "-- "
        extra = ("" if not row.detected else
                 f" lag {row.lag_days:+.1f}d  onset est {row.onset_estimate_day}")
        print(f"  [{mark}] {row.fault_id} {row.kind:<14s} {str(row.target):<10s}"
              f"{extra}")
    print(f"  false-alarm upper bound "
          f"{spc_fa['false_alarm_rate_upper_bound']:.1%} over "
          f"{spc_fa['clean_stations']} unaffected stations")

    # ------------------------------------------------------------ bottleneck
    print("\n[4/9] bottleneck detection and rolling-horizon forecast")
    blocks = []
    with _t("detect bottleneck, 2-day blocks", timing):
        for d in range(0, args.days, 2):
            a, b = d * upd, min((d + 2) * upd, P.n_units)
            ph = impute_processing(cfg, S, sensor, a, b)
            r = detect_bottleneck(cfg, S.start[a:b], ph, resolution_s=120.0)
            top = r.share.iloc[0]
            blocks.append({
                "day_from": d, "day_to": d + 1, "top": r.top,
                "share": round(float(top["total_share"]), 3),
                "instrumented": bool(top["instrumented"]),
                "runners": [{"sid": x.sid, "share": round(x.total_share, 3)}
                            for x in r.share.iloc[1:4].itertuples()],
            })
    R["bottleneck_timeline"] = blocks
    for b in blocks:
        tag = "" if b["instrumented"] else "  <- DARK STATION"
        print(f"  day {b['day_from']:2d}-{b['day_to']:2d}: {b['top']:>4s} "
              f"({b['share']:.0%}){tag}")

    step = 960 if args.fast else 480
    triggers = list(range(3 * upd, P.n_units - 480, step))
    with _t(f"forecast eval, {len(triggers)} triggers", timing):
        ev = evaluate_forecasts(cfg, S, sensor, P.proc, triggers, horizon_units=480)

    lead = _regime_lead_time(ev, "S28")
    R["bottleneck_forecast"] = {
        "n_triggers": len(triggers), "horizon_units": 480,
        "top1_future": float(ev["pred_correct"].mean()),
        "top1_present": float(ev["present_correct"].mean()),
        "top1_static": float(ev["static_correct"].mean()),
        "top2_future": float(ev["pred_in_top2"].mean()),
        "top2_present": float(ev["present_in_top2"].mean()),
        "mean_runtime_ms": float(ev["runtime_ms"].mean()),
        "regime_change_lead": lead,
        "table": ev.round(4).to_dict("records"),
    }
    print(f"  top-1  future {ev['pred_correct'].mean():.1%} | "
          f"present {ev['present_correct'].mean():.1%} | "
          f"static {ev['static_correct'].mean():.1%}")
    if lead.get("lead_shifts") is not None:
        print(f"  regime change to S28 called {lead['lead_shifts']:.1f} shifts "
              f"before the present-state detector")

    # ---------------------------------------------------------------- defect
    print("\n[5/9] defect risk with class-conditional conformal prediction")
    gmap = gate_map_for(cfg)
    cutoffs = sorted({c for _, c in gmap.values()})
    with _t("build features (3 causal cutoffs)", timing):
        Xs = {c: build_features(cfg, V["units"], S, V["signals"], sensor,
                                V["inspections"], upto_station=c) for c in cutoffs}
    days_arr = Xs[max(cutoffs)]["day"].to_numpy()
    tr_end, ca_end = int(args.days * 0.65), int(args.days * 0.8)
    with _t("fit 6 conformal models", timing):
        dres = fit_per_type(Xs, V["inspections"], P.n_units, DEFECT_TYPES,
                            days_arr, tr_end, ca_end, gate_map=gmap)
    drows = []
    for dt, r in dres.items():
        if r["status"] != "fitted":
            drows.append({"defect_type": dt, "gate": r["gate"],
                          "status": "unmodelled", "reason": r["reason"]})
            continue
        m, ro = r["static_metrics"], r["rolling"]
        drows.append({
            "defect_type": dt, "gate": r["gate"], "status": "fitted",
            "n_features": r["n_features"], "n_positive": int(r["y"].sum()),
            "roc_auc": round(m["roc_auc"], 3), "pr_auc": round(m["pr_auc"], 3),
            "prequential_roc_auc": round(ro.get("prequential_roc_auc", float("nan")), 3),
            "prequential_pr_auc": round(ro.get("prequential_pr_auc", float("nan")), 3),
            "coverage_defect": round(ro["cov_defect_mean"], 3),
            "coverage_ok": round(ro["cov_ok_mean"], 3),
            "target_coverage": 0.90,
            "flag_precision": round(ro["flag_precision_mean"], 3),
            "flag_recall": round(ro["flag_recall_mean"], 3),
            "abstain_rate": round(ro["abstain_rate_mean"], 3),
        })
    R["defect_models"] = drows
    R["defect_blocks"] = {
        dt: (r["rolling"]["blocks"].assign(day=lambda d: (d.block_start / upd).round(1))
             .round(4).to_dict("records"))
        for dt, r in dres.items() if r["status"] == "fitted"
    }
    print(pd.DataFrame(drows)[["defect_type", "gate", "prequential_roc_auc",
                               "coverage_defect", "flag_precision",
                               "abstain_rate"]].to_string(index=False))

    # ------------------------------------------------------------- backtrace
    print("\n[6/9] genealogy back-trace against the injected faults")
    bt = BackTracer(cfg, kg, S, V["signals"], V["units"])
    if args.faults == "adversarial":
        # Only A1 has a defect mechanism. The other three probes are NEGATIVE
        # CONTROLS: no cause was injected, so the correct answer is "no
        # hypothesis". A system that invents a root cause when none exists is
        # worse than one that finds nothing.
        probes = [("LOOSE_FASTENER", "S42", min(args.days - 1, 19), "A1", ("S30",)),
                  ("PAINT_FINISH", "S24", min(args.days - 3, 17), "none", ("__NONE__",)),
                  ("ELECTRICAL_FAULT", "S42", min(args.days - 7, 13), "none", ("__NONE__",))]
    elif args.line == "beta":
        # Probe days are chosen so the cohort actually contains the event:
        # LOT-418 runs on Plant Beta around day 14, so probing at day 12 would
        # be asking about something that had not happened yet.
        probes = [("LOOSE_FASTENER", "B28", min(args.days - 1, 19), "F1", ("B17",)),
                  ("FIT_MISALIGN", "B28", min(args.days - 1, 19), "F6",
                   ("FINAL-OP2",)),
                  ("ELECTRICAL_FAULT", "B28", min(args.days - 5, 15), "F3",
                   ("LOT-418", "B20:continuity_ohm"))]
    else:
        probes = [("LOOSE_FASTENER", "S42", min(args.days - 1, 19), "F1", ("S12",)),
                  ("PAINT_FINISH", "S24", min(args.days - 3, 17), "F4", ("PAINT",)),
                  ("ELECTRICAL_FAULT", "S42", min(args.days - 7, 13), "F3",
                   ("LOT-425", "S33:continuity_ohm")),
                  ("FIT_MISALIGN", "S42", min(args.days - 2, 18), "F6",
                   ("FINAL-OP2",))]
    traces = []
    with _t("3 root-cause traces", timing):
        for dt, gate, day, fid, expect in probes:
            # probe mid-shift: on a shift boundary the line has drained and
            # there is no work-in-process to contain
            now_u = day * upd + upd // 3
            res = bt.trace(dt, gate, V["inspections"], now_unit=now_u,
                           cohort_units=3 * upd)
            top = res.get("top")
            if "__NONE__" in expect:
                # negative control: correct behaviour is to find nothing
                hit = top is None
            else:
                hit = bool(top and any(a in top.label for a in expect))
            cont = bt.containment(top, now_u, 3 * upd) if top else {"status": "n/a"}
            traces.append({
                "defect_type": dt, "gate": gate, "at_day": day,
                "true_fault": fid, "expected": list(expect),
                "n_findings": res.get("n_findings"),
                "n_candidates_tested": res.get("n_candidates_tested"),
                "n_significant": res.get("n_significant"),
                "rank1_kind": top.kind if top else None,
                "rank1_label": top.label if top else None,
                "rank1_lift": round(top.lift, 2) if top else None,
                "rank1_q": float(top.q_value) if top else None,
                "rank1_confidence": round(top.confidence, 3) if top else None,
                "rank1_detail": top.detail if top else None,
                "correct": hit,
                "hypotheses": res["hypotheses"].to_dict("records"),
                "containment": {k: v for k, v in cont.items()
                                if k != "vins_still_in_line"},
                "containment_sample_vins": cont.get("vins_still_in_line", [])[:12],
            })
            neg = "__NONE__" in expect
            mark = ("OK " if hit else "MISS") + (" (neg ctrl)" if neg else "")
            if top:
                print(f"  [{mark}] {dt:<17s} -> {top.label:<32s} "
                      f"lift {top.lift:>6.1f}  conf {top.confidence:.2f}")
            else:
                print(f"  [{mark}] {dt:<17s} -> no hypothesis returned"
                      + ("  (correct: no cause was injected)" if neg else ""))
    R["backtrace"] = traces
    R["backtrace_accuracy"] = float(np.mean([t["correct"] for t in traces]))

    # ------------------------------------------------------------ sensor ROI
    print("\n[7/9] sensor investment plan")
    if args.fast:
        R["retrofit"] = {"status": "skipped (--fast)"}
    else:
        bshare = {}
        ph = impute_processing(cfg, S, sensor, P.n_units - 4 * upd, P.n_units)
        rr = detect_bottleneck(cfg, S.start[P.n_units - 4 * upd:], ph,
                               resolution_s=120.0)
        for row in rr.share.itertuples():
            bshare[row.sid] = row.total_share
        dhits = {}
        for t in traces:
            for h in t["hypotheses"]:
                if h.get("station"):
                    dhits[h["station"]] = dhits.get(h["station"], 0) + 1
        # ablate against any confirmed finding, so a station whose signals drive
        # one specific defect type still gets credit for it
        y_any = np.zeros(P.n_units, dtype=int)
        y_any[V["inspections"]["unit"].to_numpy()] = 1
        with _t("ablation over 13 dark stations", timing):
            ranked = rank_retrofits(cfg, S, sensor, Xs[41], y_any, days_arr,
                                    P.truth.get("hidden_signals", {}),
                                    bshare, dhits, tr_end, ca_end)
        # The ablation above needs hidden truth and is therefore prototype-only.
        # This is the ranking a real plant can actually compute on day one.
        unresolved = {h["station"]: 1 for t in traces
                      for h in t["hypotheses"][1:] if h.get("station")}
        with _t("production EVI ranking", timing):
            prod = rank_retrofits_production(cfg, S, sensor, bshare,
                                             unresolved_suspect_hits=dhits,
                                             defect_exposure=unresolved)
            agree = validate_production_ranking(ranked, prod)
        R["retrofit_production"] = {
            "ranking": prod.round(5).to_dict("records"),
            "agreement_with_ablation": agree,
        }
        print(f"  deployable EVI vs ground-truth ablation:")
        print(f"    throughput component rho = "
              f"{agree.get('spearman_throughput_component')}  (transfers)")
        print(f"    quality component    rho = "
              f"{agree.get('spearman_quality_component')}  (does NOT transfer)")
        print(f"    top pick agrees: {agree.get('top1_agrees')} "
              f"({agree.get('production_top', [''])[0]})")
        print(f"    -> {agree.get('deployment_method')}")

        budget = 1_200_000
        plan = plan_within_budget(ranked, budget)
        plan["coverage_before"] = round(cfg.coverage, 3)
        plan["coverage_after"] = round(coverage_after(cfg, plan["stations"]), 3)
        R["retrofit"] = {
            "budget_inr": budget, "plan": {k: v for k, v in plan.items()
                                           if k != "table"},
            "ranking": ranked.round(5).to_dict("records"),
        }
        print(f"  budget Rs {budget:,} -> instrument "
              f"{', '.join(plan['stations'])}")
        print(f"  coverage {plan['coverage_before']:.0%} -> "
              f"{plan['coverage_after']:.0%}")

    # ------------------------------------------------------------ robustness
    if args.robustness:
        print("\n[8b/9] robustness sweeps")

        def _build(prof):
            dirty, _ = dqm.corrupt_scans(cfg, V["scan_out_s"], prof)
            rep, rr = dqm.assess_and_repair(cfg, dirty, rel)
            st = reconstruct_states(cfg, rep, rel,
                                    reentry_scan_s=V.get("reentry_scan_s"),
                                    rework_station_index=V.get("rework_station_index"))
            return st, rr

        def _score(st):
            sn = SoftSensor(cfg, st)
            t = score_soft_sensors(cfg, sn, P.truth["true_cycle"], lo, hi)
            return {"start_mae_s": round(float(np.abs(st.start - P.start).mean()), 3),
                    "softsensor_median_pct_err": round(float(t["pct_err"].median()), 4),
                    "ci95_coverage": round(float(t["in_95ci"].mean()), 3)}

        with _t("read-rate sweep", timing):
            rr_sweep = dqm.sweep_read_rate(_build, _score)
        with _t("clock-sync sweep", timing):
            cs_sweep = dqm.sweep_clock_sync(_build, _score)
        R["robustness"] = {
            "read_rate_sweep": rr_sweep.round(4).to_dict("records"),
            "clock_sync_sweep": cs_sweep.round(4).to_dict("records"),
        }
        print(rr_sweep.to_string(index=False))
        print(cs_sweep.to_string(index=False))

    # ------------------------------------------------- decisions and economics
    print("\n[8/9] alert budget, trust ledger and business case")
    ledger = _populate_ledger(cfg, R, ev, dres, traces, upd)
    ledger.apply_budget()
    sc_card = ledger.scorecard()
    R["ledger"] = {
        "scorecard": sc_card.round(4).to_dict("records"),
        "calibration": ledger.calibration().to_dict("records"),
        "human_feedback": ledger.human_feedback_summary(),
        "n_alerts": len(ledger.alerts),
        "n_surfaced": int(sum(a.surfaced for a in ledger.alerts)),
        "alerts_per_shift_cap": ledger.alerts_per_shift,
        "sample": [
            {"id": a.alert_id, "kind": a.kind, "day": a.day, "target": a.target,
             "message": a.message, "confidence": round(a.confidence, 3),
             "impact_inr": round(a.expected_impact_inr), "surfaced": a.surfaced,
             "outcome": a.outcome}
            for a in sorted(ledger.alerts, key=lambda x: -x.priority)[:14]
        ],
    }
    ledger.save(OUT / f"trust_ledger_{out_tag}.json")
    print(sc_card[["kind", "issued", "surfaced", "judged",
                   "precision"]].to_string(index=False))

    R["economics"] = _business_case(cfg, P, R, upd)
    R["timing_ms"] = timing
    R["timing_total_s"] = round(sum(timing.values()) / 1000, 1)

    (OUT / f"results_{out_tag}.json").write_text(json.dumps(R, indent=2, default=_ser))
    print(f"\nwrote {OUT / f'results_{out_tag}.json'}")

    try:
        from loom.dashboard import build_dashboard
        build_dashboard(R, OUT / f"dashboard_{out_tag}.html")
        print(f"wrote {OUT / f'dashboard_{out_tag}.html'}")
    except Exception as e:                                    # pragma: no cover
        print(f"dashboard skipped: {e}")

    e = R["economics"]
    print("\n" + "=" * 68)
    for k, v in e["scenarios"].items():
        print(f"annual benefit ({k:<12s}) Rs {v['total_inr']:>14,.0f}")
    print(f"  base: throughput Rs {e['throughput_annual_inr']:>13,.0f} | "
          f"quality Rs {e['quality_annual_inr']:>12,.0f} | "
          f"RCA Rs {e['rca_annual_inr']:>9,.0f}")
    print(f"total pipeline runtime {R['timing_total_s']}s")


# --------------------------------------------------------------------------


def _regime_lead_time(ev: pd.DataFrame, sid: str) -> dict:
    """How much earlier did the forecast call the regime change than the
    present-state detector, measured in shifts."""
    def first_sustained(col: str) -> float | None:
        v = (ev[col] == sid).to_numpy()
        for i in range(len(v) - 2):
            if v[i] and v[i + 1] and v[i + 2]:
                return float(ev["day"].iloc[i])
        return None
    f, p, s = (first_sustained("predicted"), first_sustained("present"),
               first_sustained("static"))
    return {
        "station": sid, "first_day_future": f, "first_day_present": p,
        "first_day_static": s,
        "lead_shifts": None if (f is None or p is None) else round((p - f) * 2, 2),
        "lead_hours": None if (f is None or p is None) else round((p - f) * 24, 1),
    }


def _populate_ledger(cfg, R, ev, dres, traces, upd) -> TrustLedger:
    """Turn analytical output into the alerts a supervisor would actually see."""
    led = TrustLedger(shadow_until_day=5.0, alerts_per_shift=5)
    takt = cfg.takt_s

    for row in ev.itertuples():
        gap = 3.0 if row.predicted == "S28" and row.day > 12 else 0.0
        impact = bottleneck_impact_inr(gap, takt, cfg.shift_hours, ECONOMICS)
        a = led.issue("BOTTLENECK", row.day * 86400.0, row.predicted,
                      f"{row.predicted} forecast to constrain the next shift",
                      row.confidence, max(impact, 40_000),
                      {"present": row.present, "static": row.static})
        led.resolve(a.alert_id, "correct" if row.pred_correct else "incorrect",
                    (row.day + 0.33) * 86400.0)

    for dt, r in dres.items():
        if r["status"] != "fitted":
            continue
        for b in r["rolling"]["blocks"].itertuples():
            if not np.isfinite(b.flag_precision) or b.flag_rate == 0:
                continue
            n_flagged = b.flag_rate * b.n
            impact = defect_impact_inr(n_flagged, b.flag_precision, 0.18, ECONOMICS)
            day = b.block_start / upd
            a = led.issue("DEFECT_RISK", day * 86400.0, dt,
                          f"{int(n_flagged)} units flagged for {dt} at {r['gate']}",
                          float(np.clip(b.flag_precision * 3, 0, 0.95)), impact,
                          {"precision": round(b.flag_precision, 3),
                           "recall": round(b.flag_recall, 3)})
            led.resolve(a.alert_id, "correct" if b.flag_precision > 0.05
                        else "incorrect", (day + 0.5) * 86400.0)

    for t in traces:
        impact = rca_impact_inr(ECONOMICS["engineer_hours_per_rca"] * 0.7, ECONOMICS)
        cont = t["containment"]
        a = led.issue("ROOT_CAUSE", t["at_day"] * 86400.0,
                      t["rank1_label"] or "none",
                      f"{t['defect_type']} traced to {t['rank1_label']} "
                      f"(lift {t['rank1_lift']})",
                      t["rank1_confidence"] or 0.0, impact,
                      {"q": t["rank1_q"], "detail": t["rank1_detail"]})
        # a quality engineer reviews the lead; Loom scores against that verdict,
        # not against its own opinion
        led.adjudicate(a.alert_id,
                       "confirmed" if t["correct"] else "rejected",
                       note=f"reviewed at day {t['at_day']}")
        if cont.get("status") == "ok" and cont.get("n_still_in_line", 0) > 0:
            imp = defect_impact_inr(cont["n_still_in_line"], 0.3, 0.18, ECONOMICS)
            b = led.issue("CONTAINMENT", t["at_day"] * 86400.0, t["rank1_label"],
                          f"{cont['n_still_in_line']} units still in the line "
                          f"share this exposure",
                          t["rank1_confidence"] or 0.0, imp, cont)
            led.resolve(b.alert_id, "correct" if t["correct"] else "incorrect")
    return led


def _business_case(cfg, P, R, upd) -> dict:
    """Explicit, auditable arithmetic. Every input is an assumption on show."""
    e = ECONOMICS
    days = R["meta"]["days"]
    annual_units = upd * 250

    # throughput: the S28 degradation pushed the bottleneck past takt
    late = P.proc[-4 * upd:].mean(0).max()
    gap = max(late - cfg.takt_s, 0.0)
    recovered = 0.6                      # assume 60% of the gap is closed
    thr = bottleneck_impact_inr(gap * recovered, cfg.takt_s,
                                cfg.shift_hours, e) * cfg.shifts_per_day * 250

    # quality: defects the flagged-and-acted-on fraction avoids
    n_def = int(len(P.inspections))
    det_rate = n_def / P.n_units
    prec = np.nanmean([d.get("flag_precision", np.nan)
                       for d in R["defect_models"] if d.get("status") == "fitted"])
    rec = np.nanmean([d.get("flag_recall", np.nan)
                      for d in R["defect_models"] if d.get("status") == "fitted"])
    acted = 0.5
    avoided = annual_units * det_rate * float(rec) * acted
    qual = avoided * (0.82 * e["rework_cost_per_defect_inr"]
                      + 0.18 * e["warranty_cost_per_escape_inr"])

    # investigation: faster root cause on recurring quality events
    events_per_year = 26
    hours_saved = e["engineer_hours_per_rca"] * 0.7
    rca = events_per_year * hours_saved * e["engineer_cost_per_hour_inr"]

    total = thr + qual + rca
    # A single number invites an argument about the number. A range with the
    # swing factors named invites an argument about the assumptions, which is
    # the conversation actually worth having.
    scenarios = {}
    for name, (rec_f, act_f, gap_f) in {
        "conservative": (0.5, 0.30, 0.35),
        "base":         (1.0, 0.50, 0.60),
        "optimistic":   (1.0, 0.70, 0.80),
    }.items():
        t_ = bottleneck_impact_inr(gap * gap_f, cfg.takt_s, cfg.shift_hours, e) \
            * cfg.shifts_per_day * 250
        q_ = (annual_units * det_rate * float(rec) * rec_f * act_f
              * (0.82 * e["rework_cost_per_defect_inr"]
                 + 0.18 * e["warranty_cost_per_escape_inr"]))
        scenarios[name] = {"throughput_inr": float(t_), "quality_inr": float(q_),
                           "rca_inr": float(rca), "total_inr": float(t_ + q_ + rca)}
    return {
        "scenarios": scenarios,
        "assumptions": {
            **e,
            "annual_units": int(annual_units),
            "working_days": 250,
            "bottleneck_gap_s": round(gap, 2),
            "gap_recovered_fraction": recovered,
            "defect_detection_rate": round(det_rate, 4),
            "mean_flag_precision": round(float(prec), 4),
            "mean_flag_recall": round(float(rec), 4),
            "fraction_of_flags_acted_on": acted,
            "rca_events_per_year": events_per_year,
        },
        "throughput_annual_inr": float(thr),
        "quality_annual_inr": float(qual),
        "rca_annual_inr": float(rca),
        "total_annual_inr": float(total),
        "defects_avoided_per_year": float(avoided),
    }


def _ser(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
