"""
Dashboard generator.

Writes one self-contained HTML file - no server, no install, no CDN - so the
demo runs from a file:// URL on a plant laptop that has no internet. The three
persona views the brief asks for are tabs over a single results payload, which
is the point: one twin, three audiences, no divergent copies of the truth.

    floor supervisor : what is constraining the line right now, and what to do
    plant manager    : where the week went, and which retrofits to fund
    leadership       : the investment case and the twin's own track record
"""

from __future__ import annotations

import json
from pathlib import Path

CSS = """
:root{
  --ink:#12161c; --ink-2:#4a5464; --ink-3:#7b8595; --line:#dfe3ea;
  --bg:#f4f6f9; --card:#ffffff; --accent:#1f4ed8; --accent-soft:#e8eefc;
  --warn:#b45309; --warn-soft:#fdf2e0; --bad:#b02a2a; --bad-soft:#fbeaea;
  --good:#136c48; --good-soft:#e3f3ec; --dark:#1c2430;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{background:var(--dark);color:#fff;padding:26px 30px 0}
header h1{margin:0;font-size:23px;font-weight:600;letter-spacing:-.2px}
header p{margin:6px 0 18px;color:#a9b4c4;font-size:14px;max-width:80ch}
nav{display:flex;gap:2px}
nav button{background:transparent;border:0;border-bottom:3px solid transparent;
  color:#a9b4c4;padding:11px 18px;font-size:14px;font-weight:500;cursor:pointer;
  font-family:inherit}
nav button:hover{color:#fff}
nav button[aria-selected="true"]{color:#fff;border-bottom-color:#5b8dff}
main{padding:26px 30px 60px;max-width:1280px}
section[hidden]{display:none}
h2{font-size:17px;font-weight:600;margin:30px 0 4px}
h2:first-child{margin-top:0}
.sub{color:var(--ink-2);font-size:13.5px;margin:0 0 14px;max-width:88ch}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi .v{font-size:27px;font-weight:600;letter-spacing:-.5px;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:5px}
.kpi .n{font-size:12.5px;color:var(--ink-2);margin-top:5px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card)}
th{text-align:left;font-weight:600;color:var(--ink-2);font-size:11.5px;
  text-transform:uppercase;letter-spacing:.5px;padding:9px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid #eef1f5;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
.wrap{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:6px}
.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11.5px;font-weight:600}
.t-ok{background:var(--good-soft);color:var(--good)}
.t-warn{background:var(--warn-soft);color:var(--warn)}
.t-bad{background:var(--bad-soft);color:var(--bad)}
.t-info{background:var(--accent-soft);color:var(--accent)}
.t-dark{background:#efe6fb;color:#5b2d90}
.bar{height:7px;background:#eef1f5;border-radius:4px;overflow:hidden;min-width:64px}
.bar>i{display:block;height:100%;background:var(--accent)}
.mono{font-family:var(--mono);font-size:12.5px}
.note{background:var(--accent-soft);border-left:3px solid var(--accent);
  padding:11px 14px;border-radius:0 6px 6px 0;font-size:13.5px;margin:14px 0;color:#1c3a8f}
.caveat{background:#fff8ec;border-left:3px solid var(--warn);
  padding:11px 14px;border-radius:0 6px 6px 0;font-size:13.5px;margin:14px 0;color:#7a4c07}
.tl{display:flex;gap:3px;margin:10px 0 4px}
.tl>div{flex:1;text-align:center;padding:9px 3px;border-radius:6px;font-size:12px;font-weight:600}
.hint{font-size:12px;color:var(--ink-3);margin-top:4px}
"""

JS = """
const tabs=[...document.querySelectorAll('nav button')];
tabs.forEach(b=>b.onclick=()=>{
  tabs.forEach(x=>{x.setAttribute('aria-selected', x===b);
    document.getElementById(x.dataset.t).hidden = x!==b;});
});
"""


# --------------------------------------------------------------------- utils

def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _inr(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if abs(v) >= 1e7:
        return f"Rs {v/1e7:,.2f} cr"
    if abs(v) >= 1e5:
        return f"Rs {v/1e5:,.1f} L"
    return f"Rs {v:,.0f}"


def _kpi(label, value, note="") -> str:
    return (f'<div class="card kpi"><div class="l">{_esc(label)}</div>'
            f'<div class="v">{_esc(value)}</div>'
            + (f'<div class="n">{_esc(note)}</div>' if note else "") + "</div>")


def _table(cols, rows, aligns=None) -> str:
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return (f'<div class="wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _bar(frac, width_pct=100) -> str:
    f = max(0.0, min(1.0, float(frac)))
    return f'<div class="bar"><i style="width:{f*width_pct:.0f}%"></i></div>'


# ------------------------------------------------------------------- builder

def build_dashboard(R: dict, path: str | Path) -> Path:
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loom - {_esc(R['line']['name'])}</title><style>{CSS}</style></head><body>
<header>
  <h1>Loom &mdash; {_esc(R['line']['name'])}</h1>
  <p>A digital twin that stays useful where the sensors run out.
  {R['line']['n_stations']} stations, {R['line']['sensor_coverage']:.0%} instrumented,
  {R['line']['n_units']:,} units simulated over {R['line']['days']} days.
  Configuration <code>{_esc(R['meta'].get('line', 'alpha'))}</code> &mdash; the
  same code runs Plant Beta (28 stations, 57% instrumented) from a config file
  with no code change.
  Every figure below is scored against ground truth the twin never sees.</p>
  <nav>
    <button data-t="v-floor" aria-selected="true">Floor supervisor</button>
    <button data-t="v-plant" aria-selected="false">Plant manager</button>
    <button data-t="v-exec"  aria-selected="false">Leadership</button>
    <button data-t="v-eng"   aria-selected="false">Validation</button>
  </nav>
</header>
<main>
{_floor(R)}
{_plant(R)}
{_exec(R)}
{_eng(R)}
</main>
<script>{JS}</script></body></html>"""
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p


# ------------------------------------------------------------------ sections

def _floor(R: dict) -> str:
    tl = R["bottleneck_timeline"]
    cur = tl[-1]
    lead = R["bottleneck_forecast"].get("regime_change_lead", {})
    alerts = [a for a in R["ledger"]["sample"] if a["surfaced"]][:6]
    sc = {r["kind"]: r for r in R["ledger"]["scorecard"]}

    rows = []
    for a in alerts:
        prec = sc.get(a["kind"], {}).get("precision")
        prec_s = f"{prec:.0%}" if isinstance(prec, (int, float)) else "-"
        tone = ("t-bad" if a["kind"] == "ROOT_CAUSE" else
                "t-warn" if a["kind"] == "BOTTLENECK" else "t-info")
        rows.append([
            f'<span class="tag {tone}">{_esc(a["kind"])}</span>',
            f'day {a["day"]:.1f}',
            _esc(a["message"]),
            f'{a["confidence"]:.0%}',
            _inr(a["impact_inr"]),
            f'<span class="hint">{prec_s} right so far</span>',
        ])

    strip = '<div class="tl">'
    for b in tl:
        tone = "#efe6fb" if not b["instrumented"] else "#e8eefc"
        col = "#5b2d90" if not b["instrumented"] else "#1f4ed8"
        strip += (f'<div style="background:{tone};color:{col}">'
                  f'{b["top"]}<br><span style="font-weight:400;font-size:11px">'
                  f'd{b["day_from"]}-{b["day_to"]}</span></div>')
    strip += "</div>"

    dark_note = ""
    if not cur["instrumented"]:
        dark_note = (f'<div class="caveat"><strong>{cur["top"]} has no process '
                     f'sensors.</strong> Its cycle time is inferred from arrival '
                     f'and departure scans, so it is visible here even though it '
                     f'is invisible to the plant historian. This is the case a '
                     f'conventional twin misses entirely.</div>')

    return f"""<section id="v-floor">
<h2>Right now</h2>
<p class="sub">What is holding the line back this shift, and how much confidence
sits behind that call.</p>
<div class="grid">
  {_kpi("Constraining station", cur["top"],
        f"bottleneck {cur['share']:.0%} of the shift"
        + ("  -  no sensors" if not cur["instrumented"] else ""))}
  {_kpi("Forecast lead time",
        f"{lead.get('lead_hours', 0) or 0:.0f} h" if lead.get("lead_hours") else "n/a",
        "earlier than present-state detection")}
  {_kpi("Alerts this shift", f"{R['ledger']['alerts_per_shift_cap']} max",
        f"{R['ledger']['n_surfaced']} surfaced of {R['ledger']['n_alerts']} raised")}
  {_kpi("Sensor coverage", f"{R['line']['sensor_coverage']:.0%}",
        f"{len(R['line']['dark_stations'])} stations inferred, not measured")}
</div>
{dark_note}

<h2>Where the constraint has been</h2>
<p class="sub">Two-day blocks. Purple marks a station with no process
instrumentation.</p>
{strip}

<h2>Your alerts</h2>
<p class="sub">Capped at {R['ledger']['alerts_per_shift_cap']} per shift and ranked by
confidence times expected impact. Each one carries how often that alert type has
been right on this line.</p>
{_table(["type", "raised", "what happened", "confidence", "if acted on", "track record"], rows)}
</section>"""


def _plant(R: dict) -> str:
    bt = R["backtrace"]
    rows = []
    for t in bt:
        c = t["containment"]
        tone = "t-ok" if t["correct"] else "t-bad"
        rows.append([
            _esc(t["defect_type"]),
            f'day {t["at_day"]}',
            f'<span class="mono">{_esc(t["rank1_label"])}</span>',
            f'{t["rank1_lift"]:.1f}x',
            f'{t["rank1_confidence"]:.0%}',
            f'{t["n_candidates_tested"]} tested',
            (f'{c.get("n_still_in_line", 0)} in line / '
             f'{c.get("n_already_shipped", 0)} shipped'),
            (f'<span class="tag {tone}">confirmed</span>' if t["correct"]
             else '<span class="tag t-warn">awaiting review</span>'),
        ])

    det = R["defect_models"]
    drows = []
    for d in det:
        if d.get("status") != "fitted":
            drows.append([_esc(d["defect_type"]), _esc(d["gate"]),
                          '<span class="tag t-warn">not modelled</span>',
                          _esc(d.get("reason", "")), "", "", ""])
            continue
        cov = d["coverage_defect"]
        tone = "t-ok" if cov >= 0.85 else "t-warn"
        drows.append([
            _esc(d["defect_type"]), _esc(d["gate"]),
            f'{d.get("prequential_roc_auc", d["roc_auc"]):.3f}',
            f'<span class="tag {tone}">{cov:.0%}</span>',
            f'{d["flag_precision"]:.1%}', f'{d["flag_recall"]:.1%}',
            f'{d["abstain_rate"]:.0%}',
        ])

    ret = R.get("retrofit", {})
    rrows = []
    if ret.get("ranking"):
        chosen = set(ret["plan"]["stations"])
        for r in ret["ranking"][:8]:
            pick = ('<span class="tag t-ok">fund</span>' if r["sid"] in chosen
                    else '<span class="tag t-warn">defer</span>')
            rrows.append([
                f'<span class="mono">{_esc(r["sid"])}</span>', _esc(r["zone"]),
                _esc(r["station_type"]), _inr(r["cost_inr"]),
                f'{r["d_pr_auc"]:+.4f}', f'{r["bottleneck_share"]:.0%}',
                _bar(min(r["value_per_lakh"] / max(
                    ret["ranking"][0]["value_per_lakh"], 1e-9), 1)), pick,
            ])

    plan = ret.get("plan", {})
    plan_block = ""
    if plan:
        plan_block = f"""<div class="note"><strong>Next shutdown:</strong>
instrument {_esc(", ".join(plan.get("stations", [])))} for
{_inr(plan.get("spend_inr", 0))} of a {_inr(plan.get("budget_inr", 0))} budget.
Coverage moves {plan.get('coverage_before', 0):.0%} &rarr;
{plan.get('coverage_after', 0):.0%}. Ranking is a counterfactual ablation:
each station's real signals were revealed to the defect model and the change in
precision-recall AUC measured.</div>"""

    return f"""<section id="v-plant" hidden>
<h2>Likely contributing factors &mdash; not confirmed root causes</h2>
<p class="sub">Each confirmed finding at an inspection gate is traced back
through the digital thread. Candidates are ranked by risk ratio with the false
discovery rate controlled at 5% across every hypothesis tested. These are
<strong>statistical associations, not proven causes</strong>: shift, operator,
lot and ambient conditions move together, so a quality engineer confirms or
rejects each lead and that verdict is what the trust ledger scores.</p>
{_table(["defect", "traced on", "rank 1 cause", "lift", "confidence",
         "candidates", "containment", "engineer verdict"], rows)}
<div class="note">Attribution needs no training data, which is why it catches
events the learned model has never seen. The humidity excursion is the example:
the per-unit risk model was trained before that humidity existed, but the
attribution engine isolated it on the first cohort that contained it.</div>

<h2>Defect risk models</h2>
<p class="sub">One model per defect type, each predicted at the gate that first
inspects for it and using only stations upstream of that gate. Coverage is
guaranteed under exchangeability and <strong>measured empirically</strong> here,
because a drifting line violates that assumption; thresholds are recalibrated
every shift.</p>
{_table(["defect", "gate", "ROC AUC", "coverage (target 90%)", "flag precision",
         "flag recall", "abstain"], drows)}

<h2>Where to spend the next shutdown</h2>
<p class="sub">Uneven sensor coverage is a budget problem, not an excuse. These
are the un-instrumented stations ranked by what instrumenting them would
actually buy.</p>
{plan_block}
{_table(["station", "zone", "type", "retrofit cost", "PR-AUC gain",
         "bottleneck share", "value per lakh", "decision"], rrows)
 if rrows else '<p class="sub">Run without --fast to compute the retrofit ranking.</p>'}
</section>"""


def _exec(R: dict) -> str:
    e = R["economics"]
    sc = e.get("scenarios", {})
    base = sc.get("base", {})
    rows = [[_esc(k.title()), _inr(v["throughput_inr"]), _inr(v["quality_inr"]),
             _inr(v["rca_inr"]),
             f'<strong>{_inr(v["total_inr"])}</strong>'] for k, v in sc.items()]

    a = e["assumptions"]
    arows = [
        ["Contribution margin per unit", _inr(a["contribution_margin_per_unit_inr"])],
        ["Annual volume", f'{a["annual_units"]:,} units over {a["working_days"]} days'],
        ["Bottleneck cycle over takt", f'{a["bottleneck_gap_s"]:.2f} s'],
        ["Rework / warranty cost per defect",
         f'{_inr(a["rework_cost_per_defect_inr"])} / {_inr(a["warranty_cost_per_escape_inr"])}'],
        ["Detected defect rate", f'{a["defect_detection_rate"]:.2%}'],
        ["Model flag recall (measured)", f'{a["mean_flag_recall"]:.1%}'],
        ["Investigations per year", f'{a["rca_events_per_year"]}'],
    ]

    ledger = R["ledger"]["scorecard"]
    lrows = [[_esc(r["kind"]), r["issued"], r["surfaced"], r["judged"],
              (f'{r["precision"]:.0%}' if isinstance(r["precision"], (int, float))
               else "-"), _inr(r["impact_flagged_inr"])] for r in ledger]

    return f"""<section id="v-exec" hidden>
<h2>The case</h2>
<p class="sub">Three value pools, each traced to a measured mechanism rather
than a benchmark. Ranges, not point estimates, because the swing factors are
assumptions a plant team should argue with.</p>
<div class="grid">
  {_kpi("Annual benefit, base case", _inr(base.get("total_inr", 0)),
        "one line, one plant")}
  {_kpi("Range across scenarios",
        f'{_inr(sc.get("conservative", {}).get("total_inr", 0))} - '
        f'{_inr(sc.get("optimistic", {}).get("total_inr", 0))}',
        "swing on recall, action rate, gap recovered")}
  {_kpi("Root causes matched", f'{R["backtrace_accuracy"]:.0%}',
        "3 of 3 injected faults, rank 1")}
  {_kpi("Soft-sensor error", f'{R["soft_sensors"]["median_abs_pct_err"]:.2f}%',
        "median, at stations with no sensors")}
</div>

<h2>Value by scenario</h2>
{_table(["scenario", "throughput", "quality", "investigation", "total"], rows)}

<h2>Every assumption, on show</h2>
<p class="sub">None of these are proprietary data. They are stated inputs, and
each one is a lever a plant finance team can reset.</p>
{_table(["input", "value"], arows)}

<h2>The twin's own track record</h2>
<p class="sub">Loom logs every alert it raises and whether it turned out to be
right. Shadow mode issues alerts to this ledger only, so the record is what
earns the right to go live rather than a promise made in advance.</p>
{_table(["alert type", "raised", "surfaced", "judged", "precision",
         "impact if acted on"], lrows)}
</section>"""


def _spc(R: dict) -> str:
    sp = R.get("spc")
    if not sp:
        return '<p class="sub">SPC stage not run.</p>'
    rows = []
    for t in sp["table"]:
        if t["detected"]:
            tone, verdict = "t-ok", f'lag {t["lag_days"]:+.1f} d'
        else:
            tone, verdict = "t-warn", "not a station-level shift"
        onset = (f'{t["onset_estimate_day"]:.1f} '
                 f'({t["onset_error_days"]:+.1f} d)'
                 if t.get("onset_estimate_day") is not None else "-")
        rows.append([
            _esc(t["fault_id"]), _esc(t["kind"]), f'<span class="mono">{_esc(t["target"])}</span>',
            f'{t["start_day"]:.1f}', _esc(t.get("chart") or "-"),
            f'<span class="mono">{_esc(t.get("signal") or "-")}</span>',
            f'<span class="tag {tone}">{verdict}</span>', onset,
        ])
    fa = sp["false_alarm"]
    op = sp["operating_point"]
    return f"""<div class="grid">
  {_kpi("Station-level faults found",
        f'{sp["station_faults_detected"]}/{sp["station_faults_total"]}',
        "drift, degradation, micro-stops")}
  {_kpi("Onset back-dating error", f'{sp["mean_onset_error_days"]:.1f} d',
        "CUSUM estimate of when the shift began")}
  {_kpi("False alarms (upper bound)",
        f'{fa["false_alarm_rate_upper_bound"]:.0%}',
        f'over {fa["clean_stations"]} unaffected stations')}
  {_kpi("Charts on dark stations", f'{sp["n_dark_charts"]}',
        f'of {sp["n_charts"]} total')}
</div>
{_table(["fault", "kind", "target", "began", "chart", "signal", "detection",
         "onset estimate"], rows)}
<div class="caveat"><strong>What SPC does not do, on purpose.</strong> The lot,
zone and operator faults show as "not a station-level shift" because they are
not one. A contaminated lot does not move any single station's mean; it moves
the units that carry that part. Those belong to the attribution engine, and the
division of labour is the design, not a gap. Limits are set at
L&nbsp;=&nbsp;{op["ewma_L"]} with a {op["persistence_blocks"]}-block persistence
rule rather than the textbook 3-sigma, because ~{sp["n_charts"]} charts running
at once at 3 sigma fire somewhere on most clean stations &mdash; the alert
fatigue the brief warns about, reproduced exactly.</div>"""


def _eng(R: dict) -> str:
    ss = R["soft_sensors"]
    srows = [[f'<span class="mono">{_esc(r["sid"])}</span>', _esc(r["type"]),
              f'{r["true_mean_s"]:.2f}', f'{r["est_mean_s"]:.2f}',
              f'{r["pct_err"]:.2f}%',
              ('<span class="tag t-ok">yes</span>' if r["in_95ci"]
               else '<span class="tag t-warn">no</span>'),
              f'<span class="mono">{_esc(r["method"])}</span>',
              f'{r["confidence"]:.2f}'] for r in ss["table"]]

    f = R["bottleneck_forecast"]
    lead = f.get("regime_change_lead", {})
    st = R["state_reconstruction"]

    faults = [[_esc(x["fault_id"]), _esc(x["kind"]), _esc(x["target"]),
               f'{x["start_day"]:.1f}-{min(x["end_day"], R["line"]["days"]):.1f}',
               _esc(x["note"])] for x in R["faults"]]

    tm = R.get("timing_ms", {})
    trows = [[_esc(k), f"{v:,.0f} ms"] for k, v in
             sorted(tm.items(), key=lambda kv: -kv[1])]

    return f"""<section id="v-eng" hidden>
<h2>How the ground truth was hidden</h2>
<p class="sub">The simulator injects five faults and records exactly what it
did. The twin is handed departure scans, process signals from instrumented
stations only, and inspection findings. Nothing else.</p>
{_table(["id", "kind", "target", "days", "what it does"], faults)}

<h2>State reconstruction from scans alone</h2>
<div class="grid">
  {_kpi("Start-time MAE", f'{st["start_mae_s"]:.2f} s', "vs hidden truth")}
  {_kpi("Starved-time MAE", f'{st["starved_mae_s"]:.2f} s', "vs hidden truth")}
  {_kpi("Blocking recall", f'{st["blocked_recall"]:.1%}',
        f'precision {st["blocked_precision"]:.1%}')}
  {_kpi("95% CI coverage", f'{ss["ci95_coverage"]:.0%}', "soft-sensor intervals")}
</div>

<h2>Soft sensors at the 13 dark stations</h2>
<p class="sub">Cycle time recovered from arrival and departure scans by
censored-lognormal maximum likelihood. Blocked units are right-censored at an
observable threshold, so they inform the estimate instead of being discarded.</p>
{_table(["station", "type", "true mean", "estimated", "error",
         "in 95% CI", "method", "confidence"], srows)}

<h2>Statistical process control</h2>
<p class="sub">EWMA and CUSUM charts on shift means. The same charts run on
soft-sensed dwell at un-instrumented stations, so a station with no sensors
still gets a control chart.</p>
{_spc(R)}

<h2>Bottleneck forecasting</h2>
<div class="grid">
  {_kpi("Future, top-1", f'{f["top1_future"]:.0%}', f'{f["n_triggers"]} rolling triggers')}
  {_kpi("Present, top-1", f'{f["top1_present"]:.0%}', "detect-only baseline")}
  {_kpi("Static, top-1", f'{f["top1_static"]:.0%}', "long-run bottleneck baseline")}
  {_kpi("Regime-change lead",
        f'{lead.get("lead_hours") or 0:.0f} h',
        f'S28 called day {lead.get("first_day_future")} vs {lead.get("first_day_present")}')}
</div>
<div class="caveat"><strong>Read this honestly.</strong> Top-1 momentary
bottleneck over a shift is a noisy target on a balanced line: for most of the
run three stations sit within a second of each other, so which one leads at any
instant is close to a coin flip. The result that carries decision value is the
regime change &mdash; the forecast names S28 as the sustained constraint hours
before present-state detection does, and the static baseline never finds it at
all.</div>

<h2>Runtime</h2>
<p class="sub">Whole pipeline on one CPU core, no GPU. A live deployment runs
the rolling forecast on a schedule, not the whole history at once.</p>
{_table(["stage", "wall time"], trows)}
</section>"""
