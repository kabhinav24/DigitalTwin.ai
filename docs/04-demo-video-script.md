# Demo video script (4:30)

Record the terminal and the dashboard. Do not narrate code. Every number spoken
must be visible on screen at that moment.

---

**0:00–0:25 — The problem, concretely**

> "This is a vehicle assembly line: 42 stations across body, paint and final.
> Thirteen of them have no process sensors at all — just a scanner at the
> station boundary. That's 31% of the line invisible to the plant historian.
> Watch what happens when the bottleneck moves into that region."

*Show:* `docs` diagram, or the dashboard header with `69% instrumented`.

**0:25–1:00 — Run it**

```bash
python run_demo.py
```

> "Everything you're about to see is scored against ground truth the twin never
> sees. The simulator injects five faults and hides them."

*Show:* the fault table scrolling past, then the bottleneck timeline printing.

> "Day 0 to 11 the constraint is S33. From day 12 it's S28 — and S28 is one of
> the dark stations."

**1:00–1:45 — The soft sensor**

*Show:* the Validation tab, soft-sensor table.

> "S28 has no sensors, so its cycle time is reconstructed from arrival and
> departure scans. The trick is that blocking has a signature: a blocked unit
> departs at the exact instant a slot frees downstream. So every observation is
> either an exact cycle time or an upper bound at a threshold we can see. That's
> a censored-likelihood problem, and it recovers cycle time to 0.011% median
> error across all 13 dark stations."

*Point at:* S25 — 0.79% error, confidence 0.70.

> "S25 sits behind the bottleneck and is blocked most of the time. The estimate
> holds, and the confidence correctly drops. It tells you when to trust it."

**1:45–2:30 — Forecasting the regime change**

*Show:* Validation tab, bottleneck KPIs.

> "Three approaches: static long-run bottleneck, present-state detection, and
> forward simulation. The forecast calls the switch to S28 sixteen hours before
> present-state detection does. The static baseline never finds it."

> "Be honest about the top-1 numbers — 59% versus 63%. For most of the run three
> stations sit within a second of each other, so which one leads at any instant
> is close to a coin flip. The result that matters is the regime change, and we
> show both numbers rather than only the flattering one."

**2:30–3:20 — Root cause, and the confounder**

*Show:* Plant manager tab, back-trace table.

> "Final inspection finds a loose fastener. It was created thirty stations and
> forty minutes earlier. Loom tests about two hundred hypotheses over the
> digital thread — stations, signals, lots, operators, shifts, environment —
> with the false discovery rate controlled at 5%."

> "Rank one: torque at S12 below band. Lift 4.3. That is fault F1, and Loom has
> never been told it exists."

*Point at:* the paint row.

> "This one is the interesting case. A humidity excursion hits the paint shop.
> Every booth in that zone looks guilty. Loom consolidates them and reports a
> zone-level cause — one air handler, not four robots."

> "And the risk model largely missed this event, because it was trained before
> that humidity existed. The attribution engine caught it because it needs no
> training data at all. That's why there are two paths, not one."

**3:20–3:50 — Containment**

*Show:* the containment column.

> "Thirty-nine vehicles still on the line share the same exposure. Loom names
> them. That's the difference between quarantining a date range and
> quarantining a VIN list."

**3:35–3:50 — It survives a dirty feed**

*Show:* the terminal line reporting injected scanner faults.

> "And this whole run is on dirty data. We inject twelve thousand missed scans,
> three thousand duplicates and five thousand late manual scans, then detect and
> repair them. Soft-sensor error goes from one point seven percent unrepaired to
> nought point two two repaired. Confidence is downgraded by feed quality, so an
> estimate built on imputed reads is never shown as if it were clean."

**3:50–4:15 — Where to spend the shutdown**

*Show:* the retrofit table.

> "Uneven coverage is a budget problem. But here's the honest part: our best
> ranking method needs ground truth a real plant doesn't have. So we built a
> deployable version and tested it. The throughput half transfers perfectly —
> correlation one point zero. The quality half doesn't transfer at all.
>
> So the deployment method is two-stage: rank to shortlist, then a temporary
> pilot on the top stations to measure the rest. Both methods independently pick
> the same station first."

**4:15–4:30 — Trust**

*Show:* Leadership tab, trust ledger.

> "Every alert Loom raises is logged with whether it turned out to be right, and
> the floor sees that record next to the alerts. Shadow mode runs for eight
> weeks before anything reaches a supervisor. That is how you earn the attention,
> rather than asking for it."

*End on:* `python eval/run_eval.py` — all envelope checks PASS.

---

## Recording notes

- 1920×1080, terminal at ~16pt, dashboard at 100% zoom.
- Run `python run_demo.py` once before recording so imports are warm.
- Do not speed up the terminal; the 40-second runtime is a selling point.
- Total spoken words ≈ 620, which is 4:30 at a calm pace. Do not rush the
  honest-limitations line at 2:30 — it is the most credible thirty seconds in
  the video.
