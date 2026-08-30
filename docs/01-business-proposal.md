# Loom — Business Proposal

Accenture Innovation Challenge 2026 · Round 2 · Track 4 `DigitalTwin.ai`

---

## 1. Problem framing

### 1.1 What the plant actually experiences

Three things go wrong on a vehicle assembly line, and they go wrong in a
specific and unhelpful order.

**A constraint moves and nobody notices.** A station drifts slower by half a
percent a day. Nothing alarms, because half a percent is inside normal
variation. Six weeks later the line is running 40 minutes of unplanned overtime
per shift and the reason is buried in a station that was never the bottleneck
before.

**A defect is created early and found late.** A torque tool wanders out of
spec in the body shop. The vehicles that receive a loose fastener continue
through paint and final assembly. Forty minutes and thirty stations later,
final inspection finds the first one. By then several hundred more units carry
the same fault, and the units built in the last hour are already on trucks.

**The investigation costs more than the defect.** A quality engineer opens a
spreadsheet, pulls process data from four systems with different timestamps and
granularities, and spends most of three days establishing what a stronger
system would have said in seconds. Meanwhile the corrective action waits.

### 1.2 Why existing digital twins do not fix this

The literature is unambiguous about the obstacles, and they are not modelling
obstacles.

Kattenstroth et al. (2024) reviewed 25 digital-factory-twin use cases and found
19 recurring challenges. The most frequently cited was not model accuracy; it
was *the effort of connecting and pre-processing data between the real system
and the model*, followed by high initial creation effort, the need for
simulation experts, and models that go stale because keeping them current is
manual. Their conclusion is blunt: 19 of 25 simulation models were never reused
at all.

Ragazzini et al. (2024) showed that predicting the bottleneck and adapting
production control to it beats reacting to the current one — throughput
improved over 10% with the right dispatching rule. But their framework assumes
data can be acquired *from each workstation on a real-time basis*. That
assumption is exactly what a real plant cannot supply.

Wang et al. (2025) built a manufacturing knowledge graph and a graph neural
network that predicts anomaly events 25.6% better than baselines. They close by
naming their own limits: the model needs large volumes of accurately labelled
data, training is slow, and *the interpretability of the model is poor, making
it difficult to explain the prediction results*. On a factory floor, an
unexplainable alert is an ignored alert.

Shao (2024), reviewing ISO 23247 adoption across 29 architectures from 140
papers, found that implementations concentrate on functional capability and
neglect the rest. Not one implemented Data Assurance; only one implemented
Access Control.

### 1.3 The gap we target

Put those together and the shape of the gap is clear.

> Existing twins predict well **where instrumentation is dense**, explain
> poorly **when the event is new**, and quietly assume a coverage level that
> the brief itself says does not exist.

Our reference line makes this concrete: 42 stations, 13 of them with no process
sensors at all. Under a conventional approach those 13 stations are simply
absent from the model. One of them is the station that becomes the bottleneck
in the second half of our run.

---

## 2. Solution design

### 2.0 Scope

Serial and near-serial assembly flows with unit-level traceability. Rework
loops, parallel banks and overtaking are out of scope for this prototype and
named as future work rather than glossed. Clock synchronisation across read
points (NTP, ~50 ms) is a commissioning prerequisite.

### 2.1 Design principles

1. **Degrade, do not go blind.** Every capability must return something useful
   at a sensor-poor station, with a stated confidence, rather than silently
   dropping it.
2. **Two independent paths to an explanation.** A learned model for recurring
   patterns; a training-free statistical engine for events nobody has seen.
   They fail differently, which is the point.
3. **Attention is the scarce resource.** Alerts are capped and ranked. The
   system publishes its own hit rate.
4. **Read-only at the OT boundary.** No PLC writes, ever.
5. **The line is configuration, not code.** Porting to a second plant is a
   config file.

### 2.2 The four layers

**Layer 1 — Acquisition (read-only).** An edge gateway subscribes to OPC UA
nodes and MQTT topics, with store-and-forward through network drops. Nothing
writes back. Manual checklist entries arrive from tablets. This layer's only
job is to timestamp, identify and normalise.

**Layer 2 — Twin core.** Four components:

- *State reconstruction.* Departure scans plus the release schedule yield
  `working / blocked / starved` per station per unit. This is the load-bearing
  idea: those three states are exactly what the bottleneck method consumes, and
  they are recoverable without a single process sensor.
- *Soft sensors.* At a dark station, `dwell = processing + blocked`. A blocked
  unit departs at the instant a downstream slot frees, so observations split
  into exact cycle times and upper bounds at observable thresholds. Maximum
  likelihood on that censored mixture recovers cycle time to **0.011% median
  error**, with a confidence that falls as censoring rises.
- *Knowledge graph.* The digital thread: unit → station visit → station, lot,
  operator, shift, inspection, signal. Schema as a graph, instance data as an
  interval index.
- *Rolling-horizon simulation.* A discrete-event kernel regenerated from the
  config, re-parameterised each cycle from the soft sensors, seeded from
  observed work-in-process, with per-station drift extrapolated across the
  horizon so a slow degradation is caught while it is still slow.

**Layer 3 — Analytics.**

- *Bottleneck.* Roser's active-period method applied to the **simulated
  future**, as in Ragazzini et al., but with dark stations represented rather
  than dropped. Five replications vote; the vote share is the confidence.
- *Defect risk.* One gradient-boosted model per defect type, each predicted at
  the gate that first inspects for it and using only stations upstream of that
  gate. Wrapped in **class-conditional conformal prediction**, so the output is
  `flag` / `abstain` / `pass` with a coverage guarantee rather than a score.
- *Genealogy back-trace.* Contingency tests over the thread with
  Benjamini-Hochberg FDR control, and consolidation of zone-wide effects so a
  utility fault does not get attributed to four separate robots.
- *Sensor ROI.* Counterfactual ablation ranking the retrofit queue.

**Layer 4 — Decision and users.** Alert budget, trust ledger, shadow mode, and
three persona views over one payload.

### 2.3 What the prototype demonstrates

Five faults are injected into the simulated plant and hidden from the twin:

| | Fault | What it tests |
|---|---|---|
| F1 | Torque tool at S12 drifts 0.30 Nm/day from day 9 | latent defect, created in body shop, found at final inspection |
| F2 | Un-instrumented S28 degrades 0.75%/day from day 6 | bottleneck moves to a station with no sensors |
| F3 | Supplier lot LOT-425 carries 6.2× defect propensity | attribution to a material source, not a station |
| F4 | Paint-zone humidity rises 14 points for 1.5 days | confounder: a zone-wide cause that looks like four station faults |
| F5 | Intermittent micro-stoppages at instrumented S07 | SPC on a station that does have sensors |

Results, means over three seeds:

- Bottleneck moves S33 → S28 at day 12; forecast calls it **16 h ± 11 h**
  before present-state detection, and the static baseline never calls it.
- All three probed faults identified at **rank 1**, with lifts of 4.3×, 30.3×
  and 30.6×.
- The humidity event is correctly reported as a **zone-level** cause.
- Containment names **39 vehicles still on the line** carrying the same
  exposure as a confirmed finding.

---

## 3. Target users

| User | Decision | What Loom gives them | Cadence |
|---|---|---|---|
| **Line supervisor** | Where to put maintenance attention this shift | The constraining station with confidence, capped at 5 alerts, each showing how often that alert type has been right | Live |
| **Quality engineer** | What caused this finding and which units share it | Ranked hypotheses with lift, q-value and evidence; a VIN list for containment | On finding |
| **Plant manager** | Where the week went; what to fund at the shutdown | Bottleneck history, defect model performance, retrofit plan against a budget | Weekly |
| **Manufacturing IT / OT** | Whether this is safe to connect | Read-only boundary, ISO 23247 mapping, no PLC writes, audit trail | Once, then on change |
| **Operations leadership** | Whether to fund the rollout | Value by scenario with every assumption shown, and the twin's own track record | Quarterly |

The supervisor is the make-or-break user. If the floor ignores the alerts,
every other number is theoretical. That is why the alert budget and the trust
ledger are core architecture, not reporting features.

---

## 4. Business case

### 4.1 Three value pools

**Throughput.** When the constraint drifts past takt, the line loses units.
Our injected degradation puts S28 roughly 1 s over a 60 s takt by day 20. At
960 units/day across two shifts, closing 60% of that gap recovers ~4.7 units
per shift.

**Quality.** Earlier detection converts warranty escapes into in-plant rework,
and containment shrinks the affected population from a date-range sweep to a
VIN list.

**Investigation.** Root-cause traces run in ~130 ms against ~200 candidate
hypotheses. We assume 70% of a 22-hour investigation is saved, on 26
investigations a year.

### 4.2 Result, one line

| Scenario | Throughput | Quality | Investigation | **Total / year** |
|---|---|---|---|---|
| Conservative | ₹6.08 cr | ₹1.43 cr | ₹0.06 cr | **₹7.59 cr** |
| Base | ₹10.43 cr | ₹4.77 cr | ₹0.06 cr | **₹15.26 cr** |
| Optimistic | ₹13.90 cr | ₹6.68 cr | ₹0.06 cr | **₹20.61 cr** |

Scenarios swing on three assumptions, all visible in the dashboard: the
fraction of the bottleneck gap actually recovered (35 / 60 / 80%), the fraction
of flags acted on (30 / 50 / 70%), and measured model recall.

### 4.3 Cost

| Item | Year 1 | Steady state |
|---|---|---|
| Edge gateway + read-only OPC UA/MQTT tap, one line | ₹18 L | ₹3 L |
| Integration and semantic modelling (config, not code) | ₹32 L | ₹6 L |
| Sensor retrofit, phase 1 (5 stations, from the ablation) | ₹6.2 L | — |
| Platform and support | ₹14 L | ₹14 L |
| **Total** | **₹70.2 L** | **₹23 L** |

Against the conservative case, payback lands inside the first year on a single
line. We lead with the conservative figure deliberately; a proposal that only
works on optimistic assumptions is not a proposal.

### 4.4 What would falsify this

Stating this up front is part of the case. The business case fails if the floor
ignores the alerts (mitigated by the budget and ledger), if the bottleneck on a
given line never moves (then throughput value is near zero and only quality
value remains), or if defect mechanisms are dominated by causes with no
observable signature (measured directly by the conformal abstention rate — if
abstention runs above ~85%, the line is telling you the sensors are in the
wrong places, which is itself an actionable finding).

---

## 5. Phased roadmap

**Phase 0 — Shadow, weeks 1–8, one line.**
Read-only tap. The twin predicts and logs to the trust ledger; nothing reaches
the floor. Exit criterion: bottleneck alert precision above 0.6 and calibration
gap under 0.15 on that plant's own data. This phase exists because it is the
only honest way to earn a supervisor's attention.

**Phase 1 — Assist, weeks 9–20, same line.**
Alerts go live under a 5-per-shift budget. Root-cause traces open to the
quality team. The first retrofit batch is installed at the scheduled shutdown,
chosen by the ablation ranking. Exit criterion: measured throughput or
first-time-through improvement, plus supervisor override rate below 40%.

**Phase 2 — Plant, months 6–12.**
Remaining lines onboarded through the station-type library. Cross-line
comparison. Semantic layer hardened to ISO 23247 with Data Assurance and
Access Control implemented — the entities Shao (2024) found are usually
skipped. Exit criterion: a second line onboarded in under three weeks with no
new modelling code.

**Phase 3 — Multi-site, year 2.**
The station-type library becomes a shared asset. A new plant configures rather
than rebuilds. Retrofit rankings are pooled across sites, so a plant with
sparse instrumentation inherits priors from one with dense instrumentation.

Each phase is separately cancellable and each produces a standalone artefact.
Phase 0 produces a trust ledger that is valuable even if the project stops.

---

## 6. Key risks and mitigations

| Risk | Why it is real | Mitigation | Residual |
|---|---|---|---|
| **Floor stops trusting alerts** | The brief names this; false alarms erode trust faster than true positives build it | Hard alert budget; published running precision; shadow mode before go-live; conformal abstention instead of guessing | Medium — mitigated by design, not eliminated |
| **OT security / safety** | Plants refuse writes into line control, correctly | Read-only by architecture. Device Control sub-entity not implemented. One-way data diode where policy requires | Low |
| **Model drift** | Line changes model mix, tools are replaced, seasons change | Prequential evaluation; conformal recalibration every shift; drift monitors on input distributions; scheduled retrain | Medium — coverage still degrades to ~89% under drift; we report it |
| **Conformal guarantee breaks under shift** | Split conformal assumes exchangeability, which drift violates | Rolling recalibration; report measured coverage next to target, never assume it | Medium — a known open limit |
| **Sparse instrumentation worse than modelled** | Some plants scan only at zone boundaries, not per station | Soft sensors degrade to segment-level estimates with wider intervals and lower confidence; the retrofit ranking prioritises scan points | Medium |
| **Integration effort overruns** | The single most cited challenge in the literature | Station-type library; config-driven line definition; the DES is generated, never hand-built | Medium |
| **Attribution finds a correlate, not a cause** | Operators, shifts and lots are correlated with time | FDR control across all candidates; zone and time-window consolidation; UI labels output "likely contributing factors, not confirmed root causes"; engineer confirms or rejects and the ledger scores that verdict | Medium — Loom ranks hypotheses, it does not prove causation |
| **Sensor ROI needs data the plant does not have** | The ablation reveals ground truth a real plant cannot access | Two-stage method: deployable EVI ranking reproduces the throughput half exactly (rho 1.00) but not the quality half (rho -0.06), so a temporary instrumented pilot measures the rest before the budget is committed | Low — measured, not assumed |
| **Dirty scan feed** | Missed, duplicate and late reads are routine | Data-quality layer detects and repairs; soft-sensor error 1.67% to 0.22%; confidence downgraded by feed quality | Low |
| **Simulated validation does not transfer** | Our ground truth is synthetic | Phase 0 exists precisely to re-measure every claim on real data before anything goes live | High until Phase 0 completes |

---

## 7. Why this wins

Three things a reviewer can check in the repository:

1. **It works where the sensors are not.** 31% of the reference line is dark,
   the bottleneck moves into that region, and Loom follows it. Most submissions
   will assume coverage the brief explicitly says is uneven.
2. **Every claim is scored against hidden ground truth.** `eval/run_eval.py`
   runs multiple seeds and exits non-zero if any headline metric leaves its
   stated envelope. The limitations section names the metrics that are weak.
3. **It answers the question the brief asks and the literature leaves open.**
   Ragazzini predicts bottlenecks but not defects. Wang predicts anomalies but
   cannot explain them. Kattenstroth catalogues why twins go stale. Loom is
   built directly on the seam between them.

---

## References

Kattenstroth, F., Disselkamp, J.-P., Lick, J., Dumitrescu, R. (2024).
*Challenges in the implementation of simulation models for the digital factory
twin — a systematic literature review.* Procedia CIRP 128, 442–447.

Ragazzini, L., Negri, E., Fumagalli, L., Macchi, M. (2024). *Digital Twin-based
bottleneck prediction for improved production control.* Computers & Industrial
Engineering 192, 110231.

Roser, C., Nakano, M., Tanaka, M. (2001, 2002). *A practical bottleneck
detection method*; *Shifting bottleneck detection.* Winter Simulation
Conference.

Shao, G. (2024). *Manufacturing Digital Twin Standards.* MODELS Companion '24.
ISO 23247 Parts 1–4.

Siemens (2024). *Supercharging the industry transformation with the
comprehensive Digital Twin.* White paper.

Wang, S., Guo, Y., Huang, S., Lai, R., Zhang, L., Qian, W. (2025). *A deep graph
neural network-based link prediction model for proactive anomaly detection in
discrete manufacturing workshop.* Journal of Manufacturing Systems 79, 301–317.

Additional sources consulted are listed in
[`03-literature-and-gaps.md`](03-literature-and-gaps.md).
