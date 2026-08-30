# Literature and gap analysis

What the five supplied papers establish, what the wider literature adds, and
precisely where Loom goes past both.

---

## 1. The five supplied papers

### Ragazzini, Negri, Fumagalli & Macchi (2024) — DT-based bottleneck prediction
*Computers & Industrial Engineering 192, 110231*

**Establishes.** A four-block framework — data processing, digital twin,
bottleneck prediction, decision support — in which the DT runs short
simulations on a rolling horizon and the active-period method is applied to the
*synthetic future* state log rather than to history. Validated on a 7-station
lab line. Adapting order release and dispatching to the predicted bottleneck
improved throughput over 10% with the right dispatching rule (CLES > 0.90).

**Limits they state.** Lab environment only; few order-release policies and
dispatching rules tested; and explicitly: *"As Artificial Intelligence
approaches to bottleneck prediction are developed, their application to
synthetic data to predict future bottlenecks may be investigated."*

**Limit they do not state.** The framework *"relies on the possibility to
acquire data from each workstation on a real-time basis."* On a 42-station
line where 13 stations have no sensors, the framework as specified cannot run.

**What Loom does with it.** Adopts the rolling-horizon structure and the
active-period method wholesale — this is the right approach and we say so. Then
changes three things: cycle-time distributions are re-estimated each cycle from
soft sensors so dark stations participate; per-station drift is extrapolated
across the horizon so a slow degradation is caught while still slow; and five
replications vote, with the vote share serving as the forecast confidence. On
the S33 → S28 regime change the forecast leads present-state detection by
16 h ± 11 h, and the station it identifies has no sensors at all.

### Wang, Guo, Huang, Lai, Zhang & Qian (2025) — LAGNN link prediction
*Journal of Manufacturing Systems 79, 301–317*

**Establishes.** A manufacturing knowledge graph built from an OPC UA
information model, BERT extraction over process documents, and OWL semantic
mapping. Anomaly-event prediction is recast as link prediction; a deep
autoencoder with local graph learning plus an attention Seq2Seq beats 15
baselines by ≥25.6% MRR. Real predictions such as *"buffer IB05 will be blocked
at 11:50:42"* landed within 14 seconds.

**Limits they state.** Training is slow; *"the training process requires a large
amount of accurately labelled data"*; and *"the interpretability of the model is
poor, making it difficult to explain the prediction results."* Their proposed
future work is an evolutionary model to expose how anomalies develop.

**What Loom does with it.** Keeps the knowledge-graph framing and the digital
thread. Rejects the model choice for this problem. Two reasons: on a novel
event there is no labelled data by definition, and an unexplainable alert on a
factory floor is an ignored alert. Loom instead runs contingency tests over the
same thread — no training, exact p-values, FDR control — and pairs it with a
small conformal-wrapped model for recurring patterns. The paint humidity
excursion is the demonstration of why: the learned model was trained before
that humidity existed and largely missed it, while the training-free engine
isolated it on the first cohort that contained it.

We also depart on implementation. Wang et al. materialise the graph as
instances; at 800k station visits that is the wrong shape for the queries that
matter. Loom keeps the schema as a graph and the instance data as an interval
index.

### Kattenstroth, Disselkamp, Lick & Dumitrescu (2024) — DES for the digital factory twin
*Procedia CIRP 128, 442–447*

**Establishes.** A systematic review: 25 use cases, 19 challenges, 17
requirements. The most cited challenge is the effort of connecting and
pre-processing data between the real system and the DES. Others: high initial
creation effort, experts needed, models going stale, hardcoded models needing
manual adaptation, and low SME acceptance. 19 of 25 simulation models were
never reused.

**What Loom does with it.** Treated as a requirements specification rather than
a background citation. Their requirements map directly onto design decisions:

| Their requirement | Loom |
|---|---|
| Modularisation and parameterisation | station-type library; a line is config |
| DES automatically derived from a process graph | kernel generated from the config, never hand-built |
| Synchronise the model with real factory data | rolling-horizon sync from observed WIP |
| Keep the model up to date | cycle profiles re-estimated every cycle; drift extrapolated |
| Automatic triggers for anomaly detection | conformal flag/abstain plus SPC |
| Reuse across lines | station types are the cross-plant asset |
| Real-time capability | ~700 ms per forecast cycle |
| User-friendly dashboard | three persona views over one payload |

Two of their requirements we do **not** meet: 3D visualisation and supplier
integration. Both are real and both are out of scope for a prototype; they are
named here rather than quietly dropped.

### Shao (2024) — Manufacturing Digital Twin Standards
*MODELS Companion '24; NIST*

**Establishes.** ISO 23247's three domains and functional entities. Critically,
citing Ferko et al.: across 29 architectures from 140 papers, current
implementations *"mostly focus on functional aspects, neglecting non-functional
entities related to, e.g., security and maintainability."* None implemented
Plug-and-Play Support, Peer Interface or Data Assurance; only one implemented
Access Control. Data Storage and Digital Twin Versioning are used in practice
but absent from the reference architecture.

**What Loom does with it.** Uses ISO 23247 as the architectural frame and
targets the neglected entity. **Data Assurance** is the one this brief most
needs — the brief's own language about false alarms eroding floor trust is a
data-assurance problem — and it is the one nobody implements. Loom's soft-sensor
confidence, conformal coverage measured against target, FDR-controlled
attribution and trust ledger are all Data Assurance. The full mapping,
including where we deliberately decline (Device Control), is in
`02-architecture.md`.

### Siemens (2024) — Comprehensive Digital Twin white paper

**Establishes.** Vocabulary and framing: Digital Twin of Product / Production /
Performance, the Digital Thread as a semantically integrated data flow, closed
loop optimisation, virtual commissioning, and the **executable digital twin** —
a lean, self-contained, reduced-order simulation component that runs on local
compute without simulation expertise.

**What Loom does with it.** Adopts the digital-thread framing and, more
substantively, the executable-twin posture: Loom's DES kernel is dependency-free,
runs on one CPU core, and is regenerated from config rather than authored in a
simulation package. That is what makes per-shift re-parameterisation practical
and is the direct counter to Kattenstroth's staleness problem. We take the
vocabulary; the paper is a vendor white paper and carries no evaluation, so we
take no empirical claims from it.

---

## 2. Wider literature consulted

Retrieved to check whether the gaps we identified are genuinely open.

**Digital twins and knowledge graphs.** Su et al. (2025), *Digital twin system
for manufacturing processes based on a multi-layer knowledge graph model*,
Scientific Reports 15 — a three-layer concept/model/decision KG architecture,
validated on aero-engine blades over five months (qualification rate 81.3% →
85.2%). Confirms the KG-plus-twin direction; still assumes instrumented
processes. Also a 2026 spatio-temporal GNN surrogate for throughput estimation
in general assembly lines (R² 0.80 at 30-minute aggregation), which is a
throughput surrogate rather than a bottleneck or defect attributor.

**Legacy and sparse instrumentation.** *IIoT-enabled digital twin for legacy and
smart factory machines with LLM integration*, JMS (2025) — MTConnect streaming
from legacy artefacts such as seven-segment displays and toggle switches; and
Durigan et al., *On the potential of low-cost instrumentation for digitalization
of legacy machine tools*, IJAMT 128 (2023). Both address getting *some* data off
old machines. Neither addresses what to do at a station where you will get none,
which is the case Loom targets.

**Genealogy and traceability.** Well established as industrial practice —
bidirectional lot and unit genealogy, containment narrowed from a date-range
sweep to a VIN list. What the practice literature describes as a lookup, Loom
treats as a ranked statistical inference with FDR control and confounder
consolidation.

**Conformal prediction in manufacturing.** Growing but still thin. A 2026 arXiv
study on predictive quality in semiconductor materials benchmarks MLOps
retraining strategies with conformal prediction and finds a fixed retraining
cadence outperforms hyperparameter retuning under drift — which is exactly the
rolling-recalibration design Loom uses, and independently confirms the coverage
degradation we measure. Also: conditional conformal for false-alarm control in
fault detection (Control Engineering Practice, 2025) and conformal segmentation
for industrial surface defects (2025). None of these are applied to
assembly-line bottleneck or defect attribution.

---

## 3. The gap, stated precisely

| Capability | Ragazzini 2024 | Wang 2025 | Kattenstroth 2024 | Loom |
|---|---|---|---|---|
| Bottleneck prediction | yes | partial | reviewed | yes |
| Defect prediction | no | partial | no | yes, per type |
| Works at un-instrumented stations | **no** | no | named as a problem | **yes** |
| Root-cause attribution of a confirmed finding | no | **no** | no | **yes** |
| Handles a zone-wide confounder | no | no | no | **yes** |
| Calibrated confidence with a guarantee | no | no | no | **yes** |
| Needs no labelled data for novel events | n/a | **no** | n/a | **yes** |
| Sensor investment ranking | no | no | no | **yes** |
| Scored against injected ground truth | partial | partial | n/a | **yes** |

The three genuinely novel contributions:

1. **Cycle time at a blocked, un-instrumented station is a censored-MLE problem
   with observable, unit-specific thresholds.** We have not found this
   formulation in the assembly-line literature. It is what lets every
   downstream capability keep working through a sensor gap.

2. **Two independent explanation paths with complementary failure modes.** A
   conformal-wrapped learned model for recurring patterns; a training-free,
   FDR-controlled attribution engine for novel events. The humidity excursion
   shows why one is not enough.

3. **Sensor coverage as a ranked capital decision, computed by counterfactual
   ablation.** The literature treats uneven coverage as a limitation to note.
   We treat it as the output.
