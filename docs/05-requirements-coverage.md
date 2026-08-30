# Requirements coverage

Every line of the Round 2 Track 4 brief, mapped to where it is addressed and
how to verify it. Items marked **partial** or **not done** are stated as such
rather than stretched.

---

## Real-world complexities to consider

### 1. Mixed legacy and modern equipment; inconsistent sensor coverage
> *"some stations are richly instrumented, others rely entirely on manual checklists"*

**Covered — this is the centre of the solution.** Plant Alpha runs 42 stations
at 69% coverage; 13 have no process sensors. Cycle time at those stations is
recovered from arrival and departure scans by censored-lognormal MLE to a
median 0.03% error with 92% interval coverage. SPC charts, bottleneck
detection, defect features and root-cause attribution all continue to run
through the gap.

*Verify:* `outputs/results_alpha.json → soft_sensors`, or the Validation tab.

### 2. Multi-causal, intermittent root causes
> *"equipment wear, operator variation, upstream part quality, environmental conditions"*

**Covered — all four, each as a separate injected fault.**

| Cause named in the brief | Fault | Found at rank 1 |
|---|---|---|
| Equipment wear | F1 torque drift, F2 cycle degradation | yes |
| Operator variation | F6 operator FINAL-OP2 | yes, lift 23.0× |
| Upstream part quality | F3 contaminated lot LOT-425 | yes, lift 18.2× |
| Environmental conditions | F4 paint humidity excursion | yes, as a **zone-level** cause |
| Intermittent | F5 micro-stoppages at S07 | yes, via SPC |

Attribution is 4/4 at rank 1 across all three eval seeds. Critically, F6 is
attributed to the *person* and not to the shift or the station they worked,
which is the confounder that makes operator variation hard.

*Verify:* `python eval/run_eval.py` → attribution detail.

### 3. Modifying live production systems carries risk; retrofits only in windows
**Covered.** Read-only by architecture: the ISO 23247 Device Control
sub-entity is deliberately not implemented and the reasoning is documented in
`02-architecture.md`. The retrofit planner sizes its recommendation to a stated
shutdown budget (5 stations, ₹6.2 L, coverage 69% → 81%).

### 4. Defect introduced early surfaces late; downstream units carry it
**Covered — this is the headline demo.** F1 creates a defect at S12 in the body
shop; it is first detected at S42, thirty stations and roughly forty minutes
later. Back-trace identifies S12 at rank 1, and containment names the **39
vehicles still on the line** carrying the same exposure plus the count already
shipped.

### 5. Different stakeholders need different views of the same twin
**Covered.** Four tabs over one payload: floor supervisor (live constraint,
capped alerts), plant manager (root cause, defect models, retrofit plan),
leadership (scenarios, assumptions, track record), and a validation tab for
engineering review.

### 6. Extending beyond a single line or plant
> *"real variation in layout, equipment vintage, and sensor maturity"*

**Covered, and demonstrated rather than asserted.** `python run_demo.py --line
beta` runs the entire pipeline on **Plant Beta**: 28 stations instead of 42,
57% coverage instead of 69%, a 72 s takt instead of 60 s, one shift instead of
two, no paint zone at all, and different gate positions. It is a config file
and zero lines of code. Results: soft sensors 0.01% error, the dark station B14
correctly identified as the bottleneck, and 3/3 root causes at rank 1.

Building this found a real bug — the defect gate map had been hard-coded to
Plant Alpha's station IDs, which would have made the portability claim false.
It is now derived from the config.

*Verify:* `python run_demo.py --line beta` and compare
`outputs/dashboard_beta.html` with `dashboard_alpha.html`.

### 7. Predictive claims must be validated; false alarms erode trust
**Covered, and the trade-off is measured rather than assumed.** Every claim is
scored against ground truth the twin never sees. `eval/run_eval.py` runs
multiple seeds and exits non-zero if any of ten headline metrics leaves a
pre-stated envelope.

On false alarms specifically: a textbook 3-sigma chart across ~270 charts fires
somewhere on **more than half** of all clean stations. That number is in the
repository. Widening limits and adding a persistence rule brings the upper
bound to 32%, and `spc.sweep_operating_points` reports the whole
detection-versus-false-alarm curve so the chosen operating point is a stated
decision. On the product side: a hard alert budget, a trust ledger publishing
running precision per alert type, a calibration table, and an eight-week shadow
mode before anything reaches a supervisor.

---

## Solutioning areas

| Area | Status | Where |
|---|---|---|
| Modelling: explicit vs inferred, especially at sensor-poor stations | done | `twin.py` — cycle time inferred, process signals explicit where they exist |
| Predictive: anomaly detection | done | conformal defect models, `defect.py` |
| Predictive: **statistical process control** | done | `spc.py` — EWMA, CUSUM, stoppage-rate charts, incl. on soft-sensed data |
| Predictive: physics-informed models | **not done** | the blocking-after-service recursion is a structural line model, but no station-level physics |
| Predictive: ML-based bottleneck/defect prediction | done | `bottleneck.py`, `defect.py` |
| Validating before trusting output | done | `eval/run_eval.py`, shadow mode, trust ledger |
| Handling data gaps | done | censored-MLE soft sensors |
| **Low-cost sensing proposals** | done | `02-architecture.md §9`, costed per station type |
| UX: three distinct views from one model | done | `dashboard.py` |
| Integration: legacy PLCs, OT, live-production constraints | done | `02-architecture.md §6` |
| Scalability & ROI | done | Plant Beta port; scenario-based business case |

---

## Reference parameters

| Brief | Loom |
|---|---|
| 30–50 stations across body, paint, final | 42 across three zones (Plant Beta: 28) |
| Majority instrumented, meaningful minority manual | 69% / 31% (Plant Beta: 57% / 43%) |
| Instrumentation changes only in scheduled windows | retrofit planner sized to a shutdown budget |

---

## Deliverables

| Required | Status |
|---|---|
| Business proposal: problem framing | `01-business-proposal.md §1` |
| — solution design | §2 |
| — target users | §3 |
| — business case and impact | §4, three scenarios with every assumption shown |
| — phased roadmap | §5, four phases with exit criteria |
| — key risks with mitigations | §6, eight risks with residual risk stated |
| Working prototype on illustrative data | `run_demo.py`, ~40 s, two lines |
| Public GitHub repository | ready to push; see README |
| Demo video | script and shot list in `04-demo-video-script.md` — **must be recorded by the team** |
| README | `README.md` |

---

## Gaps closed after external review

An independent review of an earlier version raised five criticisms. Four were
addressed in code and are now measured; the fifth is honestly unfixable here.

| Criticism | Response | Evidence |
|---|---|---|
| "Serial-line assumption breaks on real lines" | Added parallel machines and a rework loop; rebuilt reconstruction to recover order from scans | start MAE 117 s → 0.27 s |
| "Sensor ROI can't work without ground truth" | Built a deployable EVI ranking and tested transfer | throughput rho 1.00, quality rho −0.06 → two-stage method |
| "Scan timestamps won't be reliable" | Data-quality layer; read-rate and clock-sync sweeps | error 1.67% → 0.22% after repair |
| "Root cause ≠ causation" | UI relabelled; engineer confirm/reject writes to the ledger | `decision.adjudicate` |
| "Everything is simulated" | **Not fixable without plant access.** Phase 0 exists for this | — |

We also added an **adversarial fault set** — step, oscillating and bursty
shapes the detectors were never tuned against — because scoring only against
faults you invented is how you build a system that works on imagined failures.
All three were caught by SPC. Negative-control probes, where no cause was
injected, correctly return no hypothesis.

## Known gaps, stated plainly

0. **Overtaking, skipped operations and re-entrant flow** are still unsupported.
   Rework loops and parallel machines now work; a unit revisiting the same
   station twice does not.
1. **No physics-informed station models.** Cycle times are statistical, not
   derived from tool dynamics. A torque model or a thermal model of the paint
   oven would be a genuine addition and is not here.
2. **No 3D visualisation.** Kattenstroth et al. list it as a requirement for
   shared understanding between planner and simulation developer. Out of scope
   for a prototype; named rather than dropped.
3. **No supplier-side integration.** Also on Kattenstroth's list. Loom uses lot
   IDs but does not reach into supplier systems.
4. **Static bottleneck baseline sometimes wins on top-1.** With the operator
   fault added, one station dominates for most of the run, so a long-run
   baseline is right more often than the forecast on a raw top-1 count. The
   forecast still leads on the regime change. Both numbers are reported.
5. **Conformal coverage for the weakest defect type falls to ~47%** in some
   seeds against a 90% target. Rolling recalibration recovers the average to
   85%, but per-type coverage under drift is not solved.
6. **Scanner corruption rates are parameterised, not measured.** Rather than
   quote a read rate from someone else's plant, the repo reports accuracy
   across a sweep so a plant can look up its own.
7. **Everything is simulated.** The harness proves the method works on data
   whose truth we control. Phase 0 of the roadmap exists to re-measure every
   claim on real data before anything goes live.
