# Loom

**A digital twin for vehicle assembly lines that stays useful where the sensors run out.**

Accenture Innovation Challenge 2026 — Round 2, Problem Track 4: `DigitalTwin.ai`

[![validation-harness](https://github.com/kabhinav24/loom-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/loom-twin/actions/workflows/ci.yml)

> **Demo video:** _add link here before pushing_
> Replace `OWNER` in the badge URL above with your GitHub username or org.

---

## 1. The problem in one paragraph

Every assembly-line digital twin in the literature assumes a fully instrumented
factory. Real lines are a patchwork: on our reference line **13 of 42 stations
have no process sensors at all**, and instrumentation can only be retrofitted
during a shutdown that happens twice a year. That creates two expensive blind
spots. A station slowly degrades and nobody notices until the line is running
overtime. A defect is created in the body shop and found 30 stations later at
final inspection, by which time hundreds of vehicles carry it and some have
shipped. Loom closes both — through the sensor gap, not around it.

## 2. Results

Every figure is scored against ground truth the twin never sees, averaged over
3 seeds, on **deliberately dirty scan data** (11,898 missed reads, 3,093
duplicates, 5,047 late manual scans injected and then repaired).

| Capability | Result | Scored against |
|---|---|---|
| Recover machine state from scans alone | start-time MAE **0.27 s**, blocking recall **99.9%** | hidden simulator truth |
| Cycle time at 13 un-instrumented stations | median error **0.22%**, 95% CI coverage **92%** | hidden true cycle times |
| Root cause of a confirmed defect | **4 of 4** hidden faults at rank 1, every seed | injected fault log |
| SPC on drift, degradation, micro-stops | **89%** recall, onset back-dated to **±1.4 days** | injected fault log |
| Bottleneck regime change | called **~4 days** before present-state detection | actual future bottleneck |
| Calibrated defect risk | empirical coverage **85%** vs a 90% target under drift | held-out findings |
| Repair of a dirty scan feed | median soft-sensor error **1.67% → 0.22%** | clean-scan baseline |
| Non-serial flow (rework + parallel machines) | 7.6% of units re-enter **out of sequence**; start MAE holds at **0.27 s** | hidden simulator truth |
| Fault shapes never designed for | step, oscillating and bursty faults — **3 of 3 caught** by SPC | adversarial fault set |
| Port to a different plant | 28 stations, 57% coverage — **config only, zero code** | second full pipeline run |

`eval/run_eval.py` runs the whole thing across seeds and **exits non-zero** if
any of ten headline metrics leaves a pre-stated envelope, so it doubles as CI.

Requirement-by-requirement coverage of the brief, **including what is not
done**, is in [`docs/05-requirements-coverage.md`](docs/05-requirements-coverage.md).

---

### 2.1 Robustness, measured rather than claimed

Two prerequisites are quantified instead of asserted, so a plant can look up its
own numbers rather than trusting ours.

**Missed scan reads** — accuracy degrades gracefully; even losing one read in
eight, the point estimate stays under 1%.

| Missed-read rate | 0% | 1% | 3% | 6% | 12% |
|---|---|---|---|---|---|
| Soft-sensor median error | 0.16% | 0.19% | 0.24% | 0.34% | 0.63% |
| 95% CI coverage | 92% | 92% | 85% | 77% | 62% |

**Clock synchronisation** — this is a hard prerequisite and the curve shows why.
Past about one second of reader drift the intervals stop being trustworthy.

| Reader clock drift (sd) | 0.03 s (NTP) | 0.25 s | 1 s | 3 s |
|---|---|---|---|---|
| Soft-sensor median error | 0.04% | 0.30% | 0.54% | 4.17% |
| 95% CI coverage | 92% | 69% | 39% | 8% |

Reproduce both with `python run_demo.py --robustness`.

## 3. Implementation approach

### 3.1 The core idea

At a station with no sensors we still know when a vehicle **arrived** and when
it **left**. That gap is `processing + time stuck waiting`. Separating them
looks impossible until you notice that being blocked leaves a fingerprint: a
blocked vehicle departs at the *exact instant* a slot frees downstream. So every
observation falls into one of two buckets:

```
dwell > threshold   ->  never blocked  ->  processing time observed EXACTLY
dwell = threshold   ->  was blocked    ->  processing time is an UPPER BOUND
```

The threshold is itself observable. That makes cycle-time estimation a
**censored-likelihood problem with observable, unit-specific thresholds**:

```
L(mu, sigma) = prod_uncensored f(p_i) * prod_censored F(theta_i)
```

Maximising it uses the blocked vehicles instead of discarding them — which
matters most at exactly the stations that block most, the ones sitting behind a
bottleneck. Result: **0.22% median error**, with an interval and a confidence
that widens honestly as censoring rises.

Once the blind stations are visible, everything else runs through them: control
charts, bottleneck forecasting, defect features, and root-cause attribution.

### 3.2 Two independent explanation paths, on purpose

| | Learned model | Statistical engine |
|---|---|---|
| Needs training data | yes | **no** |
| Good at | recurring patterns | events never seen before |
| Output | flag / abstain / pass, with coverage | ranked hypotheses with lift and q-value |

The paint-humidity excursion proves why both are needed: the learned model was
trained *before* that humidity existed and largely missed it; the statistical
engine isolated it on the first batch that contained it — and correctly blamed
**one air handler** rather than four paint robots.

### 3.3 Design decisions

- **Read-only at the OT boundary.** The ISO 23247 Device Control sub-entity is
  deliberately not implemented. Loom recommends; a human actuates.
- **The line is configuration, not code.** A station-type library plus a station
  list. Porting to Plant Beta is a config block.
- **Attention is a hard budget.** Five alerts per shift, ranked, each showing
  how often that alert type has been right on this line.
- **Hypotheses are leads, not verdicts.** A quality engineer confirms or rejects
  each one and that judgement is what the trust ledger scores.

---

## 4. Solution architecture

```
 PHYSICAL WORLD  (simulated; ground truth withheld from every component below)
 42 stations - body / paint / final - 69% instrumented - 6 injected faults
                              |
                    departure scans + signals from
                    instrumented stations + gate findings
                              v
 +--------------------------------------------------------------------+
 | 0. DATA QUALITY          dataquality.py                            |
 |    detect + repair missed / duplicate / late scans; flag clock     |
 |    drift; emit a per-station feed-quality score                    |
 +--------------------------------------------------------------------+
 | 1. TWIN CORE             twin.py                                   |
 |    state reconstruction   working / blocked / starved from scans   |
 |    soft sensors           censored-MLE cycle time at dark stations |
 |    knowledge graph        unit -> visit -> station, lot, operator  |
 |    rolling-horizon DES    forward simulation, drift extrapolated   |
 +--------------------------------------------------------------------+
 | 2. ANALYTICS                                                       |
 |    spc.py         EWMA / CUSUM / stoppage-rate, incl. on soft data |
 |    bottleneck.py  Roser active-period method on simulated future   |
 |    defect.py      per-type risk, class-conditional conformal       |
 |    backtrace.py   contingency tests over the thread, FDR-controlled|
 |    sensor_roi.py  retrofit ranking: ablation + deployable EVI      |
 +--------------------------------------------------------------------+
 | 3. DECISION              decision.py                               |
 |    alert budget - trust ledger - human adjudication - shadow mode  |
 +--------------------------------------------------------------------+
 | 4. VIEWS                 dashboard.py                              |
 |    floor supervisor - plant manager - leadership - validation      |
 +--------------------------------------------------------------------+
                              v
              eval/run_eval.py grades every claim against
                    the ground truth withheld at the top
```

Full ISO 23247 entity mapping, the line recursion, data contract, integration
approach and the low-cost sensing menu are in
[`docs/02-architecture.md`](docs/02-architecture.md).

### Module reference

| File | Responsibility |
|---|---|
| `src/loom/config.py` | Station-type library, two line configs, cost model |
| `src/loom/sim.py` | Ground-truth plant + fault injection — **not** the twin |
| `src/loom/dataquality.py` | Scanner fault injection, detection and repair |
| `src/loom/twin.py` | State reconstruction, soft sensors, graph, rolling DES |
| `src/loom/spc.py` | EWMA, CUSUM, stoppage-rate charts |
| `src/loom/bottleneck.py` | Active-period detection and forecasting |
| `src/loom/defect.py` | Per-type conformal defect risk, causally truncated |
| `src/loom/backtrace.py` | Genealogy attribution with FDR control |
| `src/loom/sensor_roi.py` | Retrofit ranking (ablation + deployable EVI) |
| `src/loom/decision.py` | Alert budget, trust ledger, human-in-the-loop |
| `src/loom/dashboard.py` | Self-contained HTML, four views |
| `run_demo.py` | End-to-end pipeline |
| `eval/run_eval.py` | Multi-seed scoring harness / regression test |

---

## 5. Dependencies

Python **3.11+**. Five packages, all pure-CPU. No GPU, no database, no cloud
service, no API key.

```
numpy>=1.26
pandas>=2.0
scipy>=1.11
scikit-learn>=1.4
networkx>=3.0
```

The discrete-event simulation kernel is written from scratch — a serial line
with finite buffers has an exact recursion — so there is no SimPy dependency.
The dashboard is generated HTML with no CDN calls, so it opens offline from a
`file://` URL on a plant laptop.

---

## 6. Execution instructions

```bash
git clone https://github.com/<your-org>/loom-twin.git
cd loom-twin

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the demo

```bash
python run_demo.py                 # Plant Alpha, dirty scans, ~60 s
```

Writes to `outputs/`:

- `dashboard_alpha.html` — open in any browser, no server needed
- `results_alpha.json` — every number the run produced
- `trust_ledger_alpha.json` — every alert and its adjudicated outcome

### All options

| Command | What it does |
|---|---|
| `python run_demo.py` | Plant Alpha (42 stations, 69% instrumented) |
| `python run_demo.py --line beta` | Plant Beta (28 stations, 57%) — **config only** |
| `python run_demo.py --clean-scans` | Skip the data-quality layer, ideal scans |
| `python run_demo.py --fast` | Skip the retrofit ablation, ~25 s |
| `python run_demo.py --faults adversarial` | Fault shapes the detectors were **not** designed for, plus negative controls |
| `python run_demo.py --robustness` | Sweep accuracy vs missed-read rate and clock sync |
| `python run_demo.py --days 30 --seed 42` | Different run length and seed |
| `python eval/run_eval.py --seeds 3` | Multi-seed harness; **exit 1 on regression** |

### Expected output

```
[3b/9] statistical process control
  [OK ] F1 PARAM_DRIFT    S12        lag +1.5d  onset est 8.0
  [OK ] F2 CYCLE_DEGRADE  S28        lag +11.5d onset est 7.5
  [OK ] F5 MICROSTOP      S07        lag +2.0d  onset est 2.5

[6/9] genealogy back-trace against the injected faults
  [OK ] LOOSE_FASTENER    -> S12:torque_nm low             lift  5.8  conf 0.98
  [OK ] PAINT_FINISH      -> PAINT:booth_humidity_pct high lift 26.3  conf 0.99
  [OK ] ELECTRICAL_FAULT  -> S33:continuity_ohm high       lift 19.0  conf 1.00
  [OK ] FIT_MISALIGN      -> op:FINAL-OP2                  lift 17.7  conf 1.00
```

Runtime ~60 s on one CPU core. Everything is seeded and reproducible.

---

## 7. Key features

1. **Soft sensing at un-instrumented stations** via censored-likelihood
   estimation — the technical core.
2. **Data-quality layer** that injects and repairs real scanner faults, and
   downgrades downstream confidence by feed quality rather than hiding it.
3. **Training-free root-cause attribution** with FDR control, plus zone and
   time-window consolidation so a utility fault isn't blamed on four robots.
4. **Containment** — names the 39 vehicles still on the line sharing an exposure.
5. **Conformal defect risk** that abstains instead of guessing.
6. **Two-stage sensor investment planning** (see §8).
7. **Trust infrastructure** — alert budget, published hit rate, human
   adjudication, eight-week shadow mode.
8. **Config-only portability**, demonstrated on a second plant.
9. **Non-serial flow support** — rework loops and parallel machines, with the
   cost of getting it wrong measured explicitly.
10. **Adversarial fault set and negative controls** — three fault shapes the
    detectors were never tuned against, plus probes where no cause exists to
    check the system does not invent one.

---

## 8. A negative result we kept

The retrofit ablation — reveal a station's real signals, refit, measure the
PR-AUC uplift — needs ground truth and is therefore **prototype-only**. A real
plant cannot run it, because the sensor being considered does not exist.

So we built a deployable expected-value-of-information ranking from observable
quantities and tested whether it reproduces the ablation. Decomposed:

| Component | Spearman rho vs ablation | Transfers? |
|---|---|---|
| Throughput value | **1.00** | yes, perfectly |
| Quality value | **-0.06** | **no, not at all** |

Whether a station constrains throughput is visible in timing data you already
own. Whether its process drives defects is not knowable until you measure it.

That dictates the deployment method, and it is more useful than the answer we
were hoping for: **rank by EVI to shortlist, run a temporary instrumented pilot
on the top 2–3 stations during one shutdown, measure the quality half, then
commit the rest of the budget.** Both rankings independently pick S28 first.

---

## 9. Scope and limitations

**Scope.** Loom targets **assembly flows with unit-level traceability** —
automotive general assembly, appliance and electronics lines.

The reference line now includes **parallel machines** (redundant weld robots at
S04 and S10) and a **rework loop** (S14 diverts 7.5% of units to an offline
cell that returns them out of sequence). Both were added specifically because
an earlier version assumed strict serial order, and that assumption is worth
showing the cost of:

| | Order-assuming reconstruction | Sequence-aware |
|---|---|---|
| Start-time MAE | **117 s** | **0.27 s** |
| Blocking recall | 78.9% | 99.9% |
| Soft-sensor median error | 1.44% | 0.014% |
| 95% CI coverage | 38% | 92% |

The fix is to recover processing order per station by sorting departure scans,
rather than assuming the unit index is the sequence position. One requirement
falls out of it: **a rework cell needs its own read point.** Without a re-entry
timestamp, a unit returning to an idle station is indistinguishable from a slow
station. Plants that track WIP already have this scanner.

Still out of scope: overtaking within a station queue, units skipping
operations entirely, and re-entrant flows that revisit the same station.

**Clock synchronisation is a prerequisite, not a feature.** Per-reader offsets
are **not statistically identifiable** from scans alone — the gap between two
scans confounds offset with processing time. We built an estimator that tried,
measured it as worse than doing nothing, and deleted it. Loom requires NTP to
~50 ms (routine on plant networks, negligible against a 60 s takt) and
*diagnoses* drift rather than correcting it.

**Other limitations, stated plainly:**

- **Defect discrimination is modest** for most types (prequential ROC-AUC 0.48
  to 0.70). About 40% of defects in the simulator are irreducible randomness
  with no observable driver, so a ceiling exists. The conformal layer is what
  makes a weak model safe to deploy: it abstains, and the abstention rate is
  reported.
- **Conformal coverage degrades under drift** — 85% mean against a 90% target,
  weakest type ~47% in some seeds. Coverage is guaranteed under exchangeability,
  which a drifting line violates, so we *measure* empirical coverage
  continuously rather than asserting the guarantee still holds.
- **Static bottleneck baseline sometimes wins on top-1** (0.63 vs 0.57). One
  station dominates late in the run. The forecast still leads on the regime
  change. Both numbers are in `eval_summary.json`.
- **SPC false alarms are bounded, not solved** — 32% upper bound. A textbook
  3-sigma chart across ~270 charts fires on more than half of all clean
  stations; that number is in the repo too.
- **Association is not causation.** The back-trace ranks hypotheses. The UI says
  "likely contributing factors — not confirmed root causes" and requires an
  engineer's verdict, which is what the ledger scores.
- **Everything is simulated.** The harness proves the method works on data whose
  truth we control. It does not prove transfer. Phase 0 of the roadmap exists to
  re-measure every claim on real data before anything goes live.

---

## 10. Documentation

| Document | Contents |
|---|---|
| [`docs/01-business-proposal.md`](docs/01-business-proposal.md) | Problem framing, solution design, users, business case, roadmap, risks |
| [`docs/02-architecture.md`](docs/02-architecture.md) | ISO 23247 mapping, line recursion, data contract, integration, low-cost sensing menu |
| [`docs/03-literature-and-gaps.md`](docs/03-literature-and-gaps.md) | What the source papers established and where we go past them |
| [`docs/04-demo-video-script.md`](docs/04-demo-video-script.md) | Shot list for the prototype demonstration video |
| [`docs/05-requirements-coverage.md`](docs/05-requirements-coverage.md) | Every brief requirement mapped, gaps included |

---

## 11. Reproducibility

Everything is seeded. `run_demo.py --seed N` and `eval/run_eval.py --seeds K`
produce deterministic output. The harness exits non-zero if any headline metric
leaves its envelope, so it can be wired straight into CI.

## Licence

MIT — see [`LICENSE`](LICENSE).
