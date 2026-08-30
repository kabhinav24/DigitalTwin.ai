# Architecture

## 1. ISO 23247 mapping

ISO 23247 defines a manufacturing digital twin as a *fit-for-purpose digital
representation of an Observable Manufacturing Element with synchronisation
between the element and its representation*, across three domains. Shao (2024)
reports that of 29 architectures analysed against the standard, none
implemented Plug-and-Play Support, Peer Interface or Data Assurance, and only
one implemented Access Control. Loom maps explicitly, including where it
deliberately declines to implement.

| ISO 23247 entity | Loom component | Status |
|---|---|---|
| **Data Collection sub-entity** | | |
| Data Collecting FE | read-only OPC UA / MQTT tap, scan readers | implemented |
| Data Pre-Processing FE | `twin.reconstruct_states` | implemented |
| Collection Identification FE | station-type library in `config.py` | implemented |
| **Device Control sub-entity** | | |
| Controlling / Actuation FE | — | **deliberately not implemented** |
| **Digital Twin entity** | | |
| Digital Representation FE | `twin.KnowledgeGraph` | implemented |
| Synchronization FE | rolling-horizon sync in `twin.RollingHorizonTwin` | implemented |
| Simulation FE | DES kernel generated from config | implemented |
| Analytic Service FE | `bottleneck`, `defect`, `backtrace` | implemented |
| Reporting FE | `dashboard`, `decision.TrustLedger` | implemented |
| **Resource Access sub-entity** | | |
| Access Control FE | role scoping on persona views | partial |
| Interoperability Support FE | config export, JSON payload | partial |
| Peer Interface FE | — | not implemented |
| **Cross-system entity** | | |
| Data Assurance FE | `SoftSensor` confidence, conformal coverage, trust ledger, FDR control | **implemented** |
| Security Support FE | read-only boundary, no credentials held | partial |
| **User entity** | | |
| User Interface FE | three persona views | implemented |

**On Device Control.** Not implementing it is a design decision, not a gap.
Writing to a PLC is the capability plants refuse first and it is unnecessary
for every claim Loom makes. Ragazzini et al. close their loop by adapting order
release and dispatching; that recommendation is surfaced to a human here rather
than actuated. Closing it automatically is a Phase 3 question that needs a
safety case, not a prototype feature.

**On Data Assurance.** This is the entity the literature says nobody builds,
and it is the one this brief most needs: soft-sensor confidence, conformal
coverage measured against target, FDR-controlled attribution, and a trust
ledger that publishes the twin's own hit rate.

## 1b. Scope of applicability

Loom targets **serial and near-serial assembly flows with unit-level
traceability**: automotive general assembly, appliance and electronics lines.
The recursion in the next section assumes a serial line with finite buffers and
blocking-after-service.

**Rework loops and parallel machines are now supported.** The reference line
runs redundant weld robots at S04 and S10, and S14 diverts 7.5% of units to an
offline rework cell that returns them out of sequence. Processing order is
recovered per station by sorting departure scans rather than assuming the unit
index is the sequence position — assuming it costs 117 s of start-time error.
A rework cell must have its own read point; without a re-entry timestamp a unit
returning to an idle station is indistinguishable from a slow station.

Still **not** supported: overtaking within a station queue, units skipping
operations entirely, and re-entrant flows that revisit the same station. Those
need the reconstruction extended from a chain to a general routing graph. The
censoring argument still holds on any DAG where each unit's path is known, so
this is tractable, but it is future work rather than a solved problem.

The portability claim is likewise scoped: **no core modelling code changes for
compatible serial or near-serial assembly lines.** A semiconductor fab or a
continuous chemical process is a different problem.

## 2. The exact line recursion

A serial line with finite buffers under blocking-after-service admits a closed
recursion, so no general event queue is required:

```
start[u][j]  = max(depart[u][j-1] + transport[j-1], depart[u-1][j])
finish[u][j] = start[u][j] + p[u][j]
depart[u][j] = max(finish[u][j], start[u-B_j][j+1])
```

This yields the three machine states directly:

```
starved = [depart[u-1][j], start[u][j])      inactive
working = [start[u][j],    finish[u][j])     ACTIVE
blocked = [finish[u][j],   depart[u][j])     inactive
```

Roser's active-period method needs nothing else. That is why the method
survives at stations with no process sensors.

## 3. Soft sensing as censored estimation

At a dark station the twin observes `dwell = depart - start` and the censoring
threshold `theta = gate_open - start`, both derived from scans. Then:

```
dwell > theta   never blocked   ->  p = dwell exactly
dwell = theta   blocked         ->  p < theta   (upper bound)
```

Both branches are informative. Maximum likelihood over the mixture:

```
L(mu, s) = prod_{uncensored} f(p_u; mu, s) * prod_{censored} F(theta_u; mu, s)
```

Because the thresholds are *observable and unit-specific*, this is a proper
likelihood requiring no truncation correction. The estimator returns a mean, a
95% interval from a numerical Hessian, and a confidence that falls with the
censoring fraction.

Measured on 13 dark stations: median error 0.011%, worst 0.83%, 95% CI coverage
94.9%. The worst case is S25, which sits directly behind the bottleneck and is
blocked most of the time — its confidence correctly drops to ~0.7 while its
estimate stays inside 1%.

## 4. Why the knowledge graph is an index

Three weeks of production is ~800,000 station visits. Materialising each as a
graph node is the textbook illustration and the wrong engineering choice: every
query that matters is *"which units were exposed to X during window W"*, which
is an interval-index problem, not a traversal problem.

So the **schema** is a NetworkX graph — small, inspectable, exportable to
OWL/RDF in the style of Wang et al. — and the **instance data** is columnar.
Neighbourhood subgraphs are materialised on demand for the explain panel.
`KnowledgeGraph.exposed_units` is a vectorised interval query; on this dataset
it returns in under a millisecond.

Schema classes: `Unit`, `StationVisit`, `Station`, `StationType`, `Zone`,
`SupplierLot`, `Operator`, `Shift`, `Inspection`, `DefectFinding`,
`ProcessSignal`.

## 5. Data contract

What the twin is allowed to read (`sim.observable_view`):

| Field | Source | Available at dark stations |
|---|---|---|
| `scan_out_s[u][j]` | RFID / barcode at station exit | **yes** |
| `release_s[u]` | body-shop release schedule | yes |
| process signals | station PLC via OPC UA | no |
| inspection findings | quality gate systems | yes |
| `lot_id`, `operator`, `shift`, `variant` | MES / ERP | yes |

Everything else — true cycle times, latent defect origin, the fault log — lives
in `PlantData.truth` and is touched only by scoring functions and the offline
retrofit planner, both marked in source.

## 6. Integration without disrupting production

- **Clock synchronisation is a prerequisite.** NTP to ~50 ms across read
  points. Per-reader offsets cannot be recovered statistically (see
  `dataquality.py`), so this is a commissioning requirement, not an algorithm.
- **Read path only.** OPC UA subscriptions and MQTT topics. No writes, no
  credentials held for control systems.
- **Edge gateway.** Store-and-forward across network drops; buffered replay on
  reconnect. Nothing on the line depends on the gateway being up.
- **Legacy stations.** Where a PLC is too old to expose OPC UA, the only
  requirement is a scan point at the station boundary — which most automotive
  lines already have for VIN tracking. That is the minimum viable integration
  and it is what makes the soft-sensor layer deployable.
- **Retrofits during scheduled windows only.** The ablation ranking exists to
  make those windows count.

## 7. Runtime

Whole pipeline, one CPU core, no GPU:

| Stage | Wall time |
|---|---|
| Simulate 19,200 units × 42 stations | ~1.1 s |
| State reconstruction | ~0.2 s |
| Soft sensors, 13 stations, censored MLE | ~0.1 s |
| Bottleneck detection, 10 blocks | ~1.6 s |
| Rolling forecast, 33 triggers × 5 replications | ~24 s |
| 6 conformal defect models | ~2.3 s |
| 3 root-cause traces (~200 hypotheses each) | ~0.13 s |
| Retrofit ablation, 13 stations | ~8 s |

In deployment the forecast runs once per trigger (~700 ms), not 33 times, and
attribution runs on a confirmed finding (~45 ms per trace). Both fit
comfortably inside a shift-level decision loop.

## 8. Scaling to another line or plant

A line is a list of station instances pointing at reusable station types.
Onboarding a new line means writing a config: station order, types, buffer
sizes, transport times, which stations are instrumented, where the gates are.
No modelling code changes, which is the direct answer to the "high initial
creation effort" and "hardcoded models need manual adaption" challenges in
Kattenstroth et al.

The station-type library is the cross-plant asset. A `torque_station` behaves
the same in Pune and in Chennai; only its instance parameters differ.


## 9. Low-cost sensing menu

The brief asks what sensing we would propose at partially instrumented
stations. The retrofit planner ranks *which* stations to do; this is *what* to
fit, costed per station type and chosen so every option can be installed inside
a single scheduled shutdown without touching line control.

| Station type | Proposed retrofit | Approx. cost | What it unlocks |
|---|---|---|---|
| `manual_fit` | RFID/barcode read point at station entry and exit, plus a light curtain for presence | ₹0.95 L | Turns an inferred dwell into a measured cycle time; removes the blocking ambiguity entirely |
| `torque_station` | Networked torque transducer on the existing tool, or a smart-tool gateway | ₹1.85 L | Direct torque and angle per fastener; catches F1-class drift within a shift |
| `electrical_fit` | Inline continuity and insertion-force check on the existing fixture | ₹2.05 L | Catches contaminated-lot faults at the station rather than at final inspection |
| `sealer` | Flow meter plus nozzle pressure transducer | ₹2.40 L | Bead-quality proxy without a vision system |
| `weld_robot` | Clamp-on current transformer on the weld transformer; MTConnect or Modbus adapter | ₹3.10 L | Weld current per spot from a legacy controller with no PLC changes |
| `paint_booth` | Booth-level temperature and humidity logger on the air handler | ₹4.20 L | Zone environmental monitoring; would have caught F4 as a utility alarm |
| `oven` | Multi-zone thermocouple string | ₹2.65 L | Cure-profile conformance |

Three principles behind the list:

**Scan points before sensors.** The cheapest item on the list is also the
highest-value one for a manual station, because a read point converts a soft
estimate into a measurement and costs a fifth of a process sensor. On most
automotive lines the VIN readers already exist for tracking; the retrofit is
often a network configuration rather than hardware.

**Clamp-on before inline.** A current transformer around an existing conductor
and a logger on an existing air handler require no process interruption and no
controller change. Anything requiring a PLC program change waits for a
shutdown; anything requiring a mechanical modification waits longer.

**Utility-level before station-level.** One booth-level humidity logger covers
four paint booths. Fault F4 is the case in point: a single ₹4.2 L logger on the
air handler would have raised a utility alarm before any vehicle was painted
badly, which is cheaper and earlier than instrumenting each booth.
