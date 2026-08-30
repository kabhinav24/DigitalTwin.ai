# Round 2 slide content (2–3 slides)

The challenge supplies a prescribed template via the Brief button. This file is
the **content**, written to be pasted into whatever that template's boxes are.
Do not add slides. Three is the ceiling and two is safer.

File naming: `TeamName_CampusName.pdf` (or `.pptx`).

---

## Slide 1 — The problem, and the thing everyone else assumes away

**Headline**
> Every assembly-line twin assumes a fully instrumented factory. Real lines
> aren't. Ours has 13 stations out of 42 with no sensors at all — and that's
> where the bottleneck goes.

**Three bullets, maximum**

- A station drifts 0.75% slower per day. Nothing alarms. Six weeks later the
  line runs 40 minutes of overtime a shift and nobody knows why.
- A torque tool wanders out of spec in the body shop. The defect is found 30
  stations and 40 minutes later at final inspection — by which time hundreds of
  cars carry it and some have already shipped.
- The published research has the same blind spot. Ragazzini et al. (2024)
  predict bottlenecks well, but their framework *"relies on the possibility to
  acquire data from each workstation."* That assumption is the gap.

**Visual:** the line strip from the dashboard — 42 station blocks, 13 shaded
dark, with an arrow marking where the bottleneck moves on day 12 (into the dark
region).

---

## Slide 2 — What we built, and why it works where sensors don't

**Headline**
> Loom rebuilds the invisible stations from barcode scans, then predicts and
> root-causes straight through the gap.

**The one technical idea, stated plainly**

At a station with no sensors we still see when a car arrived and when it left.
That gap is *work time + time stuck waiting*. Being stuck leaves a fingerprint —
a blocked car leaves at the exact instant a space frees downstream. So every
observation is either an exact measurement or a known upper limit. Fit both
together and you recover the true cycle time to **0.03% error**.

Once you have that, everything else keeps running through the blind spot.

**Four results, each scored against hidden ground truth**

| | |
|---|---|
| Cycle time at 13 blind stations | 0.03% median error, 92% interval coverage |
| Root cause of confirmed defects | **4 of 4** hidden faults found at rank 1, every seed |
| Bottleneck regime change | called ~4 days before present-state detection |
| Port to a second, sparser plant | 28 stations, 57% coverage — **config only, zero code** |

**The design argument (say this out loud)**

Two independent explanation paths, on purpose. The learned model is good at
patterns it has seen; the statistics engine needs no training at all. When a
humidity excursion hit the paint shop, the model — trained before that humidity
existed — largely missed it. The statistics engine caught it on the first batch,
and correctly blamed **one air handler** rather than four paint robots.

**Visual:** the back-trace table — defect, rank-1 cause, lift, and "matched" tag.

---

## Slide 3 — Impact, and what it costs to find out

**Headline**
> ₹7.6–20.6 crore a year on one line. Payback inside year one on the
> conservative case.

**Value, three pools**

| Scenario | Throughput | Quality | Investigation | Total / yr |
|---|---|---|---|---|
| Conservative | ₹6.08 cr | ₹1.43 cr | ₹0.06 cr | **₹7.59 cr** |
| Base | ₹10.43 cr | ₹4.77 cr | ₹0.06 cr | **₹15.26 cr** |
| Optimistic | ₹13.90 cr | ₹6.68 cr | ₹0.06 cr | **₹20.61 cr** |

Year-1 cost ₹70.2 L, steady state ₹23 L. We lead with the conservative number
deliberately.

**Roadmap, four phases**

Shadow (8 wks, predicts but tells nobody, builds a track record) → Assist (alerts
go live under a 5-per-shift budget) → Plant-wide → Multi-site.

**Risk we take most seriously**

False alarms erode floor trust faster than true positives build it. So: a hard
alert budget, a published hit rate shown next to every alert, and eight weeks of
shadow mode before anything reaches a supervisor. We also measured the trap — a
textbook 3-sigma control chart across 270 charts fires on **more than half** of
all clean stations. That number is in our repo.

**Visual:** the scenario table, or the trust-ledger panel showing running
precision per alert type.

---

## What to cut if the template only allows two slides

Merge 1 and 2. Keep: the 13-blind-stations framing, the scan-reconstruction
idea, the 4-of-4 result, and the money range. Drop the literature citation, the
roadmap detail, and the false-alarm number — those live in the proposal document
and the repo.
