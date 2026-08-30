# Strategic video script (2:30)

This is the **Solution Framework** video required alongside the slides —
problem, approach, impact. It is *not* the prototype demo; that is a separate
recording with its own script in `04-demo-video-script.md`.

Face-to-camera or voice over slides. No code, no terminal. ~340 spoken words.

File naming: `TeamName_CampusName.mp4`.

---

**0:00–0:35 — The problem**

> "A car assembly line has about forty stations. On a real line, a third of them
> have no sensors at all — they're old, they're manual, they were never
> instrumented, and you can only retrofit them during a shutdown that happens
> twice a year.
>
> That creates two expensive blind spots. First, a station slowly gets slower and
> nobody notices until the line is running overtime. Second, a defect gets
> created early — a torque tool drifting out of spec in the body shop — and isn't
> found until final inspection, thirty stations and forty minutes later. By then
> hundreds of cars carry it and some have already left on trucks."

**0:35–1:00 — Why this hasn't been solved**

> "We read the literature on digital twins for assembly lines. The best work on
> bottleneck prediction states plainly that it relies on acquiring data from
> every single workstation. The best work on knowledge-graph anomaly detection
> admits its model can't explain its own predictions.
>
> So the twins that exist work beautifully on the ideal factory, and go blind on
> the real one."

**1:00–1:45 — Our approach**

> "Our insight is small and it unlocks everything else.
>
> Even at a station with no sensors, you still know when a car arrived and when
> it left. That gap is work time plus time stuck waiting. And being stuck leaves
> a fingerprint — a blocked car leaves at the exact moment a space frees up
> downstream. So every car gives you either an exact measurement or a known upper
> limit. Combine both statistically and you recover the true cycle time to within
> nought point nought three percent.
>
> Once the blind stations are visible, everything runs through them: control
> charts, bottleneck forecasting, defect risk, and root-cause tracing.
>
> And we deliberately built two independent ways to explain a problem. A learned
> model for patterns it has seen before. A statistics engine that needs no
> training at all, for things that have never happened. When a humidity spike hit
> the paint shop, the learned model missed it — it was trained before that
> humidity existed. The statistics engine caught it immediately, and correctly
> blamed one air handler instead of four paint robots."

**1:45–2:15 — Impact**

> "On one line: seven and a half to twenty crore rupees a year, from recovered
> throughput, defects caught in-plant instead of under warranty, and
> investigations that take seconds instead of three days. Payback inside the
> first year on our conservative case.
>
> And it ports. We ran the identical code on a second, smaller, sparser plant —
> twenty-eight stations, fifty-seven percent coverage. Config file only. Zero
> lines of code changed."

**2:15–2:30 — Close on credibility**

> "Everything we've claimed is scored against ground truth our system never sees.
> We inject five faults into a simulated plant, hide them, and grade ourselves.
> Our test suite fails the build if any headline number drifts. And our README has
> a limitations section that names the three places our numbers are weak.
>
> We'd rather you trust the ones that are strong."

---

## Delivery notes

- 2:30 leaves 30 seconds of headroom against the 3:00 cap. Don't fill it.
- The strongest 15 seconds are the last ones. Do not rush the limitations line —
  volunteering your own weak spots is what makes the rest believable.
- Say "nought point nought three percent" slowly, or put it on screen instead.
- Do not show code. This video is for someone deciding whether the idea is worth
  funding, not whether the implementation is correct.
