# Recording run-sheet

The narration is in `04-demo-video-script.md`. This is the **operational** sheet:
exact commands in order, so the whole thing can be captured in one take.

## Before you hit record

```bash
python run_demo.py --fast >/dev/null 2>&1     # warm the imports
rm -rf outputs/*.json outputs/*.html          # so the run looks live
clear
```

- Terminal ~16pt, dark theme, window ~1600x900
- Close Slack, mail, notifications
- Open `outputs/dashboard_alpha.html` in a second tab **beforehand**, then close
  it — the browser will remember the render and reopen instantly on camera

## Take 1 — terminal (about 90 seconds)

```bash
python run_demo.py
```

Talk over it. The four moments to point at as they scroll past:

| Appears | Say |
|---|---|
| `scans: 11,898 missed, 3,093 duplicate...` | "this run is on deliberately dirty data" |
| `day 12-13: S28 (43%) <- DARK STATION` | "the bottleneck just moved to a station with no sensors" |
| `[OK ] LOOSE_FASTENER -> S12:torque_nm low` | "found the fault we hid, and never told it about" |
| `throughput component rho = 1.0 / quality rho = -0.06` | "half our method transfers to a real plant, half doesn't" |

Do not cut the 60 seconds of runtime. It is a selling point.

## Take 2 — dashboard (about 2 minutes)

```bash
open outputs/dashboard_alpha.html      # macOS
# xdg-open on Linux, start on Windows
```

Tab order, no backtracking:

1. **Floor supervisor** — the constraint now, alerts capped at five per shift
2. **Plant manager** — the back-trace table, then the retrofit plan
3. **Leadership** — scenario range, then the trust ledger
4. **Validation** — soft-sensor table, then SPC, then the honest caveat box

Scroll slowly. Let each table sit for two seconds before you speak over it.

## Take 3 — portability and proof (about 45 seconds)

```bash
python run_demo.py --line beta --fast
python eval/run_eval.py --seeds 3
```

> "Same code, different plant — 28 stations, 57% instrumented, config file only.
> And this is our test suite: ten checks against ground truth. It fails the
> build if any number drifts."

End on the wall of `PASS`. Do not add anything after it.

## Export

- 1080p, H.264, MP4
- Under 5 minutes total
- Name it as the challenge requires, then put the link at the very top of
  `README.md` before you push

## The one thing not to cut

The limitations beat around 2:30. Volunteering your weak numbers is what makes
the strong ones believable, and it is the most credible thirty seconds you have.
