"""
Decision layer: alert budget, trust ledger, shadow mode.

The brief warns that false alarms about defects that never materialise erode
floor trust quickly. That is a product problem before it is a modelling
problem, so Loom treats attention as a hard budget and treats its own track
record as a first-class, publicly visible object.

Three mechanisms:

**Alert budget.** A shift gets a fixed number of alerts, ranked by
`confidence x expected impact`. Everything else is written to the log but not
surfaced. A system that can raise 400 alerts raises none that anyone reads.

**Trust ledger.** Every alert is recorded with what it predicted and, once the
outcome is known, whether it was right. Running precision per alert class is
shown in the UI next to the alerts themselves. A supervisor who can see that
bottleneck alerts have been right 8 times out of 10 this month knows what to do
with the ninth.

**Shadow mode.** For a configurable period the twin issues alerts to the ledger
only, never to the floor. The ledger is what earns the right to go live - it
turns "trust our model" into "here is our record on your line".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ALERT_KINDS = ("BOTTLENECK", "DEFECT_RISK", "ROOT_CAUSE", "CONTAINMENT",
               "DATA_QUALITY")


@dataclass
class Alert:
    alert_id: str
    kind: str
    t_issued_s: float
    day: float
    target: str
    message: str
    confidence: float
    expected_impact_inr: float
    evidence: dict = field(default_factory=dict)
    surfaced: bool = False           # did it clear the attention budget
    shadow: bool = False             # issued while the twin was in shadow mode
    outcome: str | None = None       # "correct" | "incorrect" | "unresolved"
    resolved_t_s: float | None = None
    operator_action: str | None = None
    # human-in-the-loop: a ranked hypothesis is a lead, not a verdict, so a
    # quality engineer confirms or rejects it and that judgement is what the
    # ledger scores against
    human_verdict: str | None = None     # "confirmed" | "rejected" | "unsure"
    human_note: str | None = None

    @property
    def priority(self) -> float:
        return self.confidence * np.log1p(max(self.expected_impact_inr, 0.0))


class TrustLedger:
    """Append-only record of every alert and how it turned out."""

    def __init__(self, shadow_until_day: float = 0.0,
                 alerts_per_shift: int = 5):
        self.alerts: list[Alert] = []
        self.shadow_until_day = shadow_until_day
        self.alerts_per_shift = alerts_per_shift
        self._n = 0

    def issue(self, kind: str, t_issued_s: float, target: str, message: str,
              confidence: float, expected_impact_inr: float,
              evidence: dict | None = None) -> Alert:
        self._n += 1
        day = t_issued_s / 86400.0
        a = Alert(
            alert_id=f"A{self._n:05d}", kind=kind, t_issued_s=float(t_issued_s),
            day=round(day, 3), target=target, message=message,
            confidence=float(confidence),
            expected_impact_inr=float(expected_impact_inr),
            evidence=evidence or {}, shadow=day < self.shadow_until_day,
        )
        self.alerts.append(a)
        return a

    def adjudicate(self, alert_id: str, verdict: str,
                   note: str | None = None) -> None:
        """Record a human judgement on an alert.

        Loom ranks hypotheses; it does not prove causation. An engineer who
        rejects a hypothesis is giving the system its most valuable signal, so
        that verdict is stored and feeds the precision shown on the floor.
        """
        if verdict not in ("confirmed", "rejected", "unsure"):
            raise ValueError(verdict)
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.human_verdict = verdict
                a.human_note = note
                if verdict in ("confirmed", "rejected"):
                    a.outcome = "correct" if verdict == "confirmed" else "incorrect"
                return
        raise KeyError(alert_id)

    def human_feedback_summary(self) -> dict:
        df = self.to_frame()
        if df.empty or "human_verdict" not in df:
            return {}
        judged = df[df["human_verdict"].notna()]
        return {
            "n_adjudicated": int(len(judged)),
            "confirmed": int((judged["human_verdict"] == "confirmed").sum()),
            "rejected": int((judged["human_verdict"] == "rejected").sum()),
            "unsure": int((judged["human_verdict"] == "unsure").sum()),
            "confirm_rate": (round(float((judged["human_verdict"]
                                          == "confirmed").mean()), 4)
                             if len(judged) else None),
        }

    def resolve(self, alert_id: str, outcome: str, t_s: float | None = None,
                operator_action: str | None = None) -> None:
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.outcome = outcome
                a.resolved_t_s = t_s
                a.operator_action = operator_action
                return
        raise KeyError(alert_id)

    # ---------------- attention budget ----------------

    def apply_budget(self, shift_len_s: float = 8 * 3600.0) -> None:
        """Surface only the top-N alerts per shift, by priority."""
        if not self.alerts:
            return
        df = pd.DataFrame({"i": range(len(self.alerts)),
                           "shift": [int(a.t_issued_s // shift_len_s) for a in self.alerts],
                           "prio": [a.priority for a in self.alerts]})
        for _, grp in df.groupby("shift"):
            keep = set(grp.sort_values("prio", ascending=False)
                       .head(self.alerts_per_shift)["i"])
            for i in grp["i"]:
                self.alerts[i].surfaced = i in keep

    # ---------------- reporting ----------------

    def to_frame(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame([asdict(a) for a in self.alerts])

    def scorecard(self) -> pd.DataFrame:
        """Running precision per alert class - the number shown to the floor."""
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame()
        rows = []
        for kind, grp in df.groupby("kind"):
            judged = grp[grp["outcome"].isin(["correct", "incorrect"])]
            surfaced = grp[grp["surfaced"]]
            rows.append({
                "kind": kind,
                "issued": len(grp),
                "surfaced": len(surfaced),
                "suppressed_by_budget": len(grp) - len(surfaced),
                "judged": len(judged),
                "precision": (float((judged["outcome"] == "correct").mean())
                              if len(judged) else np.nan),
                "mean_confidence": float(grp["confidence"].mean()),
                "impact_flagged_inr": float(surfaced["expected_impact_inr"].sum()),
            })
        return pd.DataFrame(rows).sort_values("issued", ascending=False)

    def calibration(self, bins: int = 5) -> pd.DataFrame:
        """Is a 0.8-confidence alert right about 80% of the time?

        A confidence number nobody has checked is decoration. This is the check.
        """
        df = self.to_frame()
        df = df[df["outcome"].isin(["correct", "incorrect"])]
        if df.empty:
            return pd.DataFrame()
        df = df.assign(correct=(df["outcome"] == "correct").astype(float))
        edges = np.linspace(0, 1, bins + 1)
        df["bin"] = pd.cut(df["confidence"], edges, include_lowest=True)
        g = df.groupby("bin", observed=True).agg(
            n=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            observed_precision=("correct", "mean"),
        ).reset_index()
        g["bin"] = g["bin"].astype(str)
        g["gap"] = (g["observed_precision"] - g["mean_confidence"]).round(3)
        return g

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            [asdict(a) for a in self.alerts], indent=2, default=str))


# --------------------------------------------------------------------------
# Impact model - what an alert is worth if acted on
# --------------------------------------------------------------------------


def bottleneck_impact_inr(cycle_gap_s: float, takt_s: float,
                          shift_hours: float, econ: dict) -> float:
    """Units lost per shift when the bottleneck cycle exceeds takt."""
    if cycle_gap_s <= 0:
        return 0.0
    units_at_takt = shift_hours * 3600.0 / takt_s
    units_at_bn = shift_hours * 3600.0 / (takt_s + cycle_gap_s)
    return (units_at_takt - units_at_bn) * econ["contribution_margin_per_unit_inr"]


def defect_impact_inr(n_units_at_risk: float, p_defect: float,
                      escape_rate: float, econ: dict) -> float:
    """Rework plus the share that escapes to a warranty claim."""
    n_def = n_units_at_risk * p_defect
    return float(n_def * ((1 - escape_rate) * econ["rework_cost_per_defect_inr"]
                          + escape_rate * econ["warranty_cost_per_escape_inr"]))


def rca_impact_inr(hours_saved: float, econ: dict) -> float:
    return float(hours_saved * econ["engineer_cost_per_hour_inr"])
