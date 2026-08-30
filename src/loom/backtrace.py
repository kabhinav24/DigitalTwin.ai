"""
Genealogy back-trace: from a confirmed defect to a ranked, evidenced cause.

The problem this solves is the one the Round 2 brief calls out and the reviewed
literature leaves open: a defect created early in the line surfaces at a much
later inspection point, by which time many downstream units already carry it.
Ragazzini et al. (2024) predict bottlenecks but not defects. Wang et al. (2025)
predict anomaly *events* on a knowledge graph but do not attribute a confirmed
finding back to its origin, and close by naming interpretability as their main
limitation. Loom fills that gap with a method that needs no training data at
all - which matters, because the events worth root-causing are usually the
ones that have never happened before.

Method
------
Take the cohort of units that reached the detecting gate inside a recent
window. For each candidate cause, build a 2x2 table of (exposed / not) against
(defective / not), score it with a risk ratio and a Fisher exact test, then
control the false discovery rate across all candidates with Benjamini-Hochberg.

Candidates are deliberately *not* just "passed station s" - on a serial line
every unit passes every station, so that test has no power. What discriminates
is exposure to a station **while it was off-nominal**:

    STATION_SIGNAL  a process value at station s was out of band for this unit
    STATION_DWELL   dwell at a dark station s was out of band for this unit
    STATION_WINDOW  the unit passed station s inside a short sub-window
    LOT             the unit consumed supplier lot L
    OPERATOR        the unit was worked by operator O
    SHIFT           the unit was built on a given day and shift
    ENVIRONMENT     an environmental signal was out of band for this unit

Confounders
-----------
A zone-wide event such as a humidity excursion makes *every* booth in that zone
look guilty. Ranking by lift alone would blame whichever booth happened to run
the most units. So after ranking, candidates that share a zone and signal
family are consolidated into a single zone-level hypothesis, which is the
parsimonious explanation and the one that leads to the right corrective action.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .config import LineConfig
from .twin import KnowledgeGraph, LineStates

BASELINE_UNITS = 2880       # early production, used to define "in band"
Z_OUT_OF_BAND = 2.0
MIN_EXPOSED = 25
MIN_DEFECTS = 8


@dataclass
class Hypothesis:
    kind: str
    label: str
    detail: str
    n_exposed: int
    n_exposed_defective: int
    rate_exposed: float
    rate_unexposed: float
    lift: float
    p_value: float
    q_value: float = 1.0
    confidence: float = 0.0
    zone: str = ""
    station: str = ""

    def as_row(self) -> dict:
        return {
            "kind": self.kind, "label": self.label, "detail": self.detail,
            "station": self.station, "zone": self.zone,
            "n_exposed": self.n_exposed, "n_exp_defective": self.n_exposed_defective,
            "rate_exposed": round(self.rate_exposed, 4),
            "rate_unexposed": round(self.rate_unexposed, 4),
            "lift": round(self.lift, 2), "p": self.p_value, "q": self.q_value,
            "confidence": round(self.confidence, 3),
        }


class BackTracer:
    """Attributes confirmed findings to ranked causes over the digital thread."""

    def __init__(self, cfg: LineConfig, kg: KnowledgeGraph, states: LineStates,
                 signals: dict[str, dict[str, np.ndarray]], units: pd.DataFrame):
        self.cfg = cfg
        self.kg = kg
        self.states = states
        self.signals = signals
        self.units = units
        self._baseline = self._fit_baseline()

    # ---------------- baseline bands ----------------

    def _fit_baseline(self) -> dict:
        base = slice(0, BASELINE_UNITS)
        out: dict = {}
        for st in self.cfg.stations:
            for name, arr in self.signals.get(st.sid, {}).items():
                if name == "cycle_time_s":
                    continue
                out[(st.sid, name)] = (float(arr[base].mean()),
                                       float(arr[base].std() + 1e-9))
            if not st.instrumented:
                d = self.states.dwell[base, st.index - 1]
                out[(st.sid, "dwell_s")] = (float(d.mean()), float(d.std() + 1e-9))
        return out

    # ---------------- candidate generation ----------------

    def _candidates(self, cohort: np.ndarray) -> list[tuple[str, str, str, np.ndarray, str, str]]:
        """(kind, label, detail, exposed_mask_over_cohort, station, zone)"""
        out = []
        env_signals = {"booth_humidity_pct", "booth_temp_c", "zone_temp_c"}

        for st in self.cfg.stations:
            j = st.index - 1
            for name, arr in self.signals.get(st.sid, {}).items():
                if name == "cycle_time_s":
                    continue
                mu, sd = self._baseline[(st.sid, name)]
                z = (arr[cohort] - mu) / sd
                kind = "ENVIRONMENT" if name in env_signals else "STATION_SIGNAL"
                for direction, mask in (("high", z > Z_OUT_OF_BAND),
                                        ("low", z < -Z_OUT_OF_BAND)):
                    if mask.sum() >= MIN_EXPOSED:
                        out.append((kind, f"{st.sid}:{name} {direction}",
                                    f"{name} more than {Z_OUT_OF_BAND}sd {direction} "
                                    f"of the commissioning baseline at {st.sid}",
                                    mask, st.sid, st.zone))

            if not st.instrumented:
                mu, sd = self._baseline[(st.sid, "dwell_s")]
                z = (self.states.dwell[cohort, j] - mu) / sd
                mask = z > Z_OUT_OF_BAND
                if mask.sum() >= MIN_EXPOSED:
                    out.append(("STATION_DWELL", f"{st.sid}:dwell high",
                                f"soft-sensed dwell at un-instrumented {st.sid} "
                                f"more than {Z_OUT_OF_BAND}sd above baseline",
                                mask, st.sid, st.zone))

        # transient station events: sub-windows of the cohort's own timespan
        t = self.states.depart[cohort, -1]
        edges = np.quantile(t, np.linspace(0, 1, 7))
        for st in self.cfg.stations:
            j = st.index - 1
            ts = self.states.start[cohort, j]
            for k in range(6):
                mask = (ts >= edges[k]) & (ts < edges[k + 1])
                if mask.sum() >= MIN_EXPOSED:
                    out.append(("STATION_WINDOW", f"{st.sid}:w{k}",
                                f"passed {st.sid} during sub-window {k + 1} of 6",
                                mask, st.sid, st.zone))

        u = self.units.iloc[cohort]
        for lot, mask in _group_masks(u["lot_id"].to_numpy()):
            if mask.sum() >= MIN_EXPOSED:
                out.append(("LOT", f"lot:{lot}", f"consumed supplier lot {lot}",
                            mask, "", ""))
        for z in sorted({s.zone for s in self.cfg.stations}):
            col = f"operator_{z}"
            if col not in u:
                continue
            for op, mask in _group_masks(u[col].to_numpy()):
                if mask.sum() >= MIN_EXPOSED:
                    out.append(("OPERATOR", f"op:{op}", f"worked by {op} in {z}",
                                mask, "", z))
        key = u["day"].astype(str) + "-" + u["shift"].astype(str)
        for sh, mask in _group_masks(key.to_numpy()):
            if mask.sum() >= MIN_EXPOSED:
                out.append(("SHIFT", f"shift:{sh}", f"built on day-shift {sh}",
                            mask, "", ""))
        return out

    # ---------------- the attribution itself ----------------

    def trace(self, defect_type: str, gate: str, inspections: pd.DataFrame,
              now_unit: int, cohort_units: int = 2880,
              fdr: float = 0.05, top_k: int = 6) -> dict:
        lo = max(0, now_unit - cohort_units)
        cohort = np.arange(lo, now_unit)

        found = inspections[(inspections["gate"] == gate)
                            & (inspections["defect_type"] == defect_type)]
        defective = np.zeros(len(cohort), dtype=bool)
        idx = found["unit"].to_numpy()
        idx = idx[(idx >= lo) & (idx < now_unit)]
        defective[idx - lo] = True

        if defective.sum() < MIN_DEFECTS:
            return {"status": "insufficient_findings",
                    "defect_type": defect_type, "n_findings": int(defective.sum()),
                    "hypotheses": pd.DataFrame(), "cohort": (lo, now_unit)}

        cands = self._candidates(cohort)
        hyps: list[Hypothesis] = []
        for kind, label, detail, mask, sid, zone in cands:
            ne = int(mask.sum())
            nu = int((~mask).sum())
            if ne < MIN_EXPOSED or nu < MIN_EXPOSED:
                continue
            a = int((mask & defective).sum())
            b = ne - a
            c = int((~mask & defective).sum())
            d = nu - c
            r_e = a / ne
            r_u = c / max(nu, 1)
            if r_e <= r_u:
                continue                       # only elevated-risk causes
            lift = r_e / max(r_u, 1e-9)
            try:
                _, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            except ValueError:
                continue
            hyps.append(Hypothesis(kind, label, detail, ne, a, r_e, r_u,
                                   float(lift), float(p), zone=zone, station=sid))

        if not hyps:
            return {"status": "no_candidate", "defect_type": defect_type,
                    "hypotheses": pd.DataFrame(), "cohort": (lo, now_unit)}

        # Benjamini-Hochberg across every candidate tested
        ps = np.array([h.p_value for h in hyps])
        order = np.argsort(ps)
        m = len(ps)
        q = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            val = ps[i] * m / (rank + 1)
            prev = min(prev, val)
            q[i] = min(prev, 1.0)
        for h, qv in zip(hyps, q):
            h.q_value = float(qv)
            # confidence blends effect size with statistical support and
            # saturates, so a huge lift on 30 units cannot outrank a solid one
            h.confidence = float(np.clip(
                (1 - h.q_value) * np.tanh((h.lift - 1) / 2.0)
                * min(1.0, h.n_exposed_defective / 25.0), 0, 1))

        survivors = [h for h in hyps if h.q_value <= fdr]
        survivors.sort(key=lambda h: (-h.confidence, h.q_value))
        consolidated = _consolidate(survivors)

        top = consolidated[:top_k]
        return {
            "status": "ok",
            "defect_type": defect_type,
            "gate": gate,
            "cohort": (lo, now_unit),
            "n_cohort": len(cohort),
            "n_findings": int(defective.sum()),
            "n_candidates_tested": m,
            "n_significant": len(survivors),
            "hypotheses": pd.DataFrame([h.as_row() for h in top]),
            "top": top[0] if top else None,
        }

    # ---------------- containment ----------------

    def containment(self, hypothesis: Hypothesis, now_unit: int,
                    cohort_units: int = 2880, lookahead_units: int = 400) -> dict:
        """Which units still in the line share the top hypothesis's exposure.

        This is what turns a diagnosis into an action: the same root cause is
        already inside vehicles further down the line, and naming them narrows
        containment from a date-range sweep to a specific VIN list.
        """
        n_total = self.states.depart.shape[0]
        lo = max(0, now_unit - cohort_units)
        hi = min(n_total, now_unit + lookahead_units)
        cohort = np.arange(lo, hi)
        raw = self._candidates(cohort)
        cands = {(k, lab): m for k, lab, _, m, _, _ in raw}
        mask = cands.get((hypothesis.kind, hypothesis.label))

        if mask is None and hypothesis.kind.startswith("ZONE_"):
            # a consolidated zone hypothesis is the union of its members
            zone, sig = hypothesis.label.split(":", 1)
            parts = [m for k, lab, _, m, _, z in raw
                     if z == zone and lab.split(":", 1)[1].startswith(sig)]
            if parts:
                mask = np.logical_or.reduce(parts)
        if mask is None and hypothesis.kind == "TIME_WINDOW":
            win = hypothesis.label.split(":", 1)[1]
            parts = [m for k, lab, _, m, _, _ in raw
                     if k == "STATION_WINDOW" and lab.endswith(f":{win}")]
            if parts:
                mask = np.logical_or.reduce(parts)
        if mask is None:
            return {"status": "unavailable", "hypothesis": hypothesis.label}

        exposed = cohort[mask]
        now_s = float(self.states.depart[now_unit - 1, -1])
        released = self.states.start[:, 0]
        finished = self.states.depart[:, -1]

        in_line = exposed[(released[exposed] <= now_s) & (finished[exposed] > now_s)]
        shipped = exposed[finished[exposed] <= now_s]
        not_started = exposed[released[exposed] > now_s]
        return {
            "status": "ok",
            "hypothesis": hypothesis.label,
            "n_exposed": int(len(exposed)),
            "n_still_in_line": int(len(in_line)),
            "n_already_shipped": int(len(shipped)),
            "n_not_yet_started": int(len(not_started)),
            "vins_still_in_line": self.units.iloc[in_line]["vin"].tolist()[:250],
            "vins_sample_shipped": self.units.iloc[shipped]["vin"].tolist()[:25],
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _group_masks(values: np.ndarray):
    for v in pd.unique(values):
        yield v, values == v


def _consolidate(hyps: list[Hypothesis]) -> list[Hypothesis]:
    """Collapse zone-wide effects into a single hypothesis.

    When a humidity excursion hits a paint shop, every booth in that zone looks
    guilty. Reporting four near-identical station hypotheses sends a team
    chasing four robots when the answer is one air handler.
    """
    by_family: dict[tuple, list[Hypothesis]] = {}
    for h in hyps:
        if h.kind in ("STATION_SIGNAL", "ENVIRONMENT") and h.zone:
            sig = h.label.split(":")[1]
            by_family.setdefault((h.zone, sig, h.kind), []).append(h)

    absorbed: set[int] = set()
    merged: list[Hypothesis] = []

    # A transient event shows up as the same sub-window on every station, which
    # is one hypothesis restated forty times. Collapse it into a time window.
    by_window: dict[str, list[Hypothesis]] = {}
    for h in hyps:
        if h.kind == "STATION_WINDOW":
            by_window.setdefault(h.label.split(":")[1], []).append(h)
    for win, group in by_window.items():
        if len(group) < 6:
            continue
        lifts = np.array([g.lift for g in group])
        if lifts.std() / max(lifts.mean(), 1e-9) > 0.30:
            continue
        best = max(group, key=lambda g: g.confidence)
        merged.append(Hypothesis(
            kind="TIME_WINDOW", label=f"window:{win}",
            detail=(f"elevated rate across {len(group)} stations in the same "
                    f"sub-window. Points at a line-wide or time-based cause "
                    f"(material, shift, utility) rather than one station."),
            n_exposed=best.n_exposed, n_exposed_defective=best.n_exposed_defective,
            rate_exposed=best.rate_exposed, rate_unexposed=best.rate_unexposed,
            lift=float(lifts.mean()), p_value=best.p_value, q_value=best.q_value,
            confidence=float(best.confidence * 0.9), zone="", station="",
        ))
        absorbed.update(id(g) for g in group)
    for (zone, sig, kind), group in by_family.items():
        if len(group) < 3:
            continue
        lifts = np.array([g.lift for g in group])
        # only consolidate when the stations really do look alike
        if lifts.std() / max(lifts.mean(), 1e-9) > 0.45:
            continue
        best = max(group, key=lambda g: g.confidence)
        merged.append(Hypothesis(
            kind=f"ZONE_{kind}",
            label=f"{zone}:{sig}",
            detail=(f"{sig} out of band across {len(group)} stations in {zone}. "
                    f"Consistent with a zone-level cause (utility, air handling "
                    f"or material) rather than any single station."),
            n_exposed=int(np.mean([g.n_exposed for g in group])),
            n_exposed_defective=int(np.mean([g.n_exposed_defective for g in group])),
            rate_exposed=float(np.mean([g.rate_exposed for g in group])),
            rate_unexposed=float(np.mean([g.rate_unexposed for g in group])),
            lift=float(lifts.mean()),
            p_value=float(min(g.p_value for g in group)),
            q_value=float(min(g.q_value for g in group)),
            confidence=float(min(0.99, best.confidence * 1.12)),
            zone=zone, station="",
        ))
        absorbed.update(id(g) for g in group)

    kept = [h for h in hyps if id(h) not in absorbed]
    out = kept + merged
    out.sort(key=lambda h: (-h.confidence, h.q_value))
    return out
