"""
Per-unit defect risk, with calibrated confidence.

Wang et al. (2025) close their paper by naming interpretability and label
hunger as the two limits of a deep graph model for proactive anomaly
detection. Loom takes the opposite trade: a modest gradient-boosted model on
features drawn from the knowledge graph, wrapped in **class-conditional
conformal prediction** so that every score arrives with a coverage guarantee
rather than a number nobody can audit.

Why conformal, and why class-conditional
----------------------------------------
Split conformal gives distribution-free coverage: at level alpha the true label
falls inside the returned set at least 1 - alpha of the time, with no
assumption about the model being right. Under the heavy class imbalance of a
real line (a few percent of units carry a defect), *marginal* coverage is
achieved almost for free by predicting "no defect" everywhere, which is exactly
the failure mode a plant cannot afford. Class-conditional (Mondrian) conformal
calibrates a separate threshold per class, so coverage is guaranteed for
defective units too.

The output is a three-way decision rather than a score:

    {1}     flag  - route to augmented inspection
    {0, 1}  abstain - the model declines to call it; standard inspection
    {0}     pass

Abstention is the mechanism that keeps the alert budget honest. A model that
cannot tell should say so instead of spending a supervisor's attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import LineConfig
from .twin import LineStates, SoftSensor


# --------------------------------------------------------------------------
# Feature assembly from the knowledge graph
# --------------------------------------------------------------------------


def build_features(cfg: LineConfig, units: pd.DataFrame, states: LineStates,
                   signals: dict[str, dict[str, np.ndarray]],
                   sensor: SoftSensor, inspections: pd.DataFrame,
                   baseline_units: int = 2400,
                   upto_station: int | None = None) -> pd.DataFrame:
    """One row per unit, built only from data the twin is allowed to see.

    Signals are turned into deviations from an early-production baseline rather
    than used raw, so the model learns "this unit was processed off-nominal"
    instead of memorising absolute sensor scales that differ between plants.

    `upto_station` truncates the feature set at a station index. Predicting a
    finding at the body-shop gate using paint-shop sensor data would be
    leakage from the unit's own future; the twin must only use what had
    physically happened by the moment of the prediction.
    """
    cutoff = cfg.n_stations if upto_station is None else upto_station
    live = [s for s in cfg.stations if s.index <= cutoff]
    live_ids = {s.sid for s in live}
    live_j = [k for k, s in enumerate(cfg.stations) if s.sid in live_ids]
    n_u = len(units)
    feats: dict[str, np.ndarray] = {}
    base = slice(0, min(baseline_units, n_u))

    # --- instrumented stations: per-signal z-scores, plus type-level extremes
    type_pools: dict[str, list[np.ndarray]] = {}
    for st in live:
        sig = signals.get(st.sid, {})
        for name, arr in sig.items():
            if name == "cycle_time_s":
                continue
            mu, sd = float(arr[base].mean()), float(arr[base].std() + 1e-9)
            z = (arr - mu) / sd
            feats[f"z_{st.sid}_{name}"] = z
            type_pools.setdefault(f"{st.type_name}__{name}", []).append(z)

    for key, pool in type_pools.items():
        M = np.stack(pool, axis=1)
        feats[f"min_{key}"] = M.min(axis=1)
        feats[f"max_{key}"] = M.max(axis=1)

    # --- dark stations: soft-sensed dwell deviation
    dark_z = []
    for st in live:
        if st.instrumented:
            continue
        j = st.index - 1
        d = states.dwell[:, j]
        mu, sd = float(d[base].mean()), float(d[base].std() + 1e-9)
        z = (d - mu) / sd
        feats[f"zdwell_{st.sid}"] = z
        dark_z.append(z)
    if dark_z:
        M = np.stack(dark_z, axis=1)
        feats["dark_zdwell_max"] = M.max(axis=1)
        feats["dark_zdwell_mean"] = M.mean(axis=1)

    # --- context from the graph
    feats["starved_total_s"] = states.starved[:, live_j].sum(axis=1)
    feats["blocked_frac"] = states.blocked_flag[:, live_j].mean(axis=1)
    feats["line_dwell_total_s"] = states.dwell[:, live_j].sum(axis=1)

    df = pd.DataFrame(feats)
    df["variant"] = pd.Categorical(units["variant"]).codes
    df["shift"] = (units["shift"] == "B").astype(int)
    for z in sorted({s.zone for s in cfg.stations}):
        col = f"operator_{z}"
        if col in units:
            df[f"op_{z}"] = pd.Categorical(units[col]).codes
    # supplier lot as a rolling defect-rate encoding, computed causally so a
    # lot's own future defects never leak into its own features
    df["lot_code"] = pd.Categorical(units["lot_id"]).codes

    # --- upstream gate findings (available before final inspection)
    for gate, gidx in (("S14", 14), ("S24", 24)):
        if gidx >= cutoff:
            continue
        hit = np.zeros(n_u)
        g = inspections[inspections["gate"] == gate]
        hit[g["unit"].to_numpy()] = 1.0
        df[f"gate_{gate}_finding"] = hit

    df["unit"] = units["unit"].to_numpy()
    df["day"] = units["day"].to_numpy()
    return df


def build_label(inspections: pd.DataFrame, n_units: int,
                gate: str = "S42") -> np.ndarray:
    y = np.zeros(n_units, dtype=int)
    idx = inspections.loc[inspections["gate"] == gate, "unit"].to_numpy()
    y[idx] = 1
    return y


# --------------------------------------------------------------------------
# Conformal defect model
# --------------------------------------------------------------------------


@dataclass
class ConformalDefectModel:
    alpha: float = 0.10
    model: HistGradientBoostingClassifier | None = None
    q_by_class: dict[int, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            train_mask: np.ndarray, calib_mask: np.ndarray) -> "ConformalDefectModel":
        drop = [c for c in ("unit", "day") if c in X.columns]
        Xf = X.drop(columns=drop)
        self.feature_names = list(Xf.columns)

        self.model = HistGradientBoostingClassifier(
            max_iter=260, learning_rate=0.07, max_depth=6,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=17,
        )
        self.model.fit(Xf[train_mask], y[train_mask])

        # class-conditional (Mondrian) conformal calibration
        p_cal = self.model.predict_proba(Xf[calib_mask])
        y_cal = y[calib_mask]
        for cls in (0, 1):
            sel = y_cal == cls
            if sel.sum() < 20:
                self.q_by_class[cls] = 1.0
                continue
            scores = 1.0 - p_cal[sel, cls]          # nonconformity
            n = int(sel.sum())
            lvl = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q_by_class[cls] = float(np.quantile(scores, lvl, method="higher"))
        return self

    def recalibrate(self, X: pd.DataFrame, y: np.ndarray,
                    idx: np.ndarray) -> None:
        """Refresh the conformal thresholds on a recent window. No refit."""
        drop = [c for c in ("unit", "day") if c in X.columns]
        Xf = X.drop(columns=drop)
        p_cal = self.model.predict_proba(Xf.iloc[idx])
        y_cal = y[idx]
        for cls in (0, 1):
            sel = y_cal == cls
            if sel.sum() < 20:
                continue
            scores = 1.0 - p_cal[sel, cls]
            n = int(sel.sum())
            lvl = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q_by_class[cls] = float(np.quantile(scores, lvl, method="higher"))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in ("unit", "day") if c in X.columns]
        Xf = X.drop(columns=drop)
        p = self.model.predict_proba(Xf)
        in0 = (1.0 - p[:, 0]) <= self.q_by_class[0]
        in1 = (1.0 - p[:, 1]) <= self.q_by_class[1]

        decision = np.where(in1 & ~in0, "flag",
                    np.where(in0 & ~in1, "pass",
                     np.where(in0 & in1, "abstain", "flag")))
        return pd.DataFrame({
            "risk": p[:, 1],
            "set_has_defect": in1,
            "set_has_ok": in0,
            "decision": decision,
            # confidence = how far the winning label sits inside its own
            # calibrated region, on a 0-1 scale
            "confidence": np.where(
                decision == "flag", np.clip(1 - (1 - p[:, 1]) / max(self.q_by_class[1], 1e-6), 0, 1),
                np.where(decision == "pass",
                         np.clip(1 - (1 - p[:, 0]) / max(self.q_by_class[0], 1e-6), 0, 1),
                         0.0)),
        })

    def evaluate(self, X: pd.DataFrame, y: np.ndarray,
                 test_mask: np.ndarray) -> dict:
        out = self.predict(X[test_mask])
        yt = y[test_mask]
        cov1 = float(out.loc[yt == 1, "set_has_defect"].mean())
        cov0 = float(out.loc[yt == 0, "set_has_ok"].mean())
        flagged = out["decision"] == "flag"
        abst = out["decision"] == "abstain"
        self.metrics = {
            "n_test": int(test_mask.sum()),
            "prevalence": float(yt.mean()),
            "roc_auc": float(roc_auc_score(yt, out["risk"])),
            "pr_auc": float(average_precision_score(yt, out["risk"])),
            "coverage_defect_class": cov1,
            "coverage_ok_class": cov0,
            "target_coverage": 1 - self.alpha,
            "flag_rate": float(flagged.mean()),
            "abstain_rate": float(abst.mean()),
            "flag_precision": float(yt[flagged].mean()) if flagged.any() else float("nan"),
            "flag_recall": float(yt[flagged].sum() / max(yt.sum(), 1)),
            "pass_miss_rate": float(yt[out["decision"] == "pass"].mean()),
        }
        return self.metrics

    def top_evidence(self, X: pd.DataFrame, unit_row: int, k: int = 4) -> list[tuple[str, float]]:
        """Plain evidence for one flagged unit: the features furthest from the
        population median, in units of robust scale. Not a causal claim - it is
        what the inspector needs to know where to look."""
        drop = [c for c in ("unit", "day") if c in X.columns]
        Xf = X.drop(columns=drop)
        med = Xf.median(numeric_only=True)
        mad = (Xf - med).abs().median(numeric_only=True) + 1e-9
        row = Xf.iloc[unit_row]
        dev = ((row - med) / mad).abs().sort_values(ascending=False)
        return [(str(i), float(row[i])) for i in dev.head(k).index]


def time_split(days: np.ndarray, train_end: int, calib_end: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly chronological split. Random splits leak a drifting line's
    future into its own training set and flatter the model badly."""
    return (days <= train_end, (days > train_end) & (days <= calib_end),
            days > calib_end)


# --------------------------------------------------------------------------
# Per-defect-type deployment + rolling recalibration
# --------------------------------------------------------------------------


def build_label_typed(inspections: pd.DataFrame, n_units: int, defect_type: str,
                      gate: str = "S42") -> np.ndarray:
    y = np.zeros(n_units, dtype=int)
    sel = inspections[(inspections["gate"] == gate)
                      & (inspections["defect_type"] == defect_type)]
    y[sel["unit"].to_numpy()] = 1
    return y


def evaluate_rolling(model: ConformalDefectModel, X: pd.DataFrame, y: np.ndarray,
                     test_start: int, block: int = 480,
                     calib_window: int = 1920) -> dict:
    """Walk-forward evaluation with rolling recalibration.

    Split conformal assumes exchangeability between calibration and test. A
    drifting line breaks that, and coverage silently falls below target - the
    exact failure the semiconductor MLOps literature reports. Recalibrating on
    the most recent window before each block restores local exchangeability
    without retraining the model, which is cheap enough to run every shift.
    """
    n = len(X)
    rows = []
    for s in range(test_start, n, block):
        e = min(s + block, n)
        c0 = max(0, s - calib_window)
        model.recalibrate(X, y, np.arange(c0, s))
        out = model.predict(X.iloc[s:e])
        yt = y[s:e]
        flagged = (out["decision"] == "flag").to_numpy()
        rows.append({
            "block_start": s,
            "n": e - s,
            "prevalence": float(yt.mean()),
            "cov_defect": float(out.loc[yt == 1, "set_has_defect"].mean())
            if (yt == 1).any() else np.nan,
            "cov_ok": float(out.loc[yt == 0, "set_has_ok"].mean()),
            "flag_rate": float(flagged.mean()),
            "flag_precision": float(yt[flagged].mean()) if flagged.any() else np.nan,
            "flag_recall": float(yt[flagged].sum() / max(yt.sum(), 1)),
            "abstain_rate": float((out["decision"] == "abstain").mean()),
        })
    df = pd.DataFrame(rows)
    preq = np.arange(test_start, n)
    try:
        auc = float(roc_auc_score(y[preq], model.predict(X.iloc[preq])["risk"]))
        ap = float(average_precision_score(y[preq], model.predict(X.iloc[preq])["risk"]))
    except ValueError:
        auc, ap = float("nan"), float("nan")
    return {
        "blocks": df,
        "prequential_roc_auc": auc,
        "prequential_pr_auc": ap,
        "cov_defect_mean": float(df["cov_defect"].mean(skipna=True)),
        "cov_ok_mean": float(df["cov_ok"].mean()),
        "flag_precision_mean": float(df["flag_precision"].mean(skipna=True)),
        "flag_recall_mean": float(df["flag_recall"].mean()),
        "abstain_rate_mean": float(df["abstain_rate"].mean()),
    }


def gate_map_for(cfg) -> dict[str, tuple[str, int]]:
    """Which gate first inspects each defect type, derived from the line.

    This used to be a hard-coded table of Plant Alpha's station ids, which is
    precisely the kind of thing that turns a portability claim into a lie. It
    is now read off the config, so a line with different gates in different
    places works with no code change.
    """
    out: dict[str, tuple[str, int]] = {}
    for st in sorted(cfg.stations, key=lambda s: s.index):
        for dt in st.inspects:
            out.setdefault(dt, (st.sid, st.index - 1))
    return out


# Retained for the reference line so existing imports keep working.
GATE_FOR_TYPE = {
    "WELD_POROSITY":    ("S14", 13),
    "SEALANT_VOID":     ("S14", 13),
    "PAINT_FINISH":     ("S24", 23),
    "LOOSE_FASTENER":   ("S42", 41),
    "ELECTRICAL_FAULT": ("S42", 41),
    "FIT_MISALIGN":     ("S42", 41),
}


def fit_per_type(X_by_cutoff: dict[int, pd.DataFrame], inspections: pd.DataFrame,
                 n_units: int, defect_types: tuple[str, ...], days: np.ndarray,
                 train_end: int, calib_end: int, alpha: float = 0.10,
                 min_positives: int = 60,
                 gate_map: dict[str, tuple[str, int]] | None = None) -> dict:
    """One conformal model per defect type.

    A plant acts differently on a loose fastener and a paint blemish, so a
    single "is this unit bad" score is the wrong shape for the decision. Types
    with too few positives to calibrate are reported as unmodelled rather than
    fitted badly.
    """
    tr, ca, te = time_split(days, train_end, calib_end)
    out: dict = {}
    gmap = gate_map or GATE_FOR_TYPE
    for dt in defect_types:
        if dt not in gmap:
            out[dt] = {"status": "unmodelled", "gate": None,
                       "reason": "no gate on this line inspects for it",
                       "n_positive": 0}
            continue
        gate, cutoff = gmap[dt]
        X = X_by_cutoff[cutoff]
        y = build_label_typed(inspections, n_units, dt, gate=gate)
        if y[tr].sum() < min_positives or y[ca].sum() < 20:
            out[dt] = {"status": "unmodelled", "gate": gate,
                       "reason": f"only {int(y[tr].sum())} training positives",
                       "n_positive": int(y.sum())}
            continue
        m = ConformalDefectModel(alpha=alpha).fit(X, y, tr, ca)
        met = m.evaluate(X, y, te)
        # Prequential evaluation from the end of training onward. A single
        # held-out block at the end of the run can miss a transient event
        # entirely - the paint humidity excursion lands in the calibration
        # window, so a fixed test block would score the paint model on a
        # period in which nothing was wrong with the paint shop.
        roll = evaluate_rolling(m, X, y, test_start=int(np.flatnonzero(ca)[0]))
        out[dt] = {"status": "fitted", "model": m, "y": y, "gate": gate,
                   "cutoff_station": cutoff, "n_features": len(m.feature_names),
                   "static_metrics": met, "rolling": roll}
    return out
