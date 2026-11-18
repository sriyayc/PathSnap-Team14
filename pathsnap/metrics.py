"""Evaluation metrics for candidate classification and route reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def classification_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro"),
        "f1_pos": f1_score(y, yhat),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def argmax_predictions(table: pd.DataFrame, score_col="p"):
    """Classifier-only prediction: highest-scoring candidate for each (trip, pt)."""
    idx = table.groupby(["trip", "pt"])[score_col].idxmax()
    return table.loc[idx, ["trip", "pt", "arc"]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(points: pd.DataFrame, preds: pd.DataFrame, arcs_by_trip: dict):
    """Fraction of labeled observations whose predicted arc equals the reference arc.
    Observations with no candidates count as errors. Directed duplicate arcs (same
    from->to node pair) are treated as the same road segment."""
    m = points[points.label_arc >= 0].merge(preds, on=["trip", "pt"], how="left")
    correct = total = 0
    for tid, g in m.groupby("trip"):
        arcs = arcs_by_trip[tid]
        pa = g.pred_arc.fillna(-1).astype(int).values
        la = g.label_arc.values.astype(int)
        valid = pa >= 0
        correct += int((arcs[pa[valid]] == arcs[la[valid]]).all(1).sum())
        total += len(g)
    return correct / total if total else float("nan")


def top1_by_trip(points, preds, arcs_by_trip):
    return {tid: top1_accuracy(g, preds[preds.trip == tid], arcs_by_trip)
            for tid, g in points.groupby("trip")}


def route_continuity(points: pd.DataFrame, preds: pd.DataFrame, arcs_by_trip: dict):
    """Fraction of consecutive observation pairs whose predicted arcs are identical or
    directly connected (to-node of the first == from-node of the second)."""
    m = points.merge(preds, on=["trip", "pt"], how="inner").sort_values(["trip", "pt"])
    good = total = 0
    for tid, g in m.groupby("trip"):
        arcs = arcs_by_trip[tid]
        p = g.pred_arc.astype(int).values
        if len(p) < 2:
            continue
        a, b = p[:-1], p[1:]
        same = (a == b) | (arcs[a] == arcs[b]).all(1)
        conn = arcs[a, 1] == arcs[b, 0]
        good += int((same | conn).sum())
        total += len(a)
    return good / total if total else float("nan")


def route_set_metrics(points, preds, routes_by_trip, arcs_by_trip, lengths_by_trip):
    """Sequence-level: length-weighted precision/recall/F1 of the set of predicted road
    segments vs. the reference route (undirected node pairs)."""
    m = points.merge(preds, on=["trip", "pt"], how="inner")
    P = R = F = 0.0
    n = 0
    for tid, g in m.groupby("trip"):
        arcs, L = arcs_by_trip[tid], lengths_by_trip[tid]
        key = lambda a: tuple(sorted(arcs[a]))
        pred = {key(a): L[a] for a in set(g.pred_arc.astype(int))}
        ref = {key(a): L[a] for a in set(routes_by_trip[tid])}
        inter = sum(v for k, v in pred.items() if k in ref)
        p = inter / max(sum(pred.values()), 1e-9)
        r = inter / max(sum(ref.values()), 1e-9)
        P += p; R += r; F += (2 * p * r / (p + r) if p + r else 0); n += 1
    return {"route_precision": P / n, "route_recall": R / n, "route_f1": F / n}


def candidate_miss_rate(points):
    lab = points[points.label_arc >= 0]
    return float((~lab.ref_in_cands).mean())
