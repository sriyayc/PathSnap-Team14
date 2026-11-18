"""Shared helpers for loading processed tables and evaluating a fitted model."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .features import FEATURES
from .metrics import (argmax_predictions, candidate_miss_rate, classification_metrics,
                      route_continuity, top1_accuracy)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
SEED = 42


def load_partition(part):
    tab = pd.read_pickle(os.path.join(PROC, f"{part}_cands.pkl"))
    pts = pd.read_pickle(os.path.join(PROC, f"{part}_points.pkl"))
    return tab, pts


def load_arcs(part, trips):
    out = {}
    for t in trips:
        out[t] = np.loadtxt(os.path.join(ROOT, "data", part, t, f"{t}.arcs"), dtype=np.int64, ndmin=2)
    return out


def labeled(tab):
    return tab[tab.y >= 0]


def evaluate(model, tab, pts, arcs_by_trip):
    """Full evaluation of a fitted binary classifier on a partition."""
    t0 = time.perf_counter()
    p = model.predict_proba(tab[FEATURES].values)[:, 1]
    infer_s = time.perf_counter() - t0
    tab = tab.assign(p=p)
    lab = tab.y >= 0
    res = classification_metrics(tab.y[lab].values, p[lab.values])
    preds = argmax_predictions(tab)
    res["top1"] = top1_accuracy(pts, preds, arcs_by_trip)
    res["route_continuity"] = route_continuity(pts, preds, arcs_by_trip)
    res["cand_miss_rate"] = candidate_miss_rate(pts)
    res["infer_s"] = infer_s
    res["infer_us_per_row"] = infer_s / len(tab) * 1e6
    return res, tab
