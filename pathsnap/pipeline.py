"""End-to-end inference pipeline used by the API, the prediction maps and the final evaluation.

predict_trip()  : raw trip folder -> predicted arc for EVERY original GPS observation + a connected
                  route. It loads the trip WITHOUT the reference route (include_route=False).
score_against_reference() : separate, server-side only; reads the .route file and computes accuracy.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import joblib
import numpy as np

from .data import load_route, load_trip
from .features import DEFAULT_CFG, FEATURES, prepare_trip
from .hmm import HMMConfig, decode_table
from .labels import align_to_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Predictor:
    def __init__(self, model_name="lgbm_tuned", hmm_config_path=None):
        mdir = os.path.join(ROOT, "artifacts", "models")
        path = os.path.join(mdir, f"{model_name}.joblib")
        if not os.path.exists(path):  # fall back to the untuned model if tuning has not been run
            model_name, path = "lgbm", os.path.join(mdir, "lgbm.joblib")
        self.model_name = model_name
        self.model = joblib.load(path)
        hp = hmm_config_path or os.path.join(ROOT, "artifacts", "hmm_config.json")
        self.hmm_cfg = HMMConfig()
        if os.path.exists(hp):
            with open(hp) as fh:
                self.hmm_cfg = HMMConfig(**json.load(fh))

    def predict_trip(self, folder):
        t0 = time.perf_counter()
        trip = load_trip(folder, include_route=False)
        assert trip.route is None
        net, keep, tab, _ = prepare_trip(trip, DEFAULT_CFG, with_labels=False)
        t1 = time.perf_counter()
        tab["p"] = self.model.predict_proba(tab[FEATURES].values)[:, 1] if len(tab) else []
        t2 = time.perf_counter()
        xy, ts = trip.xy[keep], trip.track[keep, 2]
        hmm_kept, diag = decode_table(net, xy, ts, tab, "p", self.hmm_cfg)
        t3 = time.perf_counter()
        clf_kept = np.full(len(keep), -1)
        if len(tab):
            best = tab.loc[tab.groupby("pt").p.idxmax()]
            clf_kept[best.pt.values] = best.arc.values
        owner = np.searchsorted(keep, np.arange(len(trip.track)), side="right") - 1
        clf_all = clf_kept[owner]
        hmm_all = interpolate_dropped(net, trip.xy, keep, hmm_kept)
        route = connect_route(net, hmm_kept)
        return trip, {
            "trip_id": trip.trip_id,
            "n_observations": int(len(trip.track)),
            "n_decoded": int(len(keep)),
            "pred_arcs": hmm_all, "clf_arcs": clf_all,
            "pred_arcs_kept": hmm_kept, "clf_arcs_kept": clf_kept, "kept_idx": keep,
            "route_arcs": route,
            "timing_s": {"features": t1 - t0, "classifier": t2 - t1, "viterbi": t3 - t2, "total": time.perf_counter() - t0},
            "hmm_diagnostics": diag, "model": self.model_name, "hmm_config": asdict(self.hmm_cfg),
        }


def interpolate_dropped(net, xy_all, keep, pred_kept):
    """Assign an arc to every original observation. Kept observations use their Viterbi arc. An
    observation removed by thinning, lying between kept observations k0 and k1, gets the closest arc on
    the decoded path from pred[k0] to pred[k1] (both arcs plus the shortest connecting arcs), so
    observations where the vehicle crosses onto the next road are not stuck on the previous one."""
    from .network import point_segment
    n = len(xy_all)
    out = np.full(n, -1, dtype=np.int64)
    out[keep] = pred_kept
    for j in range(len(keep)):
        lo = keep[j]
        hi = keep[j + 1] if j + 1 < len(keep) else n
        if hi - lo <= 1:
            continue
        a0 = int(pred_kept[j])
        a1 = int(pred_kept[j + 1]) if j + 1 < len(keep) else a0
        idx = np.arange(lo + 1, hi)
        if a0 < 0:
            out[idx] = a1
            continue
        if a1 < 0 or a1 == a0:
            out[idx] = a0
            continue
        path = [a0]
        if net.arcs[a0, 1] != net.arcs[a1, 0]:
            gap = net.shortest_arc_path(int(net.arcs[a0, 1]), int(net.arcs[a1, 0]), limit=1000.0)
            path += gap or []
        path.append(a1)
        path = np.asarray(path)
        p = xy_all[idx]
        d, _ = point_segment(p[:, :1], p[:, 1:2], net.ax[path][None], net.ay[path][None], net.bx[path][None], net.by[path][None])
        out[idx] = path[d.argmin(1)]
    return out


def connect_route(net, arc_seq):
    """Collapse repeated arcs and fill gaps between non-adjacent consecutive arcs with shortest paths,
    giving a drivable, connected route."""
    seq = [int(a) for a in arc_seq if a >= 0]
    route = []
    for a in seq:
        if route and route[-1] == a:
            continue
        if route and net.arcs[route[-1], 1] != net.arcs[a, 0]:
            gap = net.shortest_arc_path(int(net.arcs[route[-1], 1]), int(net.arcs[a, 0]))
            if gap:
                route.extend(gap)
        route.append(a)
    return route


def score_against_reference(trip, folder, result, max_label_dist=40.0):
    """Server-side evaluation. Labels come from the .route file and never enter the prediction path."""
    ref = load_route(folder, trip.trip_id)
    if ref is None:
        return None
    arcs = trip.arcs
    out = {}
    for tag, xy_idx, pred_h, pred_c in [
        ("all_observations", np.arange(len(trip.track)), result["pred_arcs"], result["clf_arcs"]),
        ("decoded_observations", result["kept_idx"], result["pred_arcs_kept"], result["clf_arcs_kept"]),
    ]:
        lab, _, _ = align_to_route(trip.xy[xy_idx], trip.node_xy, arcs, ref, max_label_dist)
        ok = lab >= 0
        def acc(p):
            v = ok & (p >= 0)
            return float((arcs[p[v]] == arcs[lab[v]]).all(1).sum() / max(ok.sum(), 1))
        out[tag] = {"labeled": int(ok.sum()), "hmm_top1": acc(pred_h), "classifier_top1": acc(pred_c)}
    # route-level: length-weighted overlap of undirected road segments
    L = np.hypot(*(trip.node_xy[arcs[:, 1]] - trip.node_xy[arcs[:, 0]]).T)
    key = lambda a: tuple(sorted(arcs[a]))
    pr = {key(a): L[a] for a in set(result["route_arcs"])}
    rf = {key(a): L[a] for a in set(ref.tolist())}
    inter = sum(v for k, v in pr.items() if k in rf)
    p = inter / max(sum(pr.values()), 1e-9)
    r = inter / max(sum(rf.values()), 1e-9)
    route_arcs = result["route_arcs"]
    cont = np.mean([arcs[route_arcs[i], 1] == arcs[route_arcs[i + 1], 0] for i in range(len(route_arcs) - 1)]) if len(route_arcs) > 1 else 1.0
    out["route"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "continuity": float(cont), "pred_km": sum(pr.values()) / 1000, "ref_km": sum(rf.values()) / 1000}
    return out
