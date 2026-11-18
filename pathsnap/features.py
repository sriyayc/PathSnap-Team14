"""Candidate generation + feature engineering.

One row per (GPS observation, candidate arc). The classifier predicts P(candidate is the
reference arc | features); the HMM later uses these probabilities as emissions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Trip, thin_track
from .labels import align_to_route
from .network import RoadNetwork, point_segment

DEFAULT_CFG = dict(
    thin_dist=8.0,        # m, drop GPS points closer than this to the previous kept point
    radius=60.0,          # m, candidate search radius
    k=10,                 # max candidates per point
    max_label_dist=40.0,  # m, points aligned farther than this from the route are unlabeled
    window=2,             # +/- neighbours for the windowed distance feature
)

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "head_diff", "cos_head", "head_rank", "head_reliable",
    "t_proj", "arc_len", "n_cands",
    "speed", "dt_prev", "dt_next",
    "dist_prev", "dist_next", "win_mean_dist", "win_max_dist",
    "out_deg_to", "in_deg_from",
]


def angle_diff(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def prepare_trip(trip: Trip, cfg=DEFAULT_CFG, with_labels=True):
    """Returns (net, kept_idx, table DataFrame, point_info DataFrame)."""
    keep = thin_track(trip.xy, cfg["thin_dist"])
    xy = trip.xy[keep]
    ts = trip.track[keep, 2]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    net = RoadNetwork(trip.node_xy, trip.arcs, bbox=(x0, y0, x1, y1), margin=cfg["radius"] + 200)
    cands = net.candidates(xy, cfg["radius"], cfg["k"])
    n = len(xy)

    # GPS kinematics on the thinned sequence
    nxt = np.r_[np.arange(1, n), n - 1]
    prv = np.r_[0, np.arange(0, n - 1)]
    vx = xy[nxt, 0] - xy[prv, 0]
    vy = xy[nxt, 1] - xy[prv, 1]
    gps_head = np.degrees(np.arctan2(vx, vy)) % 360
    step = np.hypot(vx, vy)
    dt_prev = np.r_[np.nan, np.diff(ts)]
    dt_next = np.r_[np.diff(ts), np.nan]
    span = ts[nxt] - ts[prv]
    speed = np.where(span > 0, step / np.where(span > 0, span, 1), 0.0)

    label_arc = np.full(n, -1)
    if with_labels and trip.route is not None:
        label_arc, _, _ = align_to_route(xy, trip.node_xy, trip.arcs, trip.route, cfg["max_label_dist"])

    arc_key = trip.arcs[:, 0].astype(np.int64) * 10_000_000 + trip.arcs[:, 1]

    rows = []
    W = cfg["window"]
    for i, (ids, d, t) in enumerate(cands):
        m = len(ids)
        if m == 0:
            continue
        hd = angle_diff(gps_head[i], net.bearing[ids])
        lo, hi = max(0, i - W), min(n, i + W + 1)
        wxy = xy[lo:hi]
        wd, _ = point_segment(wxy[:, :1], wxy[:, 1:2], net.ax[ids][None], net.ay[ids][None],
                              net.bx[ids][None], net.by[ids][None])
        dp = wd[i - 1 - lo] if i > 0 else d
        dn = wd[i + 1 - lo] if i < n - 1 else d
        lab = label_arc[i]
        y = (arc_key[ids] == arc_key[lab]).astype(np.int8) if lab >= 0 else np.full(m, -1, np.int8)
        rows.append(pd.DataFrame({
            "pt": i, "arc": ids,
            "dist": d, "dist_rank": np.arange(m), "dist_minus_min": d - d[0],
            "dist_ratio": d / (d[0] + 1.0),
            "head_diff": hd, "cos_head": np.cos(np.radians(hd)),
            "head_rank": np.argsort(np.argsort(hd, kind="stable"), kind="stable"),
            "head_reliable": float(step[i] > 10.0),
            "t_proj": t, "arc_len": net.length[ids], "n_cands": m,
            "speed": speed[i], "dt_prev": dt_prev[i], "dt_next": dt_next[i],
            "dist_prev": dp, "dist_next": dn,
            "win_mean_dist": wd.mean(0), "win_max_dist": wd.max(0),
            "out_deg_to": net.out_deg[trip.arcs[ids, 1]], "in_deg_from": net.in_deg[trip.arcs[ids, 0]],
            "y": y,
        }))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["pt", "arc", *FEATURES, "y"])
    table.insert(0, "trip", trip.trip_id)
    table[["dt_prev", "dt_next"]] = table[["dt_prev", "dt_next"]].fillna(1.0)

    points = pd.DataFrame({
        "trip": trip.trip_id, "pt": np.arange(n), "orig_idx": keep, "t": ts,
        "label_arc": label_arc, "n_cands": [len(c[0]) for c in cands],
    })
    # candidate miss: point labeled but its reference arc is not among candidates
    hit = table[table.y == 1].groupby("pt").size()
    points["ref_in_cands"] = points.pt.map(hit).fillna(0).gt(0)
    return net, keep, table, points
