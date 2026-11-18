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

    table = _feature_table(trip, net, xy, cands, gps_head, step, speed, dt_prev, dt_next, label_arc, cfg["window"])

    points = pd.DataFrame({
        "trip": trip.trip_id, "pt": np.arange(n), "orig_idx": keep, "t": ts,
        "label_arc": label_arc, "n_cands": [len(c[0]) for c in cands],
    })
    # candidate miss: point labeled but its reference arc is not among candidates
    hit = table[table.y == 1].groupby("pt").size()
    points["ref_in_cands"] = points.pt.map(hit).fillna(0).gt(0)
    return net, keep, table, points


def _feature_table(trip, net, xy, cands, gps_head, step, speed, dt_prev, dt_next, label_arc, W):
    """Vectorised over all (observation, candidate) rows at once. Candidates of each observation are
    contiguous and sorted by distance, so per-observation quantities (nearest distance, ranks) are
    computed from each group's start offset."""
    n = len(xy)
    m = np.fromiter((len(c[0]) for c in cands), dtype=np.int64, count=n)
    if m.sum() == 0:
        table = pd.DataFrame(columns=["pt", "arc", *FEATURES, "y"])
        table.insert(0, "trip", trip.trip_id)
        return table
    nz = [c for c in cands if len(c[0])]
    ids = np.concatenate([c[0] for c in nz]).astype(np.int64)
    d = np.concatenate([c[1] for c in nz])
    t = np.concatenate([c[2] for c in nz])
    pt = np.repeat(np.arange(n), m)
    start = np.r_[0, np.cumsum(m)[:-1]]          # first row of each observation's group
    pos = np.arange(len(ids)) - start[pt]        # rank of the row within its group
    d0 = d[start[pt]]

    hd = angle_diff(gps_head[pt], net.bearing[ids])
    # stable rank of head_diff within each group (ties keep candidate order)
    order = np.lexsort((pos, hd, pt))
    head_rank = np.empty(len(ids), dtype=np.int64)
    head_rank[order] = np.arange(len(ids)) - start[pt[order]]

    # distances from the neighbouring fixes i-W..i+W to the same candidate arc
    ax, ay, bx, by = net.ax[ids], net.ay[ids], net.bx[ids], net.by[ids]
    off = {}
    for o in range(-W, W + 1):
        j = pt + o
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        dist_o, _ = point_segment(xy[jj, 0], xy[jj, 1], ax, ay, bx, by)
        off[o] = (dist_o, ok)
    wsum = np.zeros(len(ids))
    wmax = np.full(len(ids), -np.inf)
    wcnt = np.zeros(len(ids))
    for o in range(-W, W + 1):
        dist_o, ok = off[o]
        wsum = np.where(ok, wsum + dist_o, wsum)
        wmax = np.where(ok, np.maximum(wmax, dist_o), wmax)
        wcnt += ok
    dist_prev = np.where(pt > 0, off[-1][0], d) if W >= 1 else d
    dist_next = np.where(pt < n - 1, off[1][0], d) if W >= 1 else d

    arc_key = trip.arcs[:, 0].astype(np.int64) * 10_000_000 + trip.arcs[:, 1]
    lab = label_arc[pt]
    y = np.where(lab >= 0, arc_key[ids] == arc_key[np.maximum(lab, 0)], -1).astype(np.int8)

    table = pd.DataFrame({
        "trip": trip.trip_id, "pt": pt, "arc": ids,
        "dist": d, "dist_rank": pos, "dist_minus_min": d - d0,
        "dist_ratio": d / (d0 + 1.0),
        "head_diff": hd, "cos_head": np.cos(np.radians(hd)),
        "head_rank": head_rank,
        "head_reliable": (step[pt] > 10.0).astype(float),
        "t_proj": t, "arc_len": net.length[ids], "n_cands": m[pt],
        "speed": speed[pt], "dt_prev": dt_prev[pt], "dt_next": dt_next[pt],
        "dist_prev": dist_prev, "dist_next": dist_next,
        "win_mean_dist": wsum / wcnt, "win_max_dist": wmax,
        "out_deg_to": net.out_deg[trip.arcs[ids, 1]], "in_deg_from": net.in_deg[trip.arcs[ids, 0]],
        "y": y,
    })
    table[["dt_prev", "dt_next"]] = table[["dt_prev", "dt_next"]].fillna(1.0)
    return table
