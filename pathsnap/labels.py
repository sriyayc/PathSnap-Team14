"""Derive per-observation reference labels from the arc-sequence route.

The .route file lists the arcs traversed but NOT which GPS point belongs to which arc.
We align points to route positions with a monotone dynamic programme:

    D[i, j] = c[i, j] + min_{k <= j} D[i-1, k]

where c[i, j] = min(dist(point i, route arc j), CAP). Route position can only move forward
(or stay), so a road traversed twice in the trip is assigned to the right visit.
Points whose aligned distance exceeds `max_label_dist` are marked unlabeled (-1): they lie
beyond the reference route's coverage or the reference itself is offset (e.g. trip 33).
"""
from __future__ import annotations

import numpy as np

from .network import point_segment

CAP = 200.0


def align_to_route(xy: np.ndarray, node_xy: np.ndarray, arcs: np.ndarray, route: np.ndarray,
                   max_label_dist: float = 40.0):
    a = node_xy[arcs[route, 0]]
    b = node_xy[arcs[route, 1]]
    n, R = len(xy), len(route)
    c, _ = point_segment(xy[:, :1], xy[:, 1:2], a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1])
    c = np.minimum(c, CAP)

    back = np.empty((n, R), dtype=np.int32)
    D = c[0].copy()
    back[0] = np.arange(R)
    for i in range(1, n):
        # prefix minimum of previous row, plus the index where that minimum occurs
        cm = np.minimum.accumulate(D)
        is_new = np.r_[True, D[1:] < cm[:-1]]
        back[i] = np.maximum.accumulate(np.where(is_new, np.arange(R), 0))
        D = c[i] + cm
    j = int(np.argmin(D))
    pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i, j] if i > 0 else j
    # back[i, j] gives best previous position for row i ending at j
    dist = c[np.arange(n), pos]
    label_arc = route[pos].copy()
    label_arc[dist > max_label_dist] = -1
    return label_arc, pos, dist
