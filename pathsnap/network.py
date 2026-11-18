"""Road-network geometry, spatial candidate search and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised distance from points to segments. Returns (distance, t) with t in [0,1]
    the projection fraction along a->b."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * dx - px, ay + t * dy - py
    return np.hypot(qx, qy), t


class RoadNetwork:
    def __init__(self, node_xy: np.ndarray, arcs: np.ndarray, bbox=None, margin=300.0,
                 sample_step=10.0):
        self.node_xy = node_xy
        self.arcs = arcs
        a, b = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
        self.ax, self.ay, self.bx, self.by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = np.degrees(np.arctan2(self.bx - self.ax, self.by - self.ay)) % 360  # 0 = north
        n_nodes = len(node_xy)
        self.out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
        self.in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)

        # Only index arcs near the trajectory (networks have up to 300k arcs).
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            m = ((np.maximum(self.ax, self.bx) >= x0 - margin) & (np.minimum(self.ax, self.bx) <= x1 + margin) &
                 (np.maximum(self.ay, self.by) >= y0 - margin) & (np.minimum(self.ay, self.by) <= y1 + margin))
            idx = np.flatnonzero(m)
        else:
            idx = np.arange(len(arcs))
        # densify arcs into sample points every `sample_step` m for a KD-tree.
        nseg = np.maximum(1, np.ceil(self.length[idx] / sample_step).astype(int))
        owner = np.repeat(idx, nseg + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg]) if len(idx) else np.zeros(0)
        sx = self.ax[owner] + frac * (self.bx[owner] - self.ax[owner])
        sy = self.ay[owner] + frac * (self.by[owner] - self.ay[owner])
        self.sample_owner = owner
        self.sample_step = sample_step
        self.tree = cKDTree(np.column_stack([sx, sy])) if len(owner) else None

        # adjacency for Dijkstra (node -> list of (nbr, length))
        order = np.argsort(arcs[:, 0], kind="stable")
        self._adj_ptr = np.searchsorted(arcs[order, 0], np.arange(n_nodes + 1))
        self._adj_to = arcs[order, 1]
        self._adj_len = self.length[order]
        self._sp_cache: dict[tuple[int, float], dict[int, float]] = {}

    # ------------------------------------------------------------------
    def candidates(self, xy: np.ndarray, radius: float, k: int):
        """For each point return (arc_ids, dist, t) lists of up to k arcs within radius, sorted by
        distance. Directed twins (A->B and B->A) are both kept - heading separates them."""
        out = []
        if self.tree is None:
            return [(np.zeros(0, int), np.zeros(0), np.zeros(0)) for _ in xy]
        hits = self.tree.query_ball_point(xy, r=radius + self.sample_step / 2 + 0.5)
        for (px, py), h in zip(xy, hits):
            if not h:
                out.append((np.zeros(0, int), np.zeros(0), np.zeros(0)))
                continue
            ids = np.unique(self.sample_owner[h])
            d, t = point_segment(px, py, self.ax[ids], self.ay[ids], self.bx[ids], self.by[ids])
            m = d <= radius
            ids, d, t = ids[m], d[m], t[m]
            o = np.argsort(d, kind="stable")[:k]
            out.append((ids[o], d[o], t[o]))
        return out

    # ------------------------------------------------------------------
    def dist_from(self, src: int, limit: float) -> dict[int, float]:
        """Bounded Dijkstra from node `src`, exploring up to `limit` metres. Cached."""
        key = (src, limit)
        c = self._sp_cache.get(key)
        if c is not None:
            return c
        dist = {src: 0.0}
        pq = [(0.0, src)]
        ptr, to, ln = self._adj_ptr, self._adj_to, self._adj_len
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for e in range(ptr[u], ptr[u + 1]):
                v, nd = int(to[e]), d + ln[e]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._sp_cache[key] = dist
        return dist

    def route_distance(self, a: int, ta: float, b: int, tb: float, limit: float) -> float:
        """Network distance travelling from position ta on arc a to position tb on arc b
        (inf if not reachable within `limit`)."""
        if a == b:
            if tb >= ta:
                return (tb - ta) * self.length[a]
            # small backwards jitter on the same arc is tolerated as a short move
            back = (ta - tb) * self.length[a]
            return back if back < 15.0 else np.inf
        rest = (1 - ta) * self.length[a]
        d = self.dist_from(int(self.arcs[a, 1]), limit).get(int(self.arcs[b, 0]))
        if d is None:
            return np.inf
        return rest + d + tb * self.length[b]

    def connected(self, a: int, b: int) -> bool:
        return a == b or self.arcs[a, 1] == self.arcs[b, 0]
