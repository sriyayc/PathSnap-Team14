"""HMM + Viterbi route reconstruction.

Hidden state at step i : one candidate arc (with its projection position t) for GPS observation i.
Emission              : log P_clf(candidate | features), renormalised over the point's candidates.
Transition (i-1 -> i) : Newson & Krumm style. With d_gc the straight-line distance between the
                        two GPS fixes and d_route the network distance between the two projected
                        positions,  log T = -|d_route - d_gc| / beta(dt).
                        beta grows with the time gap so long gaps tolerate detours.
                        Unreachable pairs get a large finite penalty (not -inf) so a broken network
                        never kills the whole path.
Breaks                : a point with no candidates is skipped (inherits the previous prediction);
                        a time gap > `gap_reset_s` restarts the chain (no transition constraint).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HMMConfig:
    beta: float = 8.0               # m, transition scale for 1-s steps
    beta_per_s: float = 2.0         # m added to beta per second of time gap
    emission_weight: float = 1.0    # multiplier on log-emission
    unreachable_penalty: float = 40.0
    max_route_factor: float = 4.0   # Dijkstra search limit = factor * d_gc + slack
    route_slack: float = 300.0
    gap_reset_s: float = 120.0
    min_prob: float = 1e-4
    backward_factor: float = 3.0    # cost multiplier for backward movement along a directed arc
    backward_free_m: float = 0.0    # backward movement up to this many metres is treated as GPS jitter (no extra cost)


def viterbi(net, xy, ts, cand_arcs, cand_t, cand_p, cfg: HMMConfig = HMMConfig()):
    """cand_* are lists (len n) of arrays per observation. Returns predicted arc per observation
    (-1 when a point has no candidates and nothing to inherit), plus diagnostics."""
    n = len(xy)
    pred = np.full(n, -1, dtype=np.int64)
    diag = {"unreachable_steps": 0, "resets": 0}
    net.backward_factor = cfg.backward_factor
    net.backward_free_m = cfg.backward_free_m

    def log_emis(p):
        p = np.clip(p, cfg.min_prob, 1.0)
        return cfg.emission_weight * np.log(p / p.sum())

    segments = []  # list of (indices, back pointers, scores) chains
    cur_idx, cur_back = [], []
    prev_score = None
    prev_i = None
    for i in range(n):
        if len(cand_arcs[i]) == 0:
            continue
        e = log_emis(cand_p[i])
        if prev_i is None or ts[i] - ts[prev_i] > cfg.gap_reset_s:
            if prev_i is not None:
                diag["resets"] += 1
                segments.append((cur_idx, cur_back, prev_score))
            cur_idx, cur_back = [i], [None]
            prev_score = e
            prev_i = i
            continue
        dt = max(ts[i] - ts[prev_i], 1.0)
        d_gc = float(np.hypot(*(xy[i] - xy[prev_i])))
        beta = cfg.beta + cfg.beta_per_s * (dt - 1.0)
        limit = cfg.max_route_factor * d_gc + cfg.route_slack
        A, tA = cand_arcs[prev_i], cand_t[prev_i]
        B, tB = cand_arcs[i], cand_t[i]
        logT = np.empty((len(A), len(B)))
        for a in range(len(A)):
            for b in range(len(B)):
                dr = net.route_distance(int(A[a]), float(tA[a]), int(B[b]), float(tB[b]), limit)
                logT[a, b] = -cfg.unreachable_penalty if not np.isfinite(dr) else -abs(dr - d_gc) / beta
        tot = prev_score[:, None] + logT
        back = tot.argmax(0)
        score = tot[back, np.arange(len(B))] + e
        if not np.isfinite(logT.max()) or logT.max() <= -cfg.unreachable_penalty:
            diag["unreachable_steps"] += 1
        cur_idx.append(i)
        cur_back.append(back)
        prev_score = score
        prev_i = i
    if prev_i is not None:
        segments.append((cur_idx, cur_back, prev_score))

    for idx, backs, score in segments:
        s = int(np.argmax(score))
        for k in range(len(idx) - 1, -1, -1):
            pred[idx[k]] = cand_arcs[idx[k]][s]
            if backs[k] is not None:
                s = int(backs[k][s])
    # points without candidates inherit previous prediction
    last = -1
    for i in range(n):
        if pred[i] == -1:
            pred[i] = last
        else:
            last = pred[i]
    return pred, diag


def decode_table(net, xy, ts, table, score_col="p", cfg: HMMConfig = HMMConfig()):
    """Run Viterbi using a candidate table (rows: pt, arc, t_proj, score)."""
    n = len(xy)
    cand_arcs = [np.zeros(0, np.int64)] * n
    cand_t = [np.zeros(0)] * n
    cand_p = [np.zeros(0)] * n
    for pt, g in table.groupby("pt", sort=True):
        cand_arcs[pt] = g.arc.values.astype(np.int64)
        cand_t[pt] = g.t_proj.values
        cand_p[pt] = g[score_col].values
    return viterbi(net, xy, ts, cand_arcs, cand_t, cand_p, cfg)
