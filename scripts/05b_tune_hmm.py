"""Tune HMM decoder parameters on TRAIN trips using out-of-fold classifier probabilities.

Why out-of-fold: a classifier scored on trips it was trained on is over-confident, which would bias
the emission/transition balance. So each train trip is scored by a LightGBM fitted on the other
folds (5-fold GroupKFold), exactly like unseen data. Student-test is not used for tuning.
Outputs: results/hmm_tuning.csv, artifacts/hmm_config.json
Usage: python scripts/05b_tune_hmm.py [--n_trips 24]
"""
import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.data import load_trip  # noqa: E402
from pathsnap.experiment import SEED, labeled, load_partition  # noqa: E402
from pathsnap.features import DEFAULT_CFG, FEATURES, prepare_trip  # noqa: E402
from pathsnap.hmm import HMMConfig, decode_table  # noqa: E402

GRID = {
    "beta": [4.0, 8.0, 15.0],
    "emission_weight": [0.5, 1.0, 2.0],
    "backward_factor": [1.0, 3.0, 6.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trips", type=int, default=24)
    args = ap.parse_args()
    tab, pts = load_partition("train")
    oof = np.zeros(len(tab))
    for tri, vai in GroupKFold(5).split(tab, groups=tab.trip.values):
        fit = labeled(tab.iloc[tri])
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1).fit(fit[FEATURES].values, fit.y.values)
        oof[vai] = m.predict_proba(tab.iloc[vai][FEATURES].values)[:, 1]
    tab = tab.assign(p=oof)
    rng = np.random.default_rng(SEED)
    trips = sorted(rng.choice(tab.trip.unique(), size=args.n_trips, replace=False))
    print("tuning trips:", trips, flush=True)

    prepared = []
    for tid in trips:
        trip = load_trip(os.path.join(ROOT, "data", "train", tid))
        net, keep, _, tp = prepare_trip(trip, DEFAULT_CFG)
        prepared.append((tid, trip, net, trip.xy[keep], trip.track[keep, 2], tab[tab.trip == tid], tp.label_arc.values))

    rows = []
    keys = list(GRID)
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = HMMConfig(**dict(zip(keys, combo)))
        t0 = time.perf_counter()
        correct = total = 0
        clf_correct = 0
        for tid, trip, net, xy, ts, t_tab, lab in prepared:
            pred, _ = decode_table(net, xy, ts, t_tab, "p", cfg)
            ok = lab >= 0
            arcs = trip.arcs
            valid = ok & (pred >= 0)
            correct += int((arcs[pred[valid]] == arcs[lab[valid]]).all(1).sum())
            total += int(ok.sum())
            best = t_tab.loc[t_tab.groupby("pt").p.idxmax()]
            bp = np.full(len(lab), -1); bp[best.pt.values] = best.arc.values
            v2 = ok & (bp >= 0)
            clf_correct += int((arcs[bp[v2]] == arcs[lab[v2]]).all(1).sum())
        r = {**dict(zip(keys, combo)), "hmm_top1": correct / total, "clf_top1": clf_correct / total,
             "seconds": time.perf_counter() - t0}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows).sort_values("hmm_top1", ascending=False)
    df.to_csv(os.path.join(ROOT, "results", "hmm_tuning.csv"), index=False)
    best = HMMConfig(**{k: float(df.iloc[0][k]) for k in keys})
    json.dump(asdict(best), open(os.path.join(ROOT, "artifacts", "hmm_config.json"), "w"), indent=2)
    print("best", asdict(best), "top1", df.iloc[0].hmm_top1)


if __name__ == "__main__":
    main()
