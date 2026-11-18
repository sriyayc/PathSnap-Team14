"""Compare classifier-only vs classifier + HMM/Viterbi on a partition.

Usage: python scripts/05_hmm_eval.py --model lgbm_tuned --part student_test [--beta 8 --beta_per_s 2]
Outputs: results/hmm_<model>_<part>.csv (per trip) and printed summary.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.features import DEFAULT_CFG, FEATURES, prepare_trip  # noqa: E402
from pathsnap.hmm import HMMConfig, decode_table  # noqa: E402
from pathsnap.metrics import argmax_predictions, route_continuity, route_set_metrics, top1_accuracy  # noqa: E402


def run_trip(model, trip, hcfg):
    net, keep, tab, pts = prepare_trip(trip, DEFAULT_CFG)
    tab = tab.assign(p=model.predict_proba(tab[FEATURES].values)[:, 1])
    xy, ts = trip.xy[keep], trip.track[keep, 2]
    t0 = time.perf_counter()
    hmm_pred, diag = decode_table(net, xy, ts, tab, "p", hcfg)
    dec_s = time.perf_counter() - t0
    clf = argmax_predictions(tab)
    hmm = pd.DataFrame({"trip": trip.trip_id, "pt": np.arange(len(xy)), "pred_arc": hmm_pred})
    hmm = hmm[hmm.pred_arc >= 0]
    arcs = {trip.trip_id: trip.arcs}
    L = {trip.trip_id: net.length}
    routes = {trip.trip_id: trip.route}
    r = {"trip": trip.trip_id, "n_obs": len(xy), "labeled": int((pts.label_arc >= 0).sum()), "decode_s": dec_s, **diag}
    for name, pr in [("clf", clf), ("hmm", hmm)]:
        r[f"{name}_top1"] = top1_accuracy(pts, pr, arcs)
        r[f"{name}_continuity"] = route_continuity(pts, pr, arcs)
        for k, v in route_set_metrics(pts, pr, routes, arcs, L).items():
            r[f"{name}_{k}"] = v
    return r, pts, clf, hmm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--part", default="student_test")
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--hmm_config", default=None, help="json file with HMMConfig fields")
    args = ap.parse_args()
    hcfg = HMMConfig()
    if args.hmm_config:
        hcfg = HMMConfig(**json.load(open(args.hmm_config)))
    model = joblib.load(os.path.join(ROOT, "artifacts", "models", f"{args.model}.joblib"))
    pdir = os.path.join(ROOT, "data", args.part)
    rows = []
    for tid in args.trips or list_trips(pdir):
        r, *_ = run_trip(model, load_trip(os.path.join(pdir, tid)), hcfg)
        rows.append(r)
        print(f"{tid}: clf top1 {r['clf_top1']:.3f} cont {r['clf_continuity']:.3f} | "
              f"hmm top1 {r['hmm_top1']:.3f} cont {r['hmm_continuity']:.3f} routeF1 {r['hmm_route_f1']:.3f} "
              f"({r['decode_s']:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", f"hmm_{args.model}_{args.part}.csv")
    df.to_csv(out, index=False)
    w = df.labeled / df.labeled.sum()
    summ = {"model": args.model, "partition": args.part, "hmm_config": asdict(hcfg)}
    for c in [c for c in df.columns if c.startswith(("clf_", "hmm_"))]:
        summ[c + "_mean"] = float(df[c].mean())
    summ["clf_top1_pointweighted"] = float((df.clf_top1 * w).sum())
    summ["hmm_top1_pointweighted"] = float((df.hmm_top1 * w).sum())
    json.dump(summ, open(out.replace(".csv", "_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
