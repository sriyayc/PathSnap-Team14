"""Run the deployed pipeline (same code as the API) on every student-test trip:
  * results/final_student_test.csv          per-trip classifier-only vs HMM accuracy, route metrics, timing
  * results/final_student_test_summary.json overall numbers
  * maps/pred_trip_<id>.html                Folium: GPS, nearby network, reference, classifier-only, HMM route
Usage: python scripts/07_final_eval_and_maps.py [--part student_test] [--model lgbm_tuned]
"""
import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap import viz  # noqa: E402
from pathsnap.data import list_trips, load_route  # noqa: E402
from pathsnap.pipeline import Predictor, score_against_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="student_test")
    ap.add_argument("--model", default="lgbm_tuned")
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()
    P = Predictor(args.model)
    print("model", P.model_name, "hmm", P.hmm_cfg, flush=True)
    pdir = os.path.join(ROOT, "data", args.part)
    rows = []
    for tid in list_trips(pdir):
        folder = os.path.join(pdir, tid)
        trip, res = P.predict_trip(folder)
        m = score_against_reference(trip, folder, res)
        a, d, r = m["all_observations"], m["decoded_observations"], m["route"]
        rows.append({"trip": tid, "n_obs": res["n_observations"], "labeled_obs": a["labeled"],
                     "clf_top1_all": a["classifier_top1"], "hmm_top1_all": a["hmm_top1"],
                     "clf_top1_decoded": d["classifier_top1"], "hmm_top1_decoded": d["hmm_top1"],
                     "route_precision": r["precision"], "route_recall": r["recall"], "route_f1": r["f1"],
                     "route_continuity": r["continuity"], "time_s": res["timing_s"]["total"]})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        if not args.no_maps:
            ref = load_route(folder, tid)
            mp = viz.base_map(trip)
            viz.add_network(mp, trip)
            viz.add_gps(mp, trip)
            viz.add_route(mp, trip, ref, "Reference route", "#1a9850", weight=8)
            clf_seq = [int(x) for x in res["clf_arcs_kept"] if x >= 0]
            viz.add_route(mp, trip, list(dict.fromkeys(clf_seq)), "Classifier-only prediction", "#f39c12", weight=3, dash="4 6", show=False)
            viz.add_route(mp, trip, res["route_arcs"], "Predicted route (classifier + HMM)", "#d7301f", weight=4)
            viz.add_start_end(mp, trip)
            viz.add_legend(mp, f"Trip {tid}: Top-1 {a['hmm_top1']:.1%} (HMM) vs {a['classifier_top1']:.1%} (classifier)",
                           [("Raw GPS observations", "#1f4fd1"), ("Reference route", "#1a9850"),
                            ("Predicted route (HMM)", "#d7301f"), ("Classifier-only (toggle)", "#f39c12"),
                            ("Road network (nearby)", "#8c8c8c")])
            viz.finish(mp, os.path.join(ROOT, "maps", f"pred_trip_{tid}.html"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", f"final_{args.part}.csv"), index=False)
    w = df.labeled_obs / df.labeled_obs.sum()
    summary = {"model": P.model_name, "partition": args.part,
               "clf_top1_all_pointweighted": float((df.clf_top1_all * w).sum()),
               "hmm_top1_all_pointweighted": float((df.hmm_top1_all * w).sum()),
               "clf_top1_all_trip_mean": float(df.clf_top1_all.mean()), "hmm_top1_all_trip_mean": float(df.hmm_top1_all.mean()),
               "hmm_top1_decoded_trip_mean": float(df.hmm_top1_decoded.mean()),
               "route_f1_mean": float(df.route_f1.mean()), "route_continuity_mean": float(df.route_continuity.mean()),
               "mean_time_s": float(df.time_s.mean())}
    json.dump(summary, open(os.path.join(ROOT, "results", f"final_{args.part}_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
