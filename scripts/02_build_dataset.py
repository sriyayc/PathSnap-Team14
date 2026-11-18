"""Build candidate/feature tables for the train and student_test partitions.

Usage: python scripts/02_build_dataset.py [--only TRIPID ...]
Outputs: data/processed/{partition}_cands.pkl, {partition}_points.pkl
"""
import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.features import DEFAULT_CFG, prepare_trip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = os.path.join(ROOT, "data", "processed")
    os.makedirs(out, exist_ok=True)
    for part in ["train", "student_test"]:
        pdir = os.path.join(ROOT, "data", part)
        tabs, pts = [], []
        for tid in list_trips(pdir):
            if args.only and tid not in args.only:
                continue
            t0 = time.time()
            trip = load_trip(os.path.join(pdir, tid))
            _, _, tab, points = prepare_trip(trip, DEFAULT_CFG)
            tabs.append(tab)
            pts.append(points)
            lab = points.label_arc >= 0
            miss = (~points.ref_in_cands[lab]).mean() if lab.any() else float("nan")
            print(f"{part} {tid}: kept {len(points)}/{len(trip.track)} pts, labeled {lab.mean():.2f}, "
                  f"rows {len(tab)}, cand-miss {miss:.3f}, {time.time() - t0:.1f}s", flush=True)
        if args.only:
            continue
        pd.concat(tabs, ignore_index=True).to_pickle(os.path.join(out, f"{part}_cands.pkl"))
        pd.concat(pts, ignore_index=True).to_pickle(os.path.join(out, f"{part}_points.pkl"))


if __name__ == "__main__":
    main()
