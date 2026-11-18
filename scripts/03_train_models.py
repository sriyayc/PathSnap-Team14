"""Train and compare candidate classifiers from several algorithm families.

Protocol
  * 5-fold GroupKFold over the 80 TRAIN trips (a trip never spans folds) -> CV metrics
  * refit on all train trips -> evaluate on the 10 STUDENT_TEST trips
  * evaluator trips are never touched here.
Outputs: results/model_comparison.csv, results/cv_folds.csv, results/models/*.json,
         artifacts/models/<name>.joblib, results/mlp_curve.csv
Usage: python scripts/03_train_models.py [--models logreg lgbm ...] [--folds 5]
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pathsnap.experiment import SEED, evaluate, labeled, load_arcs, load_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402

# name -> (family, factory, max training rows (None = all), config description)
MODELS = {
    "logreg": ("Linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
               None, "StandardScaler + L2 LogisticRegression C=1"),
    "knn": ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
            80_000, "StandardScaler + KNN k=25 distance-weighted, 80k-row subsample"),
    "rf": ("Bagging ensemble", lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED),
           400_000, "200 trees, min_samples_leaf=5, sqrt features, 400k-row subsample"),
    "lgbm": ("Boosting", lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                               colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1),
             None, "400 trees lr=0.05 num_leaves=31 subsample=0.8 colsample=0.8"),
    "mlp": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                                                                    learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                                                                    validation_fraction=0.1, n_iter_no_change=6, random_state=SEED)),
            400_000, "StandardScaler + MLP(64,32) relu adam early-stopping, 400k-row subsample"),
}


def subsample(df, cap, seed=SEED):
    return df if cap is None or len(df) <= cap else df.sample(cap, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    res_dir = os.path.join(ROOT, "results")
    os.makedirs(os.path.join(res_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "artifacts", "models"), exist_ok=True)

    tr_tab, tr_pts = load_partition("train")
    te_tab, te_pts = load_partition("student_test")
    tr_arcs = load_arcs("train", tr_tab.trip.unique())
    te_arcs = load_arcs("student_test", te_tab.trip.unique())
    tr_lab = labeled(tr_tab).reset_index(drop=True)
    print(f"train rows {len(tr_lab)} (pos rate {tr_lab.y.mean():.3f}) from {tr_lab.trip.nunique()} trips; "
          f"student_test rows {len(te_tab)}", flush=True)

    gkf = GroupKFold(n_splits=args.folds)
    trips = tr_tab.trip.values
    summary, fold_rows = [], []
    for name in args.models:
        family, factory, cap, cfg = MODELS[name]
        print(f"\n=== {name} ({family}) : {cfg}", flush=True)
        fold_metrics = []
        for f, (tri, vai) in enumerate(gkf.split(tr_tab, groups=trips)):
            fit_df = subsample(labeled(tr_tab.iloc[tri]), cap, SEED + f)
            val_tab = tr_tab.iloc[vai]
            val_pts = tr_pts[tr_pts.trip.isin(val_tab.trip.unique())]
            m = factory()
            t0 = time.perf_counter()
            m.fit(fit_df[FEATURES].values, fit_df.y.values)
            fit_s = time.perf_counter() - t0
            r, _ = evaluate(m, val_tab, val_pts, tr_arcs)
            r.update(model=name, fold=f, fit_s=fit_s, n_train_rows=len(fit_df))
            fold_metrics.append(r)
            fold_rows.append({k: v for k, v in r.items() if k != "confusion"})
            print(f"  fold {f}: F1m {r['f1_macro']:.4f} top1 {r['top1']:.4f} logloss {r['log_loss']:.4f} fit {fit_s:.1f}s", flush=True)

        fit_df = subsample(tr_lab, cap)
        m = factory()
        t0 = time.perf_counter()
        m.fit(fit_df[FEATURES].values, fit_df.y.values)
        fit_s = time.perf_counter() - t0
        te, _ = evaluate(m, te_tab, te_pts, te_arcs)
        joblib.dump(m, os.path.join(ROOT, "artifacts", "models", f"{name}.joblib"))

        cv = pd.DataFrame(fold_metrics)
        row = {
            "model": name, "family": family, "config": cfg,
            "cv_f1_macro": cv.f1_macro.mean(), "cv_f1_macro_std": cv.f1_macro.std(),
            "cv_top1": cv.top1.mean(), "cv_top1_std": cv.top1.std(), "cv_log_loss": cv.log_loss.mean(),
            "cv_route_continuity": cv.route_continuity.mean(),
            "test_accuracy": te["accuracy"], "test_precision_macro": te["precision_macro"],
            "test_recall_macro": te["recall_macro"], "test_f1_macro": te["f1_macro"],
            "test_top1": te["top1"], "test_log_loss": te["log_loss"],
            "test_route_continuity": te["route_continuity"], "test_cand_miss_rate": te["cand_miss_rate"],
            "fit_s_full": fit_s, "infer_s_test": te["infer_s"], "infer_us_per_row": te["infer_us_per_row"],
            "train_rows": len(fit_df),
        }
        summary.append(row)
        with open(os.path.join(res_dir, "models", f"{name}.json"), "w") as fh:
            json.dump({"summary": row, "test_confusion": te["confusion"], "cv_folds": fold_metrics}, fh, indent=2, default=float)
        if name == "mlp":
            mlp = m[-1]
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(os.path.join(res_dir, "mlp_curve.csv"), index=False)
        print(f"  TEST: F1m {te['f1_macro']:.4f} top1 {te['top1']:.4f} logloss {te['log_loss']:.4f} "
              f"continuity {te['route_continuity']:.4f} infer {te['infer_us_per_row']:.2f}us/row", flush=True)

    out = os.path.join(res_dir, "model_comparison.csv")
    new = pd.DataFrame(summary)
    if os.path.exists(out):  # merge with previous runs of other models
        old = pd.read_csv(out)
        new = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True)
    new.to_csv(out, index=False)
    folds_out = os.path.join(res_dir, "cv_folds.csv")
    fnew = pd.DataFrame(fold_rows)
    if os.path.exists(folds_out):
        fold_old = pd.read_csv(folds_out)
        fnew = pd.concat([fold_old[~fold_old.model.isin(fnew.model)], fnew], ignore_index=True)
    fnew.to_csv(folds_out, index=False)
    print("\n", new[["model", "cv_f1_macro", "cv_top1", "test_f1_macro", "test_top1", "test_log_loss",
                     "test_route_continuity", "infer_us_per_row"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
