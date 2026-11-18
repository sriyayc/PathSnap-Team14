"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected LightGBM classifier.

Objective : mean observation-level Top-1 accuracy over 5-fold GroupKFold on the 80 TRAIN trips
            (group = trip). Student-test is used only once, after the search, to compare the
            tuned and untuned models. Evaluator trips are never used.
Outputs   : results/optuna_trials.csv, results/optuna_best.json, artifacts/models/lgbm_tuned.joblib,
            results/tuned_vs_untuned.csv
Usage     : python scripts/04_optuna.py [--trials 25] [--timeout 1800]
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.experiment import SEED, evaluate, labeled, load_arcs, load_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402

SEARCH_SPACE = {
    "n_estimators": "int [150, 900]",
    "learning_rate": "log-uniform [0.01, 0.2]",
    "num_leaves": "int [15, 255]",
    "min_child_samples": "int [10, 300]",
    "subsample": "uniform [0.5, 1.0]",
    "colsample_bytree": "uniform [0.4, 1.0]",
    "reg_lambda": "log-uniform [1e-3, 10]",
    "reg_alpha": "log-uniform [1e-3, 10]",
}


def make_model(p):
    return LGBMClassifier(**p, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    tr_tab, tr_pts = load_partition("train")
    te_tab, te_pts = load_partition("student_test")
    tr_arcs = load_arcs("train", tr_tab.trip.unique())
    te_arcs = load_arcs("student_test", te_tab.trip.unique())
    folds = list(GroupKFold(n_splits=5).split(tr_tab, groups=tr_tab.trip.values))

    def cv_score(params, trial=None):
        tops, f1s, lls = [], [], []
        for f, (tri, vai) in enumerate(folds):
            fit_df = labeled(tr_tab.iloc[tri])
            m = make_model(params).fit(fit_df[FEATURES].values, fit_df.y.values)
            val = tr_tab.iloc[vai]
            r, _ = evaluate(m, val, tr_pts[tr_pts.trip.isin(val.trip.unique())], tr_arcs)
            tops.append(r["top1"]); f1s.append(r["f1_macro"]); lls.append(r["log_loss"])
            if trial is not None:
                trial.report(float(np.mean(tops)), f)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(tops)), float(np.mean(f1s)), float(np.mean(lls))

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
        t0 = time.perf_counter()
        top1, f1, ll = cv_score(p, trial)
        trial.set_user_attr("cv_f1_macro", f1)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number}: top1 {top1:.4f} f1 {f1:.4f} ll {ll:.4f} {p}", flush=True)
        return top1

    untuned = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
                                study_name="lgbm_top1")
    study.enqueue_trial({**untuned, "min_child_samples": 20, "reg_lambda": 1e-3, "reg_alpha": 1e-3})  # baseline as trial 0
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_s = time.perf_counter() - t0

    res = os.path.join(ROOT, "results")
    study.trials_dataframe().to_csv(os.path.join(res, "optuna_trials.csv"), index=False)
    best = study.best_params
    print("best", study.best_value, best, flush=True)

    rows = []
    for name, p in [("lgbm_untuned", untuned), ("lgbm_tuned", best)]:
        top1, f1, ll = cv_score(p)
        fit_df = labeled(tr_tab)
        t1 = time.perf_counter()
        m = make_model(p).fit(fit_df[FEATURES].values, fit_df.y.values)
        fit_s = time.perf_counter() - t1
        te, _ = evaluate(m, te_tab, te_pts, te_arcs)
        joblib.dump(m, os.path.join(ROOT, "artifacts", "models", f"{name}.joblib"))
        rows.append({"model": name, "cv_top1": top1, "cv_f1_macro": f1, "cv_log_loss": ll,
                     "test_top1": te["top1"], "test_f1_macro": te["f1_macro"], "test_log_loss": te["log_loss"],
                     "test_route_continuity": te["route_continuity"], "fit_s": fit_s,
                     "infer_us_per_row": te["infer_us_per_row"], "params": json.dumps(p)})
    pd.DataFrame(rows).to_csv(os.path.join(res, "tuned_vs_untuned.csv"), index=False)
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": len(study.trials),
               "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
               "search_seconds": search_s, "search_space": SEARCH_SPACE, "objective": "mean 5-fold GroupKFold Top-1 (train trips)",
               "sampler": "TPE seed 42", "pruner": "MedianPruner"},
              open(os.path.join(res, "optuna_best.json"), "w"), indent=2)
    print(pd.DataFrame(rows).drop(columns="params").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
