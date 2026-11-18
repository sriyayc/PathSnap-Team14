"""In-sample (training-set) metrics of each saved model, to compare with grouped-CV and student-test
results when judging over/underfitting.
Output: results/train_fit.csv
"""
import os
import sys

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.experiment import evaluate, load_arcs, load_partition  # noqa: E402

tab, pts = load_partition("train")
arcs = load_arcs("train", tab.trip.unique())
rows = []
for name in ["logreg", "knn", "rf", "lgbm", "mlp", "lgbm_tuned"]:
    m = joblib.load(os.path.join(ROOT, "artifacts", "models", f"{name}.joblib"))
    r, _ = evaluate(m, tab, pts, arcs)
    rows.append({"model": name, "train_f1_macro": r["f1_macro"], "train_top1": r["top1"], "train_log_loss": r["log_loss"]})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "train_fit.csv"), index=False)
