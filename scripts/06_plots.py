"""Model-comparison plots from results/model_comparison.csv, cv_folds.csv, mlp_curve.csv,
optuna trials and HMM results. Every number comes from saved experiment outputs.
Usage: python scripts/06_plots.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "plots")
os.makedirs(OUT, exist_ok=True)


def bar(ax, names, vals, title, ylabel, err=None, lim=None):
    ax.bar(names, vals, yerr=err, capsize=4, color="#4C78A8")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=30)
    if lim: ax.set_ylim(*lim)


comp = pd.read_csv(os.path.join(RES, "model_comparison.csv")).sort_values("test_top1", ascending=False)
names = comp.model.tolist()

fig, ax = plt.subplots(figsize=(9, 4))
bar(ax, names, comp.cv_f1_macro.values, "5-fold grouped CV Macro F1 (train trips)", "Macro F1",
    err=comp.cv_f1_macro_std.values, lim=(0.7, 1.0))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cv_macro_f1.png"), dpi=130)

fig, axs = plt.subplots(1, 2, figsize=(13, 4))
bar(axs[0], names, comp.test_top1.values, "Student-test observation Top-1 accuracy", "Top-1", lim=(0.7, 1.0))
bar(axs[1], names, comp.test_f1_macro.values, "Student-test Macro F1", "Macro F1", lim=(0.7, 1.0))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "test_top1_f1.png"), dpi=130)

fig, axs = plt.subplots(1, 2, figsize=(13, 4))
bar(axs[0], names, comp.test_log_loss.values, "Student-test log loss (lower is better)", "log loss")
bar(axs[1], names, comp.infer_us_per_row.values, "Inference time per candidate row", "microseconds")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "logloss_inference.png"), dpi=130)

folds = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot([folds[folds.model == m].top1 for m in names], tick_labels=names)
ax.set_title("Top-1 accuracy across CV folds (stability)"); ax.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cv_top1_box.png"), dpi=130)

for f in glob.glob(os.path.join(RES, "models", "*.json")):
    d = json.load(open(f))
    cm = np.array(d["test_confusion"])
    name = os.path.basename(f)[:-5]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"{name}: student-test confusion")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"confusion_{name}.png"), dpi=120); plt.close(fig)

if os.path.exists(os.path.join(RES, "mlp_curve.csv")):
    c = pd.read_csv(os.path.join(RES, "mlp_curve.csv"))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(c.epoch, c.train_loss, "b-o", ms=3, label="training loss (log loss)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(c.epoch, c.val_accuracy, "r-s", ms=3, label="held-out validation accuracy")
    ax2.set_ylabel("validation accuracy", color="r")
    ax1.set_title("MLP training history (early stopping)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mlp_training_curve.png"), dpi=130)

trials = os.path.join(RES, "optuna_trials.csv")
if os.path.exists(trials):
    t = pd.read_csv(trials)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t.number, t.value, "o", ms=4, alpha=0.6, label="trial")
    ax.plot(t.number, t.value.cummax(), "r-", label="best so far")
    ax.set(title="Bayesian optimisation (TPE): CV objective per trial", xlabel="trial", ylabel="CV objective")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "optuna_history.png"), dpi=130)

for f in glob.glob(os.path.join(RES, "hmm_*_student_test.csv")):
    h = pd.read_csv(f)
    tag = os.path.basename(f)[4:-4]
    x = np.arange(len(h)); w = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].bar(x - w / 2, h.clf_top1, w, label="classifier only")
    axs[0].bar(x + w / 2, h.hmm_top1, w, label="classifier + HMM")
    axs[0].set_xticks(x, h.trip.astype(str).str[-2:]); axs[0].set_ylim(0.5, 1.0)
    axs[0].set(title=f"Top-1 per student-test trip ({tag})", xlabel="trip"); axs[0].legend()
    axs[1].bar(x - w / 2, h.clf_continuity, w, label="classifier only")
    axs[1].bar(x + w / 2, h.hmm_continuity, w, label="classifier + HMM")
    axs[1].set_xticks(x, h.trip.astype(str).str[-2:]); axs[1].set_ylim(0.5, 1.0)
    axs[1].set(title="Route continuity per trip", xlabel="trip"); axs[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"hmm_vs_clf_{tag}.png"), dpi=130)
print("plots written to", OUT)
