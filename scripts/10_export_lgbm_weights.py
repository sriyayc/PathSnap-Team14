"""Export the trained LightGBM models to LightGBM's native text format.

The .joblib files in artifacts/models/ are scikit-learn pickles, which need matching
Python/sklearn versions to load. The native .txt model holds every tree (split features,
thresholds, leaf values) and can be loaded by any LightGBM binding (Python, R, C++, CLI)
via lgb.Booster(model_file=...).

Outputs   : artifacts/weights/<name>.txt, artifacts/weights/<name>_meta.json
Usage     : python scripts/10_export_lgbm_weights.py [--models lgbm_tuned lgbm]
"""
import argparse
import json
import os

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgbm_tuned"])
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "artifacts", "weights")
    os.makedirs(out_dir, exist_ok=True)

    for name in args.models:
        clf = joblib.load(os.path.join(ROOT, "artifacts", "models", f"{name}.joblib"))
        booster = clf.booster_
        txt_path = os.path.join(out_dir, f"{name}.txt")
        booster.save_model(txt_path)

        classes = clf.classes_.tolist() if isinstance(clf.classes_, np.ndarray) else list(clf.classes_)
        meta = {
            "model": name,
            "lightgbm_params": {k: v for k, v in clf.get_params().items() if not callable(v)},
            "classes": classes,
            "n_classes": len(classes),
            "n_features": booster.num_feature(),
            "feature_names": booster.feature_name(),
            "num_trees": booster.num_trees(),
            "best_iteration": booster.best_iteration,
            "load_example": f"lgb.Booster(model_file='artifacts/weights/{name}.txt')",
            "note": ("Binary: Booster.predict returns P(classes[1]). "
                     "Multiclass: per-class probabilities in the order of 'classes'."),
        }
        with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Round-trip check: reloaded booster must reproduce the sklearn model's probabilities.
        import lightgbm as lgb
        reloaded = lgb.Booster(model_file=txt_path)
        x = np.random.default_rng(0).normal(size=(64, booster.num_feature()))
        ref = clf.predict_proba(x)
        ref = ref[:, 1] if ref.shape[1] == 2 else ref
        diff = np.abs(reloaded.predict(x) - ref).max()
        print(f"{name}: {booster.num_trees()} trees, {booster.num_feature()} features, "
              f"{len(classes)} classes -> {txt_path} "
              f"({os.path.getsize(txt_path) / 1e6:.1f} MB), max |diff| on reload = {diff:.2e}")
        assert diff < 1e-9, f"round-trip mismatch for {name}"


if __name__ == "__main__":
    main()
