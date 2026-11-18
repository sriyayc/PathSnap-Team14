# PathSnap – Map-matching & route reconstruction

Noisy GPS trajectories are matched to a road network. A trained classifier scores the candidate
road segments near each GPS fix, and an HMM / Viterbi decoder turns those scores into a connected route.
The model is served by a FastAPI backend (Laptop 1) and an evaluator UI calls it (Laptop 2).

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/data.py` | Trip loading, cleaning, local metric projection, track thinning |
| `pathsnap/network.py` | Road-network geometry, KD-tree candidate search, bounded Dijkstra |
| `pathsnap/labels.py` | Derives per-observation labels from the arc-sequence route (monotone DP) |
| `pathsnap/features.py` | Candidate generation and feature engineering (one row per GPS-fix × candidate arc) |
| `pathsnap/metrics.py` | Macro-F1, log loss, Top-1, candidate miss rate, route continuity, route-set F1 |
| `pathsnap/hmm.py` | HMM / Viterbi decoder |
| `scripts/01_eda.py` | Dataset inventory, data-quality report, EDA plots, Folium maps |
| `scripts/02_build_dataset.py` | Builds the candidate/feature tables for the train and student_test partitions |
| `scripts/03_train_models.py` | Trains and compares 8 classifiers (grouped 5-fold CV + student_test) |
| `scripts/05_hmm_eval.py` | Compares classifier-only vs classifier+HMM |
| `scripts/06_plots.py` | Comparison plots from the saved results |
| `results/` | CSV/JSON experiment outputs, logs, plots |
| `maps/` | Folium HTML maps |
| `artifacts/models/` | Trained model files (joblib) |

## Setup

```bash
pip install -r requirements.txt
```
Unzip the provided `Student folder.zip` so that `data/train/<trip>/` and `data/student_test/<trip>/` exist.
Raw data is not committed.

## Reproduce

```bash
python scripts/01_eda.py              # EDA -> results/eda, maps/
python scripts/02_build_dataset.py    # candidate + feature tables -> data/processed
python scripts/03_train_models.py     # model comparison -> results/model_comparison.csv
python scripts/06_plots.py            # plots -> results/plots
python scripts/05_hmm_eval.py --model lgbm --part student_test
```
All random seeds are fixed (`SEED = 42`).

## Method summary

1. **Projection.** Each trip is projected to a local equirectangular plane centred on its mean GPS
   position, with units in metres. Trips span at most ~50 km, so the distortion is far below GPS noise.
2. **Cleaning and thinning.**
   - Non-finite or out-of-range rows are dropped.
   - Rows are sorted by time and duplicate timestamps are removed.
   - Fixes closer than 8 m to the previous kept fix are removed, because stationary jitter and 1 Hz redundancy add no information.
   - Removed fixes inherit the prediction of the last kept fix.
3. **Labels.** The `.route` file is an ordered arc list with no per-fix assignment. Each fix is aligned
   to a route position with a monotone dynamic programme, so the position can only move forward along the route.
   Fixes more than 40 m from their aligned arc are marked unlabeled. These lie outside the reference
   coverage or come from an offset reference (e.g. trip 33).
4. **Candidates.** All directed arcs within 60 m of a fix (at most 10, nearest first), found with a KD-tree over
   densified arc geometry. Both directions of a two-way road are kept as candidates. Heading tells them apart.
5. **Features** (20 per candidate):
   - distance, distance rank, and gap/ratio to the nearest candidate
   - heading difference, its cosine and its rank, plus a heading-reliability flag
   - projection position along the arc, arc length, and candidate count
   - speed and time gaps to the neighbouring fixes
   - distance from the previous and next fixes to the same arc, and a ±2-fix window mean/max distance
   - node degrees
6. **Classifier.** Binary "is this candidate the reference arc?". Experiments use grouped CV by trip.
7. **HMM.**
   - Hidden states are the candidate arcs.
   - Emissions are the classifier probabilities.
   - Transitions follow Newson–Krumm: |network distance − great-circle distance| / β, with β widened for time gaps.
   - A gap of more than 120 s restarts the chain.

## Data partitions and leakage rules
- Train (80 trips): fitting, grouped CV and hyperparameter search.
- Student test (10 trips): development comparison only.
- Evaluator trips are never used for training, tuning or selection. Reference labels stay server-side.
