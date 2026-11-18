# PathSnap – Map-matching & route reconstruction

Noisy GPS trajectories are matched to a road network. A trained classifier scores the candidate
road segments near each GPS fix, and an HMM / Viterbi decoder turns those scores into a connected route.
The model is served by a FastAPI backend (Laptop 1) and an evaluator UI calls it (Laptop 2).

## Quick start (demo)

The trained model (`artifacts/models/lgbm_tuned.joblib`) and decoder settings (`artifacts/hmm_config.json`)
are committed, so the demo needs **no training**.

**Laptop 1 – API**
1. `git clone https://github.com/sriyayc/PathSnap-Team14.git` and `cd PathSnap-Team14`
2. `pip install -r requirements.txt`
3. Put the trips to serve in `data/eval/<trip_id>/` (files `<trip_id>.track/.nodes/.arcs/.route`).
   To rehearse without evaluator trips, unzip `Student folder.zip` and copy its `student_test` folder to
   `data/student_test` (the API falls back to it when `data/eval` is empty).
4. Double-click `run_api.bat` (or run `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`).
   Note the IPv4 address it prints, and allow Python through the firewall on private networks.
5. Check it from the same laptop: http://localhost:8000/health and http://localhost:8000/docs

**Laptop 2 – UI** (only needs Python; no packages)
1. Clone the repo (or copy just the `ui/` folder).
2. Double-click `run_ui.bat` (or `cd ui` then `python -m http.server 8501`) and open http://localhost:8501
3. Enter `http://<Laptop-1-IPv4>:8000` in the API box, press **Load trips**, and click a trip.

If Laptop 2 cannot reach Laptop 1, check that both are on the same network (a phone hotspot works) and
that `http://<Laptop-1-IPv4>:8000/health` opens in Laptop 2's browser.

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
| `scripts/03_train_models.py` | Trains and compares 5 classifiers: Logistic Regression, KNN, Random Forest, LightGBM, MLP (grouped 5-fold CV + student_test) |
| `scripts/04_optuna.py` | Bayesian (Optuna TPE) tuning of LightGBM |
| `scripts/05b_tune_hmm.py` | Tunes HMM parameters on out-of-fold train predictions |
| `scripts/05_hmm_eval.py` | Compares classifier-only vs classifier+HMM |
| `scripts/07_final_eval_and_maps.py` | Runs the deployed pipeline on student_test and writes prediction maps |
| `pathsnap/pipeline.py` | End-to-end inference used by the API (reference route never loaded) |
| `pathsnap/viz.py` | Folium map helpers |
| `api/main.py` | FastAPI backend |
| `ui/index.html` | Evaluator UI (Leaflet) |
| `report/REPORT_DRAFT.md` | Report draft with all results |
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

## Full pipeline (in order)

```bash
python scripts/01_eda.py                 # EDA, data-quality report, Folium maps
python scripts/02_build_dataset.py       # candidates + features + labels -> data/processed
python scripts/03_train_models.py        # 5-model comparison (grouped CV + student_test)
python scripts/04_optuna.py              # Bayesian (TPE) tuning of LightGBM -> artifacts/models/lgbm_tuned.joblib
python scripts/05b_tune_hmm.py           # HMM decoder parameters on out-of-fold train predictions -> artifacts/hmm_config.json
python scripts/05_hmm_eval.py --model lgbm_tuned --hmm_config artifacts/hmm_config.json   # classifier vs HMM
python scripts/07_final_eval_and_maps.py # deployed pipeline on student_test + prediction maps
python scripts/06_plots.py               # all charts
```

## Deployment (two laptops on the same network)

**Laptop 1 – Model & API**
```bash
pip install -r requirements.txt
# put the evaluation trips in data/eval/<trip_id>/<trip_id>.{track,nodes,arcs,route}
#   (or set PATHSNAP_DATA_DIR to their folder; with no data/eval it serves data/student_test)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig        # note this laptop's IPv4 address, e.g. 192.168.1.20
```
Allow Python through the Windows firewall on private networks when prompted. The terminal prints
one log line per request with the trip id, accuracy and timing. Show these logs to the evaluator.

**Laptop 2 – Evaluator UI**
```bash
cd ui
python -m http.server 8501
```
Open `http://localhost:8501`, enter `http://<laptop-1-IP>:8000` in the API box and press **Load trips**.
Click a trip to run it. The UI shows the GPS points, the predicted route, the classifier-only route
(which can be toggled) and the accuracy computed by the server.

API endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. Interactive docs are at `/docs`.
The reference route is read only inside `score_against_reference()` on the server. The response contains
metrics, never reference labels or geometry.

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

## Results (student_test, 10 trips)

| | Top-1 on decoded obs (trip mean) | Top-1 on all obs (trip mean) | Route F1 | Route continuity |
|---|---|---|---|---|
| Classifier only (tuned LightGBM) | 0.920 | 0.889 | — | 0.940 |
| **Classifier + HMM/Viterbi** | **0.956** | **0.944** | **0.990** | **0.999** |

## Data partitions and leakage rules
- Train (80 trips): fitting, grouped CV and hyperparameter search.
- Student test (10 trips): development comparison only.
- Evaluator trips are never used for training, tuning or selection. Reference labels stay server-side.
