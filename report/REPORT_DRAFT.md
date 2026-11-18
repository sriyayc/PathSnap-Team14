# PathSnap: Map-Matching and Route Reconstruction — Team 14

> DRAFT. Every number here comes from files in `results/`. Rewrite the explanations in your own words before
> submitting. The plagiarism policy checks for AI-generated text, and every member must be able to defend each claim.

## 1. Problem

GPS fixes from a moving vehicle are noisy. Each trip comes with its own road network. The task is to
recover the road segment (arc) the vehicle used at every GPS observation and a connected route. A
classifier scores candidate arcs near each observation, and an HMM/Viterbi decoder enforces road-network
connectivity. A FastAPI service on Laptop 1 serves the system, and an evaluator UI on Laptop 2 calls it.

## 2. Dataset understanding and EDA

**Partitions:** train = 80 trips (208,417 GPS observations), student_test = 10 trips (19,160), evaluator = 10
trips (not available to us). We used train for fitting, cross-validation and tuning, and student_test only for
development comparisons. Every analysis below states its partition.

**Files** (per trip, header-less, whitespace-separated): `.track` (lon, lat, t), `.nodes` (lon, lat; the row
index is the node id), `.arcs` (from, to; the row index is the arc id; directed), `.route` (reference arc ids in order).

**Findings (train + student_test, `results/eda/inventory.csv`, `data_quality.md`, `eda_overview.png`):**
- Each trip has its own local network (649 to 307,392 arcs, median 18.8k), and the trips cover 4 continents. Arc ids
  are trip-local, so arc ids cannot be predicted as classes. We score candidates instead.
- `.route` gives the arc order but no observation→arc assignment, so per-observation labels must be derived.
- Sampling: 96.5% of steps are exactly 1 s. 23 trips have gaps > 60 s (max 2,161 s). Trips 2, 71 and 97 have
  non-integer timestamps. We found no invalid coordinates or duplicate timestamps.
- GPS noise: the median GPS-to-reference distance per trip is 4.6 m (median 90th percentile 11.8 m).
  Trips 33 and 54 are offset by about 120 m, so their reference is unreliable.
- In 20 trips the GPS path is >1.5× the route length (stationary jitter or partial reference coverage).
  11 reference routes contain non-adjacent consecutive arcs.
- Network: 95% of arcs have a reverse twin (two-way roads), 60 networks contain duplicate arcs, 9 contain
  self-loops, and networks split into many weak components (median 19). There are no isolated nodes.
- Folium maps: `maps/eda_trip_00000001.html`, `eda_trip_00000033.html` (offset reference) and `eda_trip_00000007.html`.

**Preprocessing:**
1. Drop non-finite or out-of-range rows, sort by time and drop duplicate timestamps.
2. Project each trip to a local equirectangular plane (metres) centred on its mean position. The distortion is
   < 0.1% at trip scale (≤ ~50 km), and the projection works for trips in any UTM zone.
3. Thinning: drop observations < 8 m from the last kept observation (just above the median GPS noise). Train goes
   from 208k to 145k decoded observations. Dropped observations inherit the last kept prediction, so every
   original observation still receives an arc.
4. **Label derivation:** a monotone dynamic programme assigns each observation to a route position
   (`D[i,j] = c[i,j] + min_{k≤j} D[i-1,k]`, where c is the capped point-to-arc distance), so the route
   order is respected and repeated streets are assigned to the correct visit. Observations > 40 m from their
   aligned arc are unlabeled. This leaves 92.7% of train and 99.8% of student_test observations labeled.

**Candidates and features:** all directed arcs within 60 m (max 10), found with a KD-tree over arc points
sampled every 10 m. Candidate miss rate: 0.28% (train), 0.13% (student_test). There are 20 features per
(observation, candidate):
- distance, distance rank, gap and ratio to the nearest candidate
- heading difference, its cosine and its rank, plus a heading-reliability flag
- projection position along the arc, arc length, candidate count
- speed and time gaps to neighbouring observations
- distance of the previous and next observations to the same arc, and the mean/max over a ±2 window
- node in/out degree

Heading separates the two directions of a road. The neighbour and window distances add temporal context.

## 3. Model experimentation and comparison

**Task:** binary classification "is this candidate the reference arc?" (positive rate 14.6%).
**Protocol:** 5-fold GroupKFold by trip on the 80 train trips, then refit on all train trips and score on student_test once.
Seed 42. **Primary metric:** observation-level Top-1 accuracy, because the route is built from per-observation
choices. Plain accuracy is inflated by easy negatives.

| Model | Family | Key configuration | CV Macro F1 | Student-test Top-1 | Route continuity | Student-test Macro F1 | Log loss | Inference |
|---|---|---|---|---|---|---|---|---|
| LightGBM | Boosting | 400 trees, lr 0.05, 31 leaves | 0.944 ± 0.026 | **0.918** | 0.937 | **0.948** | **0.066** | 4.3 µs/row |
| Random Forest | Bagging | 200 trees, leaf 5, 400k rows | 0.944 | 0.914 | 0.935 | 0.947 | 0.068 | 6.2 µs/row |
| MLP | Neural net | (64, 32), Adam, early stopping | 0.940 | 0.909 | 0.936 | 0.942 | 0.072 | 1.0 µs/row |
| KNN | Instance-based | k = 25, distance-weighted, 80k rows | 0.928 | 0.900 | 0.928 | 0.933 | 0.115 | 86 µs/row |
| Logistic Regression | Linear | L2, C = 1, standardised | 0.921 | 0.897 | 0.923 | 0.924 | 0.108 | 0.5 µs/row |

(Fill CV standard deviations from `results/model_comparison.csv`. Plots: `results/plots/cv_macro_f1.png`,
`test_top1_f1.png`, `logloss_inference.png`, `cv_top1_box.png`, `confusion_*.png`, `mlp_training_curve.png`.)

**Observations to write up per model (section 3.2):**
- CV ≈ student-test for every model, so there is no sign of overfitting.
- All models rank the folds identically (fold 4 lowest, fold 3 highest), so the variance comes from trip difficulty, not model instability.
- The linear model underfits the distance × heading interaction.
- KNN is slow at inference and its probabilities are poor.
- Gaussian NB, a single decision tree and ExtraTrees were also tried early (see `results/train_log_batch2.txt` and the git
  history) and dropped. NB gave log loss 0.75 because its independence assumption is violated. The other two duplicated families already covered.

**Selected model: LightGBM.** It has the best Top-1, Macro F1 and log loss, is fast, and gives well-calibrated
probabilities, which the HMM needs.

## 4. Bayesian hyperparameter optimisation (LightGBM)

- **Search space:**
  - n_estimators 150–900
  - learning_rate 0.01–0.2 (log)
  - num_leaves 15–255
  - min_child_samples 10–300
  - subsample 0.5–1
  - colsample_bytree 0.4–1
  - reg_lambda and reg_alpha 1e-3–10 (log)
- **Objective:** mean 5-fold GroupKFold Top-1 on the train trips.
- **Sampler:** Optuna TPE (seed 42) with a median pruner. The untuned configuration was enqueued as trial 0.
- **Trials:** 8 (1 pruned) in a 25-minute budget, 2,505 s of search. The budget was CPU-limited because the search
  ran alongside the HMM tuning. Files: `results/optuna_trials.csv`, `optuna_best.json`, `plots/optuna_history.png`.
- **Best parameters:** n_estimators 378, learning_rate 0.048, num_leaves 119, min_child_samples 94,
  subsample 0.81, colsample_bytree 0.48, reg_lambda 0.015, reg_alpha 0.029.

| | CV Top-1 | CV Macro F1 | CV log loss | Test Top-1 | Test Macro F1 | Test log loss | Inference |
|---|---|---|---|---|---|---|---|
| Untuned | 0.9120 | 0.9444 | 0.0720 | 0.9179 | 0.9479 | 0.0657 | 6.2 µs/row |
| Tuned | **0.9137** | **0.9456** | **0.0700** | 0.9176 | 0.9473 | **0.0648** | 7.7 µs/row |

Tuning gave a small CV gain (+0.17 pt Top-1) and better log loss, and student-test Top-1 is unchanged within noise.
The tuned model was selected on CV, never on student_test. The untuned defaults were already near the optimum:
the sequence decoder, not the classifier settings, is the main lever (section 5).

## 5. HMM and Viterbi route reconstruction

- **Hidden states:** the candidate arcs (with their projection positions) of each decoded observation.
- **Emission:** `w · log p_clf`, the classifier probability renormalised over the observation's candidates, with weight w.
- **Transition (Newson & Krumm):**
  - d_route = network distance between the two projected positions, from a bounded Dijkstra over the directed graph.
  - d_gc = straight-line distance between the two GPS fixes.
  - log T = −|d_route − d_gc| / β(Δt), where β(Δt) = β + 2·(Δt − 1). β widens for time gaps.
  - Unreachable pairs get a finite penalty of −40 instead of −∞, so network breaks don't kill the path.
- **Viterbi:** δ_i(s) = max_r [δ_{i−1}(r) + log T(r→s)] + log E_i(s), with back-pointers. The best final state is traced back.
- **Gaps:** gaps > 120 s restart the chain. Observations without candidates inherit the previous arc.
  Non-adjacent consecutive predicted arcs are joined by a shortest path in the final route.
- **Decoder tuning** (train only, out-of-fold probabilities, 24 trips, grid of 27):
  - best β = 15 m, w = 0.5, backward_factor = 1 → Top-1 0.9534, against classifier-only 0.9134 (`results/hmm_tuning.csv`, `plots/hmm_tuning_grid.png`).
  - A penalty for moving backwards along a directed arc (meant to fix wrong-direction twins) did **not** help on train (best factor = 1), so it was not used.
- A second-stage grid that allowed 5–10 m of backward jitter for free also did not beat factor 1
  (`results/hmm_tuning_stage2.csv`).
- **Classifier-only vs classifier + HMM:** final pipeline (tuned LightGBM + tuned HMM) on student_test,
  `results/final_student_test.csv`, `plots/final_student_test.png`:

| Trip | Obs | Classifier Top-1 (all obs) | HMM Top-1 (all obs) | HMM Top-1 (decoded obs) | Route F1 | Continuity |
|---|---|---|---|---|---|---|
| 0 | 2503 | 0.947 | 0.960 | 0.976 | 0.995 | 0.990 |
| 7 | 1354 | 0.920 | 0.995 | 0.996 | 0.995 | 1.000 |
| 9 | 1049 | 0.860 | 0.922 | 0.933 | 0.980 | 1.000 |
| 40 | 2995 | 0.936 | 0.955 | 0.971 | 1.000 | 1.000 |
| 52 | 262 | 0.958 | 1.000 | 1.000 | 0.998 | 1.000 |
| 56 | 3958 | 0.903 | 0.971 | 0.978 | 0.984 | 0.999 |
| 65 | 768 | 0.866 | 0.928 | 0.923 | 0.995 | 1.000 |
| 76 | 1486 | 0.892 | 0.967 | 0.972 | 0.970 | 1.000 |
| 77 | 3911 | 0.741 | 0.811 | 0.844 | 0.992 | 1.000 |
| 91 | 874 | 0.866 | 0.933 | 0.963 | 0.987 | 1.000 |
| **Mean** | | **0.889** | **0.944** | **0.956** | **0.990** | **0.999** |

  Point-weighted Top-1 on all observations: classifier 0.876 → HMM 0.930. The classifier-only route has
  continuity 0.94, which rises to 0.999 with the HMM. "Decoded obs" are the 8 m-thinned observations the
  Viterbi decodes. Removed observations are assigned the nearest arc on the decoded path between their kept neighbours.
- **Where the HMM did not help:** the HMM improved every trip, but trip 77 stays at 0.81. Most of its errors
  (251 of 343) pick the opposite-direction twin of the reference arc. This is a slow (~2.5 m/s) out-and-back trip,
  and 29% of the twin arcs we predicted also appear in its reference route. In trip 9's twin errors, the GPS
  heading is 13° from our predicted direction and 167° from the reference direction, which points to label
  ambiguity in the derived alignment rather than a decoding mistake. During development (untuned settings) the HMM
  briefly lowered trip 7 (0.976 → 0.944). The tuned settings (β = 15, w = 0.5) fixed this (0.996).

## 6. Folium visualisation

`maps/pred_trip_<id>.html` (all 10 student-test trips) has these layers: raw GPS (blue points), nearby road network
(grey), reference route (green), classifier-only route (orange dashed, toggle) and HMM route (red), plus a legend,
layer control and start/end markers. Code: `pathsnap/viz.py` and `scripts/07_final_eval_and_maps.py`.

## 7. API, UI and deployment

- FastAPI (`api/main.py`) endpoints:
  - `GET /health`
  - `GET /trips`
  - `POST /predict {trip_id}`
- The pipeline loads trips with `include_route=False`. The reference is read only in `score_against_reference()`
  and only metrics are returned. Each request is logged with trip id, accuracy and timing.
- The UI (`ui/index.html`, Leaflet) lists the trips, posts the selected id, draws GPS + predicted + classifier-only
  routes, and shows Top-1, route F1/precision/recall/continuity and timing.
- Screenshots: <<add UI screenshot and Laptop-1 terminal log screenshot>>.

## 8. Limitations

- Labels are derived (DP alignment) from a reference that is itself imperfect (offset trips, broken routes).
- Top-1 counts an arc as wrong even when the observation sits at the junction between two consecutive arcs.
- Most remaining errors are wrong-direction twins at low speed.

## 9. AI-use disclosure

<<Describe how AI assistance was used, per the event policy.>>
