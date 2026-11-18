"""PathSnap FastAPI backend (Laptop 1).

Run:   uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:   PATHSNAP_DATA_DIR   folder containing one sub-folder per evaluation trip (default: data/eval;
                           falls back to data/student_test if data/eval is empty)
       PATHSNAP_MODEL      model artifact name (default: lgbm_tuned)

Endpoints
  GET  /health            liveness + loaded model/config
  GET  /trips             trip ids available for evaluation
  POST /predict {trip_id} runs the full pipeline (features -> classifier -> HMM/Viterbi) and returns the
                          predicted route geometry plus server-side accuracy. The reference route is only
                          read inside score_against_reference() and is never returned.
"""
import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.pipeline import Predictor, score_against_reference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("pathsnap.api")


def _data_dir():
    d = os.environ.get("PATHSNAP_DATA_DIR", os.path.join(ROOT, "data", "eval"))
    if not os.path.isdir(d) or not any(os.path.isdir(os.path.join(d, x)) for x in os.listdir(d)):
        fallback = os.path.join(ROOT, "data", "student_test")
        log.warning("No trips in %s - falling back to %s", d, fallback)
        d = fallback
    return d


DATA_DIR = _data_dir()
predictor = Predictor(os.environ.get("PATHSNAP_MODEL", "lgbm_tuned"))
log.info("Loaded model=%s  hmm=%s  data_dir=%s", predictor.model_name, predictor.hmm_cfg, DATA_DIR)

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def list_trips():
    return sorted(x for x in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, x)))


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.model_name, "data_dir": DATA_DIR, "n_trips": len(list_trips())}


@app.get("/trips")
def trips():
    ids = list_trips()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


def _latlon(trip, node):
    return [float(trip.nodes[node, 1]), float(trip.nodes[node, 0])]


def _route_lines(trip, arcs):
    lines, cur, prev = [], [], None
    for a in arcs:
        u, v = trip.arcs[a]
        if prev is None or u != prev:
            if len(cur) > 1:
                lines.append(cur)
            cur = [_latlon(trip, u)]
        cur.append(_latlon(trip, v))
        prev = v
    if len(cur) > 1:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in list_trips():
        log.warning("POST /predict unknown trip_id=%s", tid)
        raise HTTPException(404, f"unknown trip_id {tid}")
    t0 = time.perf_counter()
    folder = os.path.join(DATA_DIR, tid)
    log.info("POST /predict trip_id=%s  -> running pipeline", tid)
    trip, res = predictor.predict_trip(folder)
    metrics = score_against_reference(trip, folder, res)  # server-side only
    elapsed = time.perf_counter() - t0
    if metrics:
        log.info("trip_id=%s  obs=%d  decoded=%d  Top-1(HMM)=%.4f  Top-1(classifier)=%.4f  route-F1=%.4f  "
                 "continuity=%.3f  time=%.2fs", tid, res["n_observations"], res["n_decoded"],
                 metrics["all_observations"]["hmm_top1"], metrics["all_observations"]["classifier_top1"],
                 metrics["route"]["f1"], metrics["route"]["continuity"], elapsed)
    else:
        log.info("trip_id=%s  obs=%d  (no reference available)  time=%.2fs", tid, res["n_observations"], elapsed)
    step = max(1, len(trip.track) // 2500)
    clf_seq = [int(a) for a in res["clf_arcs_kept"] if a >= 0]
    return {
        "trip_id": tid,
        "n_observations": res["n_observations"],
        "n_decoded": res["n_decoded"],
        "gps": trip.track[::step, [1, 0]].round(6).tolist(),
        "predicted_route": _route_lines(trip, res["route_arcs"]),
        "classifier_only_route": _route_lines(trip, list(dict.fromkeys(clf_seq))),
        "predicted_arcs_per_observation": res["pred_arcs"].astype(int).tolist(),
        "metrics": metrics,
        "timing_s": {k: round(v, 3) for k, v in res["timing_s"].items()} | {"request_total": round(elapsed, 3)},
        "model": res["model"],
        "hmm_config": res["hmm_config"],
        "hmm_diagnostics": res["hmm_diagnostics"],
    }
