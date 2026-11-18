"""Close-up Folium map of the stretch of a student-test trip where classifier-only and HMM predictions
disagree most, showing why sequence decoding matters. Output: maps/zoom_trip_<id>.html
Usage: python scripts/09_zoom_map.py 00000076
"""
import os
import sys

import folium
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap import viz  # noqa: E402
from pathsnap.data import load_route  # noqa: E402
from pathsnap.pipeline import Predictor  # noqa: E402

tid = sys.argv[1] if len(sys.argv) > 1 else "00000076"
folder = os.path.join(ROOT, "data", "student_test", tid)
P = Predictor("lgbm_tuned")
trip, res = P.predict_trip(folder)
keep, h, c = res["kept_idx"], res["pred_arcs_kept"], res["clf_arcs_kept"]
A = trip.arcs
# -1 = no prediction; compare only where both exist (A[-1] would silently pick the last arc)
valid = (h >= 0) & (c >= 0)
diff = valid & ~(A[np.maximum(h, 0)] == A[np.maximum(c, 0)]).all(1)
win = 60
score = np.convolve(diff.astype(float), np.ones(win), "same")
centre = int(np.argmax(score))
lo, hi = max(0, centre - win // 2), min(len(keep), centre + win // 2)
idx = keep[lo:hi]
lat, lon = trip.track[idx, 1], trip.track[idx, 0]
m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=17, tiles=None)
folium.TileLayer(viz.ESRI, attr="Tiles &copy; Esri", name="Esri streets", max_zoom=19).add_to(m)
ref = load_route(folder, tid)
viz.add_network(m, trip, within=60)
viz.add_route(m, trip, ref, "Reference route", "#1a9850", weight=9)
clf_seq = [int(x) for x in c[lo:hi] if x >= 0]
viz.add_route(m, trip, list(dict.fromkeys(clf_seq)), "Classifier-only prediction", "#f39c12", weight=4, dash="5 6")
viz.add_route(m, trip, [int(x) for x in dict.fromkeys(h[lo:hi].tolist())], "HMM prediction", "#d7301f", weight=4)
fg = folium.FeatureGroup("Raw GPS observations (this stretch)")
for i in idx:
    folium.CircleMarker([trip.track[i, 1], trip.track[i, 0]], radius=3, color="#1f4fd1", fill=True, fill_opacity=0.9).add_to(fg)
fg.add_to(m)
m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]], padding=(40, 40))
viz.add_legend(m, f"Trip {tid} close-up: {int(diff[lo:hi].sum())} of {hi-lo} fixes differ",
               [("Raw GPS", "#1f4fd1"), ("Reference route", "#1a9850"), ("Classifier-only", "#f39c12"),
                ("Classifier + HMM", "#d7301f"), ("Road network", "#8c8c8c")])
viz.finish(m, os.path.join(ROOT, "maps", f"zoom_trip_{tid}.html"))
print("saved", tid, "disagreements in window:", int(diff[lo:hi].sum()), "of", hi - lo)
