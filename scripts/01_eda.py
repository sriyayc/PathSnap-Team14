"""Exploratory data analysis for the train and student_test partitions.

Outputs (results/eda/):
  inventory.csv          per-trip counts, sampling, gaps, network size/connectivity, route checks
  data_quality.md        anomalies found
  *.png                  distributions (sampling interval, gaps, GPS-to-route distance, speeds, network size)
maps/eda_trip_<id>.html  Folium map: GPS points, nearby road network, reference route
Usage: python scripts/01_eda.py [--map TRIPID]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.network import point_segment  # noqa: E402

OUT = os.path.join(ROOT, "results", "eda")


def trip_stats(part, trip):
    tr, xy = trip.track, trip.xy
    dt = np.diff(tr[:, 2])
    step = np.hypot(*np.diff(xy, axis=0).T)
    speed = step / np.where(dt > 0, dt, np.nan)
    N, A = len(trip.nodes), len(trip.arcs)
    arcs = trip.arcs
    g = coo_matrix((np.ones(A), (arcs[:, 0], arcs[:, 1])), shape=(N, N))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs.ravel(), minlength=N)
    pairs = set(map(tuple, arcs))
    rev = sum((b, a) in pairs for a, b in arcs)
    r = dict(partition=part, trip=trip.trip_id, n_points=len(tr), **trip.cleaning,
             duration_s=tr[-1, 2] - tr[0, 2], t_start=tr[0, 2],
             frac_dt_1s=float(np.mean(dt == 1)), n_gaps_gt1s=int((dt > 1).sum()), n_gaps_gt10s=int((dt > 10).sum()),
             max_gap_s=float(dt.max()), non_integer_t=int((tr[:, 2] != np.round(tr[:, 2])).sum()),
             gps_path_km=step.sum() / 1000, median_speed_ms=float(np.nanmedian(speed)), max_speed_ms=float(np.nanmax(speed)),
             frac_stationary=float(np.mean(step < 1.0)),
             lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
             n_nodes=N, n_arcs=A, isolated_nodes=int((deg == 0).sum()), self_loops=int((arcs[:, 0] == arcs[:, 1]).sum()),
             duplicate_arcs=A - len(pairs), frac_bidirectional=rev / A,
             weak_components=n_weak, largest_weak_frac=np.bincount(lab_w).max() / N,
             strong_components=n_strong, largest_strong_frac=np.bincount(lab_s).max() / N)
    if trip.route is not None:
        rt = trip.route
        L = np.hypot(*(trip.node_xy[arcs[rt, 1]] - trip.node_xy[arcs[rt, 0]]).T)
        a, b = trip.node_xy[arcs[rt, 0]], trip.node_xy[arcs[rt, 1]]
        sel = xy[:: max(1, len(xy) // 400)]
        d, _ = point_segment(sel[:, :1], sel[:, 1:2], a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1])
        dmin = d.min(1)
        r.update(route_arcs=len(rt), route_km=L.sum() / 1000, route_invalid_ids=int(((rt < 0) | (rt >= A)).sum()),
                 route_continuity=float(np.mean(arcs[rt[:-1], 1] == arcs[rt[1:], 0])) if len(rt) > 1 else 1.0,
                 route_repeated_arcs=len(rt) - len(set(rt)),
                 gps_route_dist_median=float(np.median(dmin)), gps_route_dist_p90=float(np.percentile(dmin, 90)),
                 gps_route_dist_max=float(dmin.max()), frac_gps_over_40m=float(np.mean(dmin > 40)))
    return r, dt, step


def folium_map(part, tid, path):
    from pathsnap import viz
    trip = load_trip(os.path.join(ROOT, "data", part, tid))
    m = viz.base_map(trip)
    viz.add_network(m, trip)
    viz.add_gps(m, trip)
    if trip.route is not None:
        viz.add_route(m, trip, trip.route, "Reference route", "#1a9850", weight=5)
    viz.add_start_end(m, trip)
    viz.add_legend(m, f"Trip {tid} ({part})", [("Raw GPS observations", "#1f4fd1"),
                   ("Reference route", "#1a9850"), ("Road network (nearby)", "#8c8c8c")])
    viz.finish(m, path)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=["00000001", "00000033"])
    ap.add_argument("--maps-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "maps"), exist_ok=True)
    if args.maps_only:
        for tid in args.map:
            part = "train" if os.path.isdir(os.path.join(ROOT, "data", "train", tid)) else "student_test"
            folium_map(part, tid, os.path.join(ROOT, "maps", f"eda_trip_{tid}.html"))
        return
    rows, dts, steps = [], [], []
    for part in ["train", "student_test"]:
        for tid in list_trips(os.path.join(ROOT, "data", part)):
            trip = load_trip(os.path.join(ROOT, "data", part, tid))
            r, dt, step = trip_stats(part, trip)
            rows.append(r)
            if part == "train":
                dts.append(dt); steps.append(step)
            print(part, tid, "ok", flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    summ = inv.groupby("partition").agg(trips=("trip", "count"), points=("n_points", "sum"),
                                        median_points=("n_points", "median"), median_arcs=("n_arcs", "median"),
                                        max_arcs=("n_arcs", "max"), route_arcs=("route_arcs", "sum"))
    summ.to_csv(os.path.join(OUT, "partition_summary.csv"))
    print(summ)

    tr = inv[inv.partition == "train"]
    dt = np.concatenate(dts)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].hist(np.clip(dt, 0, 20), bins=40); axs[0, 0].set_yscale("log")
    axs[0, 0].set(title="Sampling interval (train, clipped at 20 s)", xlabel="seconds")
    axs[0, 1].hist(tr.n_points, bins=30); axs[0, 1].set(title="GPS points per trip (train)", xlabel="points")
    axs[0, 2].hist(tr.gps_route_dist_median, bins=30); axs[0, 2].set(title="Median GPS-to-route distance per trip (train)", xlabel="m")
    axs[1, 0].scatter(tr.gps_path_km, tr.route_km, s=12); axs[1, 0].plot([0, 130], [0, 130], "r--", lw=1)
    axs[1, 0].set(title="GPS path length vs reference route length (train)", xlabel="GPS km", ylabel="route km")
    axs[1, 1].hist(np.log10(tr.n_arcs), bins=30); axs[1, 1].set(title="Road arcs per trip network (train)", xlabel="log10(arcs)")
    axs[1, 2].hist(tr.max_gap_s.clip(upper=600), bins=30); axs[1, 2].set(title="Max time gap per trip (train, clipped 600 s)", xlabel="s")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_overview.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(inv.lon_min, inv.lat_min, c=(inv.partition == "train"), cmap="coolwarm", s=20)
    ax.set(title="Trip locations (train = red, student_test = blue)", xlabel="lon", ylabel="lat")
    fig.savefig(os.path.join(OUT, "trip_locations.png"), dpi=120)

    with open(os.path.join(OUT, "data_quality.md"), "w") as fh:
        fh.write("# Data quality findings (train + student_test)\n\n")
        fh.write(f"- Invalid / non-finite GPS rows dropped: {inv.dropped_invalid.sum()}; duplicate timestamps dropped: {inv.dropped_dup_time.sum()}\n")
        fh.write(f"- Trips with non-integer timestamps: {list(inv.trip[inv.non_integer_t > 0])}\n")
        fh.write(f"- Trips with gaps > 60 s: {list(inv.trip[inv.max_gap_s > 60])}\n")
        fh.write(f"- Trips where median GPS-to-route distance > 30 m (offset reference): {list(inv.trip[inv.gps_route_dist_median > 30])}\n")
        fh.write(f"- Trips with GPS path > 1.5x route length (route covers part of trip / stationary jitter): "
                 f"{list(inv.trip[inv.gps_path_km > 1.5 * inv.route_km])}\n")
        fh.write(f"- Trips whose reference route has disconnected consecutive arcs: {list(inv.trip[inv.route_continuity < 1])}\n")
        fh.write(f"- Networks with self-loops: {int((inv.self_loops > 0).sum())}; with duplicate arcs: {int((inv.duplicate_arcs > 0).sum())}\n")
        fh.write(f"- Median share of arcs that have a reverse twin: {inv.frac_bidirectional.median():.2f}\n")
        fh.write(f"- Median number of weakly connected components per network: {inv.weak_components.median():.0f}\n")
    for tid in args.map:
        part = "train" if os.path.isdir(os.path.join(ROOT, "data", "train", tid)) else "student_test"
        folium_map(part, tid, os.path.join(ROOT, "maps", f"eda_trip_{tid}.html"))
    print("done")


if __name__ == "__main__":
    main()
