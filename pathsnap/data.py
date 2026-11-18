"""Raw-data loading and geographic projection for PathSnap trips.

Each trip folder <id>/ holds four header-less, whitespace-separated files:
  <id>.track  lon lat t        GPS observations (~1 Hz)
  <id>.nodes  lon lat          road-network nodes, row index = node id
  <id>.arcs   from to          directed road segments, row index = arc id
  <id>.route  arc_id           reference route (ordered arc ids) - may be absent for evaluator trips
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

EARTH_R = 6371008.8  # mean Earth radius (m)


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n, 3) lon, lat, t   (cleaned, time-sorted)
    nodes: np.ndarray            # (N, 2) lon, lat
    arcs: np.ndarray             # (A, 2) int from, to
    route: np.ndarray | None     # (R,) int arc ids, or None
    # derived (filled by project())
    lon0: float = 0.0
    lat0: float = 0.0
    xy: np.ndarray = field(default=None, repr=False)        # GPS in local metres
    node_xy: np.ndarray = field(default=None, repr=False)   # nodes in local metres
    cleaning: dict = field(default_factory=dict)

    @property
    def n_arcs(self):
        return len(self.arcs)


def _read(path, ncols, dtype=float):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def to_local_xy(lon, lat, lon0, lat0):
    """Local equirectangular projection centred on (lon0, lat0), output in metres.

    Trips span at most ~50 km, so the scale error of this tangent-plane approximation
    is well below GPS noise (<0.1 %). It is far cheaper than a full UTM transform and works
    for trips anywhere on the globe (the dataset spans 4 continents)."""
    k = np.pi / 180.0 * EARTH_R
    x = (np.asarray(lon) - lon0) * k * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * k
    return np.column_stack([x, y])


def load_trip(folder: str, trip_id: str | None = None) -> Trip:
    trip_id = trip_id or os.path.basename(os.path.normpath(folder))
    p = lambda ext: os.path.join(folder, f"{trip_id}.{ext}")
    track = _read(p("track"), 3)
    nodes = _read(p("nodes"), 2)
    arcs = _read(p("arcs"), 2, dtype=np.int64)
    route = _read(p("route"), 1, dtype=np.int64)
    route = route[:, 0] if route is not None else None

    info = {"raw_points": len(track)}
    # --- cleaning --------------------------------------------------------
    finite = np.isfinite(track).all(1)
    plausible = (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    keep = finite & plausible
    info["dropped_invalid"] = int((~keep).sum())
    track = track[keep]
    order = np.argsort(track[:, 2], kind="stable")
    info["reordered"] = int((order != np.arange(len(order))).sum())
    track = track[order]
    # duplicate timestamps -> keep first
    dup = np.r_[False, np.diff(track[:, 2]) <= 0]
    info["dropped_dup_time"] = int(dup.sum())
    track = track[~dup]

    trip = Trip(trip_id, track, nodes, arcs, route, cleaning=info)
    project(trip)
    return trip


def project(trip: Trip):
    trip.lon0 = float(np.mean(trip.track[:, 0]))
    trip.lat0 = float(np.mean(trip.track[:, 1]))
    trip.xy = to_local_xy(trip.track[:, 0], trip.track[:, 1], trip.lon0, trip.lat0)
    trip.node_xy = to_local_xy(trip.nodes[:, 0], trip.nodes[:, 1], trip.lon0, trip.lat0)


def list_trips(partition_dir: str):
    return sorted(d for d in os.listdir(partition_dir)
                  if os.path.isdir(os.path.join(partition_dir, d)))


def thin_track(xy: np.ndarray, min_dist: float) -> np.ndarray:
    """Indices of GPS points kept after removing points closer than `min_dist` metres to the
    previously kept point (stationary jitter / 1 Hz redundancy). First and last points are kept.
    Dropped points inherit the prediction of the most recent kept point."""
    keep = [0]
    last = xy[0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - last)) >= min_dist:
            keep.append(i)
            last = xy[i]
    if keep[-1] != len(xy) - 1:
        keep.append(len(xy) - 1)
    return np.asarray(keep)
