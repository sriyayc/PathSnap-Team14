"""Folium helpers shared by the EDA maps, prediction maps and report figures.

OpenStreetMap's own tile servers refuse requests from local HTML files (HTTP 403), so the
base layers are CartoDB Positron and Esri World Street Map, which serve local files fine.
"""
from __future__ import annotations

import folium
import numpy as np
from scipy.spatial import cKDTree

CARTO = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"


def base_map(trip):
    m = folium.Map(location=[trip.track[:, 1].mean(), trip.track[:, 0].mean()], zoom_start=14, tiles=None)
    folium.TileLayer(CARTO, attr="&copy; OpenStreetMap contributors &copy; CARTO", name="CartoDB light").add_to(m)
    folium.TileLayer(ESRI, attr="Tiles &copy; Esri", name="Esri streets").add_to(m)
    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()], [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    return m


def arcs_latlon(trip, arc_ids):
    """Polyline coordinates for a sequence of arcs, split wherever consecutive arcs don't connect."""
    lines, cur = [], []
    prev_to = None
    for a in arc_ids:
        u, v = trip.arcs[a]
        if prev_to is None or u != prev_to:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[trip.nodes[u, 1], trip.nodes[u, 0]]]
        cur.append([trip.nodes[v, 1], trip.nodes[v, 0]])
        prev_to = v
    if len(cur) > 1:
        lines.append(cur)
    return lines


def add_network(m, trip, within=100.0):
    mid = (trip.node_xy[trip.arcs[:, 0]] + trip.node_xy[trip.arcs[:, 1]]) / 2
    d, _ = cKDTree(trip.xy).query(mid, distance_upper_bound=within)
    fg = folium.FeatureGroup(f"Road network (arcs within {within:.0f} m of GPS)")
    seen = set()
    for a in np.flatnonzero(np.isfinite(d)):
        u, v = trip.arcs[a]
        if (v, u) in seen:
            continue
        seen.add((u, v))
        folium.PolyLine([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color="#8c8c8c", weight=1.5, opacity=0.7).add_to(fg)
    fg.add_to(m)


def add_gps(m, trip, max_points=2000):
    fg = folium.FeatureGroup("Raw GPS observations")
    step = max(1, len(trip.track) // max_points)
    for lon, lat, t in trip.track[::step]:
        folium.CircleMarker([lat, lon], radius=2.5, color="#1f4fd1", weight=1, fill=True, fill_opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(fg)
    fg.add_to(m)


def add_route(m, trip, arc_ids, name, color, weight=5, dash=None, show=True):
    fg = folium.FeatureGroup(name, show=show)
    for line in arcs_latlon(trip, arc_ids):
        folium.PolyLine(line, color=color, weight=weight, opacity=0.85, dash_array=dash, tooltip=name).add_to(fg)
    fg.add_to(m)


def add_start_end(m, trip):
    fg = folium.FeatureGroup("Start / end")
    folium.Marker([trip.track[0, 1], trip.track[0, 0]], tooltip="GPS start", icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker([trip.track[-1, 1], trip.track[-1, 0]], tooltip="GPS end", icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)


def add_legend(m, title, items):
    rows = "".join(f'<div><span style="display:inline-block;width:22px;height:5px;background:{c};'
                   f'vertical-align:middle;margin-right:6px"></span>{label}</div>' for label, c in items)
    html = (f'<div style="position:fixed;bottom:28px;left:28px;z-index:9999;background:white;padding:8px 12px;'
            f'border:1px solid #888;border-radius:6px;font:13px sans-serif"><b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def finish(m, path):
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(path)
