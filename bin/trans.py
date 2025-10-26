# transmogrify.py
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry

_EPS = 1e-12
_SAMPLES = 50  # coarse samples for station-consistent lookup


# ---------- low-level helpers ----------

def _point_on_curve_by_u(curve: LineString, u: float) -> Point:
    """Arclength parameter on the curve itself (used only for the centerline)."""
    if curve.length <= _EPS:
        x, y = curve.coords[0]
        return Point(x, y)
    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    return curve.interpolate(u * curve.length)


def _normalized_u_from_projection(P: Point, curve: LineString) -> float:
    """Project P to 'curve' and return normalized station in [0,1]."""
    if curve.length <= _EPS:
        return 0.0
    s = curve.project(P)
    return min(max(s / curve.length, 0.0), 1.0)


def _as_np(pt: Point) -> np.ndarray:
    return np.array([pt.x, pt.y], dtype=float)


# ---------- station-consistent sampling ----------

def _point_on_curve_at_center_u(
    curve: LineString, centerline: LineString, u: float, samples: int = _SAMPLES
) -> Point:
    """
    Return the point on 'curve' whose projection onto 'centerline' is at station u*len(centerline).
    Coarse sampling + short local refinement. Robust for corridor-like polylines.
    """
    if curve.length <= _EPS:
        x, y = curve.coords[0]
        return Point(x, y)

    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    target_s = u * centerline.length

    # 1) coarse search along 'curve' by arclength parameter t in [0,1]
    best_pt = None
    best_err = float("inf")
    best_t = 0.0
    for i in range(samples + 1):
        t = i / samples
        cand = curve.interpolate(t * curve.length)
        s = centerline.project(cand)
        err = abs(s - target_s)
        if err < best_err:
            best_err = err
            best_pt = cand
            best_t = t

    # 2) tiny local refinement around best sample
    refine_steps = 6
    window = 1.0 / samples
    for _ in range(refine_steps):
        improved = False
        for dt in (-window, 0.0, window):
            tt = min(max(best_t + dt, 0.0), 1.0)
            cand = curve.interpolate(tt * curve.length)
            s = centerline.project(cand)
            err = abs(s - target_s)
            if err + 1e-15 < best_err:
                best_err = err
                best_pt = cand
                best_t = tt
                improved = True
        if not improved:
            break
        window *= 0.5

    return best_pt


def _proj_param_and_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """
    Project P onto segment A->B. Return (t_clamped in [0,1], squared distance).
    """
    AB = B - A
    AP = P - A
    denom = float(np.dot(AB, AB))
    if denom <= _EPS:
        return 0.0, float(np.dot(AP, AP))
    t = float(np.dot(AP, AB) / denom)
    t_clamped = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    closest = A + t_clamped * AB
    d2 = float(np.dot(P - closest, P - closest))
    return t_clamped, d2


# ---------- forward / inverse maps ----------

def map_uv_to_xy(
    u: float,
    v: float,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
) -> Point:
    """
    Piecewise blend that is station-consistent:
      if v<=0.5: (1-2v) * B(u) + (2v) * S(u)
      else:      (2-2v) * S(u) + (2v-1) * T(u)
    where B,T are sampled at the SAME centerline station u as S.
    """
    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    v = 0.0 if math.isnan(v) else min(max(v, 0.0), 1.0)

    # Centerline is parameterized by itself
    S = _as_np(_point_on_curve_by_u(centerline, u))
    # Top/Bottom aligned to the same centerline station u
    B = _as_np(_point_on_curve_at_center_u(bottom_line, centerline, u))
    T = _as_np(_point_on_curve_at_center_u(top_line, centerline, u))

    if v <= 0.5:
        t = 2.0 * v
        XY = (1.0 - t) * B + t * S
    else:
        t = 2.0 * (v - 0.5)
        XY = (1.0 - t) * S + t * T

    return Point(float(XY[0]), float(XY[1]))


def map_xy_to_uv(
    P: Point,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
) -> tuple[float, float]:
    """
    Inverse of the piecewise forward map:
      - u: station on centerline
      - v: choose the closer of segments B->S or S->T and map its local t into [0,0.5] or [0.5,1].
    """
    # u from projection on centerline
    u = _normalized_u_from_projection(P, centerline)

    # sample corridor frame at the same station
    S = _as_np(_point_on_curve_by_u(centerline, u))
    B = _as_np(_point_on_curve_at_center_u(bottom_line, centerline, u))
    T = _as_np(_point_on_curve_at_center_u(top_line, centerline, u))
    Pn = _as_np(P)

    # project onto both branches
    t_bs, d2_bs = _proj_param_and_dist(Pn, B, S)
    t_st, d2_st = _proj_param_and_dist(Pn, S, T)

    # handle near-degenerate widths by favoring center
    if (np.linalg.norm(S - B) <= 1e-9) and (np.linalg.norm(T - S) <= 1e-9):
        v = 0.5
    elif d2_bs <= d2_st:
        v = 0.5 * t_bs          # maps t in [0,1] to v in [0,0.5]
    else:
        v = 0.5 + 0.5 * t_st     # maps t in [0,1] to v in [0.5,1]

    # clamp for safety
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return -5*u, 1.5*v


# ---------- geometry converters (unchanged API) ----------

def to_uv_geometry(
    geom: BaseGeometry,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
):
    if geom is None:
        return None
    gt = geom.geom_type
    if gt == "Point":
        return Point(map_xy_to_uv(geom, centerline, top_line, bottom_line))
    elif gt == "LineString":
        coords = [map_xy_to_uv(Point(x, y), centerline, top_line, bottom_line)
                  for x, y in geom.coords]
        return LineString(coords)
    elif gt == "Polygon":
        ext = [map_xy_to_uv(Point(x, y), centerline, top_line, bottom_line)
               for x, y in geom.exterior.coords]
        if ext[0] != ext[-1]:
            ext.append(ext[0])
        holes = []
        for ring in geom.interiors:
            hole = [map_xy_to_uv(Point(x, y), centerline, top_line, bottom_line)
                    for x, y in ring.coords]
            if hole[0] != hole[-1]:
                hole.append(hole[0])
            holes.append(hole)
        return Polygon(ext, holes)
    elif gt == "MultiLineString":
        return MultiLineString([
            to_uv_geometry(g, centerline, top_line, bottom_line) for g in geom.geoms
        ])
    elif gt == "MultiPolygon":
        return MultiPolygon([
            to_uv_geometry(g, centerline, top_line, bottom_line) for g in geom.geoms
        ])
    elif gt.startswith("Multi"):
        return type(geom)([
            to_uv_geometry(g, centerline, top_line, bottom_line) for g in geom.geoms
        ])
    else:
        return None


def from_uv_geometry(
    geom_uv: BaseGeometry,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
):
    if geom_uv is None:
        return None
    gt = geom_uv.geom_type
    if gt == "Point":
        u, v = geom_uv.x, geom_uv.y
        return map_uv_to_xy(u, v, centerline, top_line, bottom_line)
    elif gt == "LineString":
        pts = [map_uv_to_xy(u, v, centerline, top_line, bottom_line)
               for u, v in geom_uv.coords]
        return LineString([(p.x, p.y) for p in pts])
    elif gt == "Polygon":
        ext_pts = [map_uv_to_xy(u, v, centerline, top_line, bottom_line)
                   for u, v in geom_uv.exterior.coords]
        ext = [(p.x, p.y) for p in ext_pts]
        if ext[0] != ext[-1]:
            ext.append(ext[0])
        holes = []
        for ring in geom_uv.interiors:
            ring_pts = [map_uv_to_xy(u, v, centerline, top_line, bottom_line)
                        for u, v in ring.coords]
            hole = [(p.x, p.y) for p in ring_pts]
            if hole[0] != hole[-1]:
                hole.append(hole[0])
            holes.append(hole)
        return Polygon(ext, holes)
    elif gt == "MultiLineString":
        return MultiLineString([
            from_uv_geometry(g, centerline, top_line, bottom_line) for g in geom_uv.geoms
        ])
    elif gt == "MultiPolygon":
        return MultiPolygon([
            from_uv_geometry(g, centerline, top_line, bottom_line) for g in geom_uv.geoms
        ])
    elif gt.startswith("Multi"):
        return type(geom_uv)([
            from_uv_geometry(g, centerline, top_line, bottom_line) for g in geom_uv.geoms
        ])
    else:
        return None


# ---------- public helpers (unchanged) ----------

def straighten(
    geoms,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
):
    # XY → UV using the real corridor
    geoms_uv = [to_uv_geometry(
        g, centerline, top_line, bottom_line) for g in geoms]
    return geoms_uv


def unstraighten(
    geoms_uv,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
):
    # UV → XY using the same corridor
    return [from_uv_geometry(g, centerline, top_line, bottom_line) for g in geoms_uv]
