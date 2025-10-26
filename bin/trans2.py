# transmogrify.py
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry

_EPS = 1e-12


def _point_on_curve_by_u(curve: LineString, u: float) -> Point:
    if curve.length <= _EPS:
        x, y = curve.coords[0]
        return Point(x, y)
    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    return curve.interpolate(u * curve.length)


def _normalized_u_from_projection(P: Point, curve: LineString) -> float:
    if curve.length <= _EPS:
        return 0.0
    s = curve.project(P)
    return min(max(s / curve.length, 0.0), 1.0)


def _clamped_param_on_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    AB = B - A
    denom = AB.dot(AB)
    if denom <= _EPS:
        return 0.0
    t = (P - A).dot(AB) / denom
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def _closest_point_sqdist(P: np.ndarray, A: np.ndarray, B: np.ndarray, t: float) -> float:
    Q = A + t * (B - A)
    d = P - Q
    return d.dot(d)


def _as_np(pt: Point) -> np.ndarray:
    return np.array([pt.x, pt.y], dtype=float)


def map_uv_to_xy(u: float, v: float,
                 centerline: LineString,
                 top_line: LineString,
                 bottom_line: LineString) -> Point:
    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    v = 0.0 if math.isnan(v) else min(max(v, 0.0), 1.0)

    B = _as_np(_point_on_curve_by_u(bottom_line, u))
    S = _as_np(_point_on_curve_by_u(centerline,   u))
    T = _as_np(_point_on_curve_by_u(top_line,     u))

    if v <= 0.5:
        XY = (1.0 - 2.0*v) * B + (2.0*v) * S
    else:
        XY = (2.0 - 2.0*v) * S + (2.0*v - 1.0) * T

    return Point(float(XY[0]), float(XY[1]))


def map_xy_to_uv(P: Point,
                 centerline: LineString,
                 top_line: LineString,
                 bottom_line: LineString) -> tuple[float, float]:
    uS_prov = _normalized_u_from_projection(P, centerline)
    Bpt = _point_on_curve_by_u(bottom_line, uS_prov)
    Spt = _point_on_curve_by_u(centerline,   uS_prov)
    Tpt = _point_on_curve_by_u(top_line,     uS_prov)

    B = _as_np(Bpt)
    S = _as_np(Spt)
    T = _as_np(Tpt)
    Pn = _as_np(P)

    len_BS2 = np.dot(S - B, S - B)
    len_ST2 = np.dot(T - S, T - S)
    valid_BS = len_BS2 > _EPS
    valid_ST = len_ST2 > _EPS

    if not valid_BS and not valid_ST:
        return uS_prov, 0.5

    if valid_BS:
        t_bs = _clamped_param_on_segment(Pn, B, S)
        d_bs2 = _closest_point_sqdist(Pn, B, S, t_bs)
    else:
        t_bs, d_bs2 = 0.0, float('inf')

    if valid_ST:
        t_st = _clamped_param_on_segment(Pn, S, T)
        d_st2 = _closest_point_sqdist(Pn, S, T, t_st)
    else:
        t_st, d_st2 = 0.0, float('inf')

    uB = _normalized_u_from_projection(P, bottom_line)
    uS = _normalized_u_from_projection(P, centerline)
    uT = _normalized_u_from_projection(P, top_line)

    if d_bs2 <= d_st2:
        v = 0.5 * t_bs
        u = (1.0 - t_bs) * uB + t_bs * uS
    else:
        v = 0.5 + 0.5 * t_st
        u = (1.0 - t_st) * uS + t_st * uT

    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return u, v


def to_uv_geometry(geom: BaseGeometry,
                   centerline: LineString,
                   top_line: LineString,
                   bottom_line: LineString):
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


def from_uv_geometry(geom_uv: BaseGeometry,
                     centerline: LineString,
                     top_line: LineString,
                     bottom_line: LineString):
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


def straighten(geoms,
               centerline: LineString,
               top_line: LineString,
               bottom_line: LineString):
    return [to_uv_geometry(g, centerline, top_line, bottom_line) for g in geoms]


def unstraighten(geoms_uv,
                 centerline: LineString,
                 top_line: LineString,
                 bottom_line: LineString):
    return [from_uv_geometry(g, centerline, top_line, bottom_line) for g in geoms_uv]
