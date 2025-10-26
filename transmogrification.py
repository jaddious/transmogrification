# transmogrify.py (optimized)
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
import weakref

_EPS = 1e-12

# ------------------------------------------------------------------------------
# Polyline preprocessing & fast geometry kernels (NumPy)
# ------------------------------------------------------------------------------


def _resample_line(line: LineString, step: float) -> LineString:
    """Uniform arclength resampling to spacing ~step (in same units as the line)."""
    P = _Poly(line)
    if P.length <= _EPS or step <= _EPS:
        return line
    n = int(max(2, round(P.length / step)))  # at least 2 points
    us = np.linspace(0.0, 1.0, n)
    pts = [tuple(P.point_at_u(float(u))) for u in us]
    return LineString(pts)


def _smooth_monotone_s(s: np.ndarray) -> np.ndarray:
    """Light smoothing + strictly increasing enforcement."""
    if s.size <= 2:
        return s.copy()
    # 5-tap triangular FIR: [1,2,3,2,1] / 9
    w = np.array([1, 2, 3, 2, 1], dtype=float)
    w /= w.sum()
    s_pad = np.pad(s, (2, 2), mode="edge")
    s_sm = np.convolve(s_pad, w, mode="valid")
    # strictly increasing with tiny epsilon to kill plateaus
    eps = 1e-9 * max(1.0, s_sm[-1] - s_sm[0])
    for i in range(1, s_sm.size):
        if s_sm[i] <= s_sm[i-1]:
            s_sm[i] = s_sm[i-1] + eps
    return s_sm

class _Poly:
    """Lightweight, cached numeric view of a Shapely LineString."""
    __slots__ = ("pts", "seg_vec", "seg_len", "seg_len2", "cumlen", "length")

    def __init__(self, line: LineString):
        # Shape: (N,2)
        pts = np.asarray(line.coords, dtype=float)
        if pts.shape[0] == 1:  # normalize degenerate lines to 2 identical points
            pts = np.vstack([pts, pts])
        self.pts = pts
        diffs = pts[1:] - pts[:-1]
        self.seg_vec = diffs                            # (M,2)
        self.seg_len2 = np.einsum("ij,ij->i", diffs, diffs)  # (M,)
        self.seg_len = np.sqrt(np.maximum(self.seg_len2, 0.0))
        self.cumlen = np.concatenate(
            [[0.0], np.cumsum(self.seg_len)])  # (M+1,)
        self.length = float(self.cumlen[-1]) if self.cumlen.size else 0.0

    # ---- arclength interpolation: t in [0,1] -> point ----
    def point_at_u(self, u: float) -> np.ndarray:
        if self.length <= _EPS:
            return self.pts[0].copy()
        u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
        s = u * self.length
        # find segment index i with cumlen[i] <= s <= cumlen[i+1]
        i = int(np.searchsorted(self.cumlen, s, side="right") - 1)
        i = max(0, min(i, self.seg_vec.shape[0] - 1))
        s0, s1 = self.cumlen[i], self.cumlen[i + 1]
        w = 0.0 if s1 <= s0 + _EPS else (s - s0) / (s1 - s0)
        return self.pts[i] + w * self.seg_vec[i]

    # ---- fast point projection onto polyline ----
    def project_u(self, P: np.ndarray) -> float:
        """Return normalized arclength parameter u in [0,1] for projection of P."""
        if self.length <= _EPS:
            return 0.0
        A = self.pts[:-1]            # (M,2)
        V = self.seg_vec             # (M,2)
        AP = P - A                   # (M,2)
        denom = self.seg_len2        # (M,)

        # raw t along each segment
        t = np.where(denom > _EPS, np.einsum("ij,ij->i", AP, V) / denom, 0.0)
        t = np.clip(t, 0.0, 1.0)

        # closest point per segment and squared distance
        C = A + (t[:, None] * V)     # (M,2)
        d2 = np.einsum("ij,ij->i", (P - C), (P - C))

        j = int(np.argmin(d2))
        s = self.cumlen[j] + float(t[j]) * float(self.seg_len[j])
        return min(max(s / self.length, 0.0), 1.0)

# ------------------------------------------------------------------------------
# Corridor context: precomputes the centerline-projection station of each vertex
# on top/bottom lines to answer queries by binary search (no coarse sampling).
# ------------------------------------------------------------------------------


class _CorridorCtx:
    __slots__ = ("center", "top", "bot", "top_proj_s", "bot_proj_s")

    def __init__(self, center: LineString, top: LineString, bot: LineString):
        self.center = _Poly(center)
        self.top = _Poly(top)
        self.bot = _Poly(bot)

        # Precompute, for every vertex on top/bottom, its projected arclength s on centerline.
        # We'll use these arrays to invert: target s -> point on curve via a binary search
        # over the vertex stations followed by linear interpolation of vertex positions.
        self.top_proj_s = self._vertex_proj_s(
            self.top, self.center)  # shape (Nt,)
        self.bot_proj_s = self._vertex_proj_s(
            self.bot, self.center)  # shape (Nb,)

    @staticmethod
    def _vertex_proj_s(curve: _Poly, center: _Poly) -> np.ndarray:
        if center.length <= _EPS:
            return np.zeros(curve.pts.shape[0], dtype=float)
        s_list = np.empty(curve.pts.shape[0], dtype=float)
        for i, P in enumerate(curve.pts):
            u = center.project_u(P)
            s_list[i] = u * center.length
        # enforce weak monotonicity then smooth and enforce strict monotonicity
        np.maximum.accumulate(s_list, out=s_list)
        s_list = _smooth_monotone_s(s_list)
        return s_list

    @staticmethod
    def _point_on_curve_at_s(curve: _Poly, s_vertices: np.ndarray, target_s: float) -> np.ndarray:
        # target_s is clamped to [0, center.length] by caller
        # Find bracketing vertices by station on the centerline: s_i <= s <= s_{i+1}
        Nt = s_vertices.shape[0]
        if Nt == 0:
            return curve.pts[0].copy()
        if Nt == 1:
            return curve.pts[0].copy()

        # Binary search in vertex stations
        idx = int(np.searchsorted(s_vertices, target_s, side="right") - 1)
        idx = max(0, min(idx, Nt - 2))
        s0, s1 = s_vertices[idx], s_vertices[idx + 1]
        if s1 <= s0 + _EPS:
            # degenerate stretch: fall back to nearest vertex
            return curve.pts[idx if abs(target_s - s0) <= abs(target_s - s1) else (idx + 1)].copy()

        w = (target_s - s0) / (s1 - s0)
        return curve.pts[idx] + w * (curve.pts[idx + 1] - curve.pts[idx])
    
    def __init__(self, center: LineString, top: LineString, bot: LineString):
        self.center = _Poly(center)

        # --- NEW: resample top/bottom to uniform arclength before projecting
        # Use a step proportional to centerline length; tweak 0.005 if needed.
        step = max(self.center.length * 0.005, 1e-6)
        top_rs = _resample_line(top, step)
        bot_rs = _resample_line(bot, step)

        self.top = _Poly(top_rs)
        self.bot = _Poly(bot_rs)

        self.top_proj_s = self._vertex_proj_s(self.top, self.center)
        self.bot_proj_s = self._vertex_proj_s(self.bot, self.center)

    # API helpers used by forward/inverse maps
    def center_point_at_u(self, u: float) -> np.ndarray:
        return self.center.point_at_u(u)

    def bottom_point_at_center_u(self, u: float) -> np.ndarray:
        target_s = (0.0 if math.isnan(u) else min(
            max(u, 0.0), 1.0)) * self.center.length
        return self._point_on_curve_at_s(self.bot, self.bot_proj_s, target_s)

    def top_point_at_center_u(self, u: float) -> np.ndarray:
        target_s = (0.0 if math.isnan(u) else min(
            max(u, 0.0), 1.0)) * self.center.length
        return self._point_on_curve_at_s(self.top, self.top_proj_s, target_s)

    def project_u_on_center(self, P: np.ndarray) -> float:
        return self.center.project_u(P)

# ------------------------------------------------------------------------------
# Simple last-corridor cache (object-identity keyed)
# ------------------------------------------------------------------------------

# Keep weak references so GC of LineStrings isn’t hindered.


class _Key:
    __slots__ = ("c", "t", "b")

    def __init__(self, c, t, b):
        self.c = weakref.ref(c)
        self.t = weakref.ref(t)
        self.b = weakref.ref(b)


_last_key: _Key | None = None
_last_ctx: _CorridorCtx | None = None


def _get_ctx(centerline: LineString, top_line: LineString, bottom_line: LineString) -> _CorridorCtx:
    global _last_key, _last_ctx
    if _last_key is not None:
        if _last_key.c() is centerline and _last_key.t() is top_line and _last_key.b() is bottom_line:
            return _last_ctx  # type: ignore[return-value]
    _last_key = _Key(centerline, top_line, bottom_line)
    _last_ctx = _CorridorCtx(centerline, top_line, bottom_line)
    return _last_ctx

# ------------------------------------------------------------------------------
# Original helpers adapted to fast kernels
# ------------------------------------------------------------------------------


def _as_np(pt: Point) -> np.ndarray:
    return np.array([pt.x, pt.y], dtype=float)


def _point_on_curve_by_u(curve: LineString, u: float) -> Point:
    poly = _Poly(curve)
    p = poly.point_at_u(u)
    return Point(float(p[0]), float(p[1]))


def _normalized_u_from_projection(P: Point, curve: LineString) -> float:
    poly = _Poly(curve)
    return poly.project_u(_as_np(P))


def _proj_param_and_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
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

# ------------------------------------------------------------------------------
# Forward / inverse maps (same API/behavior)
# ------------------------------------------------------------------------------


def map_uv_to_xy(
    u: float,
    v: float,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
) -> Point:
    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    v = 0.0 if math.isnan(v) else min(max(v, 0.0), 1.0)

    ctx = _get_ctx(centerline, top_line, bottom_line)

    S = ctx.center_point_at_u(u)
    B = ctx.bottom_point_at_center_u(u)
    T = ctx.top_point_at_center_u(u)

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
    ctx = _get_ctx(centerline, top_line, bottom_line)

    # u from projection on centerline
    Pn = _as_np(P)
    u = ctx.project_u_on_center(Pn)

    # sample corridor frame at the same station
    S = ctx.center_point_at_u(u)
    B = ctx.bottom_point_at_center_u(u)
    T = ctx.top_point_at_center_u(u)

    # project onto both branches
    t_bs, d2_bs = _proj_param_and_dist(Pn, B, S)
    t_st, d2_st = _proj_param_and_dist(Pn, S, T)

    # handle near-degenerate widths by favoring center
    if (np.linalg.norm(S - B) <= 1e-9) and (np.linalg.norm(T - S) <= 1e-9):
        v = 0.5
    elif d2_bs <= d2_st:
        v = 0.5 * t_bs
    else:
        v = 0.5 + 0.5 * t_st

    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return -8*u, 1*v  # preserve your scaling

# ------------------------------------------------------------------------------
# Geometry converters (unchanged API)
# ------------------------------------------------------------------------------


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
        try:
            coords = [map_xy_to_uv(Point(x, y), centerline, top_line, bottom_line)
                    for x, y in geom.coords]
        except Exception as e:
            coords = [map_xy_to_uv(Point(x, y), centerline, top_line, bottom_line)
                      for x, y, z in geom.coords]
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

# ------------------------------------------------------------------------------
# Public helpers (unchanged)
# ------------------------------------------------------------------------------


def straighten(
    geoms,
    centerline: LineString,
    top_line: LineString,
    bottom_line: LineString
):
    # Warm the corridor cache once per batch.
    _get_ctx(centerline, top_line, bottom_line)
    return [to_uv_geometry(g, centerline, top_line, bottom_line) for g in geoms]

def _resample_line(line: LineString, step: float) -> LineString:
    """Uniform arclength resampling to spacing ~step (in same units as the line)."""
    P = _Poly(line)
    if P.length <= _EPS or step <= _EPS:
        return line
    n = int(max(2, round(P.length / step)))  # at least 2 points
    us = np.linspace(0.0, 1.0, n)
    pts = [tuple(P.point_at_u(float(u))) for u in us]
    return LineString(pts)

def _smooth_monotone_s(s: np.ndarray) -> np.ndarray:
    """Light smoothing + strictly increasing enforcement."""
    if s.size <= 2:
        return s.copy()
    # 5-tap triangular FIR: [1,2,3,2,1] / 9
    w = np.array([1, 2, 3, 2, 1], dtype=float)
    w /= w.sum()
    s_pad = np.pad(s, (2, 2), mode="edge")
    s_sm = np.convolve(s_pad, w, mode="valid")
    # strictly increasing with tiny epsilon to kill plateaus
    eps = 1e-9 * max(1.0, s_sm[-1] - s_sm[0])
    for i in range(1, s_sm.size):
        if s_sm[i] <= s_sm[i-1]:
            s_sm[i] = s_sm[i-1] + eps
    return s_sm