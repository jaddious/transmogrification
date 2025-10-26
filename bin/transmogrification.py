import numpy as np
from shapely.geometry import Point, LineString, Polygon


def _centerline_segments(centerline):
    C = np.asarray(centerline.coords, float)
    seg = C[1:] - C[:-1]
    L = np.hypot(seg[:, 0], seg[:, 1])
    S = np.concatenate([[0], np.cumsum(L)])
    seg_unit = seg / L[:, None]
    return C, seg, L, S, seg_unit


def _project_point(P, centerline, S, seg_unit):
    s = centerline.project(P, normalized=False)
    # segment index for station s
    i = np.searchsorted(S, s, side="left") - 1
    i = max(0, min(i, len(seg_unit) - 1))  # explicit clamp (protect endpoints)
    t = seg_unit[i]
    n = np.array([-t[1], t[0]])
    base = centerline.interpolate(s)
    offset = (P.x - base.x) * n[0] + (P.y - base.y) * n[1]
    return s, offset  # raw (not signed) so we can build A(s)


def _collect_s_samples(geoms, centerline, S, seg_unit, tol=1e-9):
    """
    Collect s for all coordinates across input geoms,
    excluding exact *interior* vertex stations S[1:-1] (endpoints kept).
    """
    Ss = []
    S_interior_set = set(np.round(S[1:-1], 9).tolist())

    def _maybe_add(s):
        rs = float(np.round(s, 9))
        if rs not in S_interior_set:  # keep endpoints; skip only interior vertex verticals
            Ss.append(s)

    for g in geoms:
        gt = g.geom_type
        if gt == "Point":
            s, _ = _project_point(g, centerline, S, seg_unit)
            _maybe_add(s)
        elif gt == "LineString":
            for x, y in g.coords:
                s, _ = _project_point(Point(x, y), centerline, S, seg_unit)
                _maybe_add(s)
        elif gt == "Polygon":
            for x, y in g.exterior.coords:
                s, _ = _project_point(Point(x, y), centerline, S, seg_unit)
                _maybe_add(s)
        else:
            # basic support for Multi*
            try:
                for p in g.geoms:
                    sub_ss = _collect_s_samples(
                        [p], centerline, S, seg_unit, tol)
                    Ss.extend(sub_ss)
            except Exception:
                pass
    return np.asarray(Ss, float)


def _build_horizontal_map(S_occ, clip_strategy="median"):
    """
    Build f(s) via clipping large gaps between occupied s's.
    Returns (s_support, s_star) so that s* = interp(s on support -> s_star).
    """
    if S_occ.size == 0:
        return None, None  # nothing to map

    # Sort and unique occupied s values
    s_support = np.unique(np.round(S_occ, 9))
    if s_support.size == 1:
        return s_support, np.array([0.0])  # collapse to a single column

    gaps = np.diff(s_support)
    positive = gaps[gaps > 0]

    if positive.size == 0:
        # degenerate: all identical after rounding
        return s_support, np.linspace(0.0, 0.0, s_support.size)

    if clip_strategy == "median":
        typical = float(np.median(positive))
    elif clip_strategy == "p90":
        typical = float(np.percentile(positive, 90))
    else:
        typical = float(np.median(positive))

    # Clip each gap to at most 'typical' -> compress big empty spans
    clipped = np.minimum(gaps, typical)
    s_star = np.concatenate([[0.0], np.cumsum(clipped)])

    # Make mapped support strictly monotone (avoid accidental x-collapses)
    min_dx = 1e-6
    for i in range(1, len(s_star)):
        if s_star[i] <= s_star[i-1] + min_dx:
            s_star[i] = s_star[i-1] + min_dx

    return s_support, s_star


def _apply_f_of_s(s, s_support, s_star):
    if s_support is None:
        return s  # identity if nothing built
    # Piecewise-linear interpolation, stays monotone.
    return float(np.interp(s, s_support, s_star))


def _keep_linesafe(coords, rescue=True):
    """
    Remove consecutive duplicate coords; ensure ≥2 coords.
    If still 1 coord and rescue=True, nudge last x by epsilon to survive as a line.
    """
    if not coords:
        return None
    dedup = [coords[0]]
    for q in coords[1:]:
        if q != dedup[-1]:
            dedup.append(q)
    if len(dedup) >= 2:
        ls = LineString(dedup)
        if ls.length > 0:
            return ls
    if rescue and len(dedup) == 1:
        x, y = dedup[0]
        return LineString([(x, y), (x + 1e-9, y)])
    return None


def straighten(geoms, centerline, clip_strategy="median", map=None):
    """
    Straighten with horizontal offset A(s) so there are no empty data spaces.
    Returns geometries in (x, y) = (-f(s), -n) coordinates (matching your signs).
    geoms: list/iterable of shapely geometries.
    """
    # Precompute centerline geometry helpers
    
    C, seg, L, S, seg_unit = _centerline_segments(centerline)
    
    # Sets for quick membership checks
    S_interior_set = set(np.round(S[1:-1], 9).tolist())  # endpoints excluded
    def is_interior(s): return float(np.round(s, 9)) in S_interior_set

    # 1) Collect occupied s samples (excluding *interior* vertex stations only)
    S_occ = _collect_s_samples(geoms, centerline, S, seg_unit)

    # 2) Build the monotone compression map f(s) -> s*
    s_support, s_star = map

    EPS_TIE = 1e-9  # tiny mapped-x nudge to prevent collapse on interior stations

    def project_point_with_A(P):
        s, offset = _project_point(P, centerline, S, seg_unit)
        s_mapped = _apply_f_of_s(s, s_support, s_star)
        # If this vertex lies exactly on an *interior* station, nudge mapped x slightly.
        if is_interior(s):
            s_mapped += EPS_TIE
        # Apply your sign convention: x = -f(s); y = -offset
        return (-s_mapped, -offset)

    def transform_geom(g):
        gt = g.geom_type
        if gt == "Point":
            return Point(project_point_with_A(g))

        elif gt == "LineString":
            coords = [project_point_with_A(Point(x, y)) for x, y in g.coords]
            ls = _keep_linesafe(coords)
            return ls if ls is not None else None

        elif gt == "Polygon":
            ext = [project_point_with_A(Point(x, y))
                   for x, y in g.exterior.coords]
            # Ensure polygon is valid (needs at least 3 coords and closed ring)
            if len(ext) >= 3:
                if ext[0] != ext[-1]:
                    ext.append(ext[0])
                try:
                    poly = Polygon(ext)
                    return poly if not poly.is_empty else None
                except Exception:
                    return None
            return None

        else:
            # Handle Multi* generically
            try:
                parts = [transform_geom(p) for p in g.geoms]
                parts = [p for p in parts if p is not None]
                if not parts:
                    return None
                return type(g)(parts)
            except Exception:
                return None

    return [transform_geom(g) for g in geoms]

