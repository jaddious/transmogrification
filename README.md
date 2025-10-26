# transmogrify.py — Corridor UV Mapping (Optimized)

A small, NumPy/Shapely–accelerated toolkit to “straighten” geometry inside a corridor. It defines a local coordinate system **(u, v)** along a **centerline** and between **bottom** and **top** boundaries, then maps between **XY ↔︎ UV** while preserving longitudinal stationing and cross‐sectional interpolation.

> **TL;DR**
>
> * `map_xy_to_uv(P, center, top, bottom)` converts a point in world **XY** to local **(u, v)**.
> * `map_uv_to_xy(u, v, center, top, bottom)` converts back.
> * `straighten([...], center, top, bottom)` transforms entire geometries (Points/Lines/Polygons/Multi*).
> * Under the hood: vectorized arclength parameterization, fast segment projection, and per-vertex stationing with smoothed monotonicity for robust inversion.

---

## Why this exists

When you want to analyze or edit features **along** a long, curvy corridor (streets, rivers, utility easements), it’s useful to work in a “straightened” frame: **u** measures distance along the corridor and **v** measures the cross-section from bottom → top. This module provides a numerically stable, high‑performance version of that mapping.

---

## Installation

```bash
pip install shapely numpy
```

---

## Quick start

```python
from shapely.geometry import LineString, Point
from transmogrify import map_xy_to_uv, map_uv_to_xy, straighten

# Define corridor
center = LineString([(0,0), (10,0), (20,5), (30,5)])
bottom = LineString([(0,-2), (10,-2), (20, 3), (30, 3)])
top    = LineString([(0, 2), (10, 2), (20, 7), (30, 7)])

# Point ↔︎ UV
P = Point(14.0, 2.0)
u, v = map_xy_to_uv(P, center, top, bottom)
# NOTE: u is returned with an external scaling; see “Coordinate conventions & scaling”.

P_back = map_uv_to_xy(u, v, center, top, bottom)

# Whole geometry
line_xy = LineString([(5, 1), (12, 2), (18, 4)])
line_uv = straighten([line_xy], center, top, bottom)[0]
```

---

## Coordinate conventions & scaling (important)

* **Internal** normalized longitudinal parameter: (\tilde u \in [0,1]) is the arclength fraction along the **centerline**.
* **Public API:** `map_xy_to_uv` currently **returns** `(u, v)` as
  [
  u_{public} = -8, \tilde u, \qquad v_{public} = v\in[0,1].
  ]
  This preserves a legacy/external scaling. As a consequence, `u_public` is **negative** and outside ([0,1]).
* `map_uv_to_xy` **expects** an input `u` and clamps it to ([0,1]) before evaluating. If you pass the raw `u_public` from `map_xy_to_uv` back into `map_uv_to_xy`, it will clamp to `0` and you will **lose invertibility**.

**Recommendation**

If you need round‑tripping, convert explicitly:

```python
# After map_xy_to_uv
u_pub, v = map_xy_to_uv(P, center, top, bottom)
# Convert back to internal normalized u in [0,1]
u_tilde = -u_pub / 8.0

P_back = map_uv_to_xy(u_tilde, v, center, top, bottom)
```

If you control the API, consider removing the external scaling (return `\tilde u` directly) to make the maps involutive up to numerical tolerances.

---

## Core concepts

### 1) Arclength parameterization of a polyline

Given a `LineString` with vertices (\mathbf p_0, \dots, \mathbf p_N), segment vectors (\mathbf v_i = \mathbf p_{i+1}-\mathbf p_i), and lengths (\ell_i = \lVert \mathbf v_i \rVert), we build cumulative lengths
[
L_0=0,\quad L_k=\sum_{i=0}^{k-1} \ell_i,\quad L= L_{N}.
]
For a normalized parameter (\tilde u\in[0,1]), the target arclength is (s= \tilde u,L). We locate the segment (j) such that (L_j \le s \le L_{j+1}), then linearly interpolate:
[
\mathbf x(s)= \mathbf p_j + w, \mathbf v_j,\quad w=\frac{s-L_j}{\max(L_{j+1}-L_j,,\varepsilon)}.
]
This is what `_Poly.point_at_u` implements.

### 2) Fast orthogonal projection of a point to a polyline

For each segment ([\mathbf A,\mathbf B]) with (\mathbf{AB}=\mathbf B-\mathbf A), the unconstrained parameter is
[
t = \frac{(\mathbf P-\mathbf A)\cdot \mathbf{AB}}{\mathbf{AB}\cdot \mathbf{AB}}.
]
Clamp (t) to ([0,1]), compute the closest point (\mathbf C=\mathbf A+t,\mathbf{AB}), and squared distance (\lVert \mathbf P-\mathbf C\rVert^2). Pick the segment with minimal distance, convert its station to (\tilde u=s/L). This is vectorized in `_Poly.project_u`.

### 3) Corridor framing at a station

At any (\tilde u), we need the cross‑section tuple ((\mathbf B, \mathbf S, \mathbf T)): bottom, center, top at the **same longitudinal station**. The center (\mathbf S) is direct via arclength.

For top/bottom we must **invert**: given a station (s) on the centerline, find the corresponding location on the boundary polyline that “aligns” with that station. We precompute, for each boundary vertex (k), the projected centerline station (s_k) using `_Poly.project_u`. Those arrays ({s_k}) are then made weakly monotone (cumulative max) and **lightly smoothed** with a 5‑tap triangular FIR filter; finally strict monotonicity is enforced by nudging ties with an (\varepsilon). During queries, we binary‑search (s_k) to bracket (s) and linearly interpolate boundary vertices to get (\mathbf B(s)) or (\mathbf T(s)).

### 4) The UV ↔︎ XY maps

**Forward (u, v → XY)**

* Clamp (\tilde u\in[0,1]), compute (\mathbf S, \mathbf B, \mathbf T) at the same station.
* Piecewise linear cross‑section: if (v\le 1/2) interpolate (\mathbf B\to\mathbf S); else (\mathbf S\to\mathbf T):
  [
  \mathbf X(\tilde u,v)=\begin{cases}
  (1-2v),\mathbf B + (2v),\mathbf S, & v\le 1/2,\
  (1-2(v-1/2)),\mathbf S + (2(v-1/2)),\mathbf T, & v>1/2.
  \end{cases}
  ]

**Inverse (XY → u, v)**

1. Project (\mathbf P) to the centerline to get (\tilde u).
2. Evaluate (\mathbf B,\mathbf S,\mathbf T) at that station.
3. Project (\mathbf P) onto segments ([\mathbf B,\mathbf S]) and ([\mathbf S,\mathbf T]). Choose the closer branch and convert the segment parameter to (v) as
   (v=\tfrac12 t_{BS}) or (v=\tfrac12+\tfrac12 t_{ST}).
4. Handle degenerate widths ((\lVert\mathbf S-\mathbf B\rVert) and (\lVert\mathbf T-\mathbf S\rVert) tiny) by returning (v=0.5).

---

## Numerical robustness & preprocessing

* **Resampling boundaries:** Before projection, top/bottom are resampled to roughly uniform arclength spacing with step `~ 0.5%` of centerline length (min `1e-6`). This improves station smoothness and the quality of binary‑search inversion.
* **Monotonicity & smoothing:** Vertex stations may regress due to geometry; we enforce monotonicity with a cumulative max, then apply a gentle `[1,2,3,2,1]/9` FIR, and finally enforce **strict** increase by adding tiny (\varepsilon) offsets to kill plateaus. This stabilizes interpolation.
* **Epsilons:** `_EPS=1e-12` for geometric checks; additional thresholds guard near‑degenerate segments and widths.

---

## Performance characteristics

* `_Poly` caches vectors, segment lengths, cumulative arclength; point sampling is **O(1)** after a `searchsorted`, projection is **O(M)** over segments but vectorized in NumPy.
* Corridor context (`_CorridorCtx`) precomputes per‑vertex stations once and is **identity‑cached** with weak references so repeated calls for the same `LineString` objects are fast and don’t block GC.
* Query path (per point): 1 centerline projection + 2 short segment projections + 2 binary searches.

---

## API reference (public)

### `map_xy_to_uv(P, centerline, top_line, bottom_line) -> (u, v)`

Converts a point in XY to corridor coordinates.

* Returns **`u` with external scaling** (see “Coordinate conventions”).
* `v∈[0,1]`: 0 at bottom, 0.5 at center, 1 at top.

### `map_uv_to_xy(u, v, centerline, top_line, bottom_line) -> Point`

Converts local (u, v) to XY.

* **Expects** `u` in ([0,1]); clamps otherwise.
* To invert `map_xy_to_uv`, first rescale `u_public → \tilde u`.

### `straighten(geoms, centerline, top_line, bottom_line) -> list`

Vectorizes `map_xy_to_uv` over a list of Shapely geometries. Returns corresponding geometries in UV space.

### `to_uv_geometry(geom, ...)` / `from_uv_geometry(geom_uv, ...)`

Convert a single geometry **to** or **from** the corridor frame. Handles `Point`, `LineString`, `Polygon`, and Multi* variants. Rings are closed if needed.

---

## Known quirks & footguns

1. **External `u` scaling**: `map_xy_to_uv` returns `u_public = -8·\tilde u`, but `map_uv_to_xy` clamps its input to `[0,1]`. Avoid passing the public `u` back into `map_uv_to_xy` without rescaling. Consider removing the scaling for a cleaner API.
2. **Duplicate helper definitions**: `_resample_line` and `_smooth_monotone_s` appear twice. Prefer the first definitions; consider removing the duplicates to avoid import-time shadowing and confusion.
3. **Object‑identity cache**: The corridor cache keys by the **identity** of the `LineString` objects, not value-equality. If you recreate identical geometries, you won’t hit the cache.
4. **Degenerate lines**: A `LineString` with one vertex is normalized to two identical points; projections collapse to that point.

---

## Testing guidelines

* **Round‑trip**: Sample random points in the corridor, transform XY→UV (rescale u)→XY and assert sub‑millimetre error within your units.
* **Monotonicity**: For random polyline pairs, verify that precomputed vertex stations `s_k` are strictly increasing after smoothing.
* **Edge cases**: Extremely narrow cross sections; zero-length segments; very short centerlines; cusps.
* **Performance**: Benchmark batched projections (≥ 10⁴ points). Expect vectorized speed‑ups vs. naïve Shapely ops.

---

## Complexity (rough)

* Precompute `_CorridorCtx`: (O(N_c + N_t + N_b)) with vectorized passes.
* Query `map_xy_to_uv`: one (O(N_c)) projection plus constant‑time local ops.
* Query `map_uv_to_xy`: two binary searches over boundary vertex stations ((O(\log N))) plus constant‑time blends.

---

## Design notes

* **Arclength‑consistent pairing** of top/bottom with the centerline avoids the drift you get from naïve parameterizations by vertex index.
* **Uniform resampling** stabilizes inversion and removes sensitivity to heterogeneous vertex spacing.
* **Tiny FIR smoothing** keeps the map smooth without overfitting across inflections.
* **No reliance on buffered offsets** (which can self‑intersect on curvature spikes); this is purely projection‑based.

---

## FAQ

**Q: Is the mapping conformal or area‑preserving?**
*A:* No. It is a piecewise‑linear cross‑sectional warp anchored to the centerline stationing. It preserves station order and interpolates linearly across width.

**Q: Can I use splines instead of piecewise linear frames?**
*A:* Yes—replace boundary interpolation with a spline over `(s_k, points_k)`; keep the station arrays monotone.

**Q: What if the corridor boundaries cross or fold back?**
*A:* The monotone station enforcement mitigates backtracking artifacts, but if the boundary truly folds w.r.t. the centerline, the local cross‑section may not be one‑to‑one.

---

## Changelog (this optimized version)

* Vectorized segment projection across the whole polyline (`_Poly.project_u`).
* Pre‑resample top/bottom to uniform arclength before stationing.
* Weak monotonicity + FIR smoothing + strict monotonicity for vertex stations.
* Identity‑based weakref cache for corridor contexts.
* Robust edge handling (degenerate center or zero‑width frames).

---

## License

MIT