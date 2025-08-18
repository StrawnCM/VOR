# pip install numpy matplotlib shapely scipy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as SPoly, Point as SPoint, MultiPoint
from scipy.spatial import HalfspaceIntersection

# ---------- helpers ----------
def rect_polygon(w, h):
    return SPoly([(0,0),(w,0),(w,h),(0,h)])

def polygon_inward_halfspaces(poly):
    """Return halfspaces ax+by<=c for the polygon interior (robust to orientation)."""
    pts = np.array(poly.exterior.coords[:-1])
    cx, cy = np.array(poly.representative_point().coords[0])  # guaranteed inside
    H = []
    for i in range(len(pts)):
        p = pts[i]
        q = pts[(i+1)%len(pts)]
        v = q - p
        # right normal (not assuming CCW)
        n = np.array([v[1], -v[0]], dtype=float)
        a, b = n
        c = a*p[0] + b*p[1]
        # flip so that centroid satisfies ax+by<=c
        if a*cx + b*cy > c:
            a, b, c = -a, -b, -c
        H.append([a, b, c])
    return np.array(H, float)

def bisector_halfspaces(seeds, idx):
    """Halfspaces for points closer to seeds[idx] than any other seed."""
    p = np.array(seeds[idx], float)
    p2 = p @ p
    H = []
    for j, q in enumerate(seeds):
        if j == idx: 
            continue
        q = np.array(q, float)
        a, b = (q - p)                 # (q-p)·x <= (|q|^2 - |p|^2)/2
        c = 0.5 * ((q @ q) - p2)
        H.append([a, b, c])
    return np.array(H, float)

def convex_poly_from_halfspaces(H, interior, eps=1e-7):
    """
    Intersect 2D halfspaces a x + b y <= c.
    SciPy expects A x + b <= 0, i.e. [a, b, -c].
    Add tiny inward slack so the feasible point is strictly inside.
    """
    # shrink inward a hair for numerical safety
    nrm = np.linalg.norm(H[:, :2], axis=1)
    C = H[:, 2] - eps * nrm
    Aqh = np.c_[H[:,0], H[:,1], -C]      # <-- correct sign
    hs = HalfspaceIntersection(Aqh, np.array(interior, float))
    pts = hs.intersections
    if len(pts) < 3:
        return None
    hull = MultiPoint(pts).convex_hull
    return hull if isinstance(hull, SPoly) else None

def voronoi_cells_in_polygon(poly, seeds):
    base = polygon_inward_halfspaces(poly)
    cells = []
    for i, s in enumerate(seeds):
        H = np.vstack([base, bisector_halfspaces(seeds, i)])
        cells.append(convex_poly_from_halfspaces(H, s))
    return cells

def random_points_in_polygon(poly, n, rng):
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    for _ in range(2000):
        if len(pts) >= n: break
        p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(SPoint(p)):
            pts.append(p)
    return pts

def draw_poly(ax, poly, color, lw):
    c = np.array(poly.exterior.coords)
    ax.plot(c[:,0], c[:,1], color=color, lw=lw)

# ---------- recursion + plotting ----------
def subdivide(ax, poly, level, max_levels, rng, lw_by_level, color_by_level):
    if level > max_levels:
        return
    seeds = random_points_in_polygon(poly, 3, rng)
    if len(seeds) < 2:
        return
    for cell in voronoi_cells_in_polygon(poly, seeds):
        if cell is None or cell.area <= 1e-10:
            continue
        draw_poly(ax, cell, color_by_level[level], lw_by_level[level])
        subdivide(ax, cell, level+1, max_levels, rng, lw_by_level, color_by_level)

def hierarchical_voronoi(width=10, height=8, levels=4, seed=42):
    rng = np.random.default_rng(seed)
    rect = rect_polygon(width, height)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal'); ax.set_xlim(0, width); ax.set_ylim(0, height)
    ax.axis('off')

    # outer border
    draw_poly(ax, rect, 'black', 4.5)

    # styles
    lw = {1:3.8, 2:2.4, 3:1.4, 4:0.8}
    colors = {1:'black', 2:'navy', 3:'olive', 4:'firebrick'}

    subdivide(ax, rect, 1, levels, rng, lw, colors)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    hierarchical_voronoi()
