# pip install numpy matplotlib shapely scipy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as SPoly, Point as SPoint, MultiPoint, LineString
from scipy.spatial import HalfspaceIntersection
from matplotlib.animation import FuncAnimation

# ---------- geometry + halfspaces ----------
def rect_polygon(w, h):
    return SPoly([(0,0),(w,0),(w,h),(0,h)])

def polygon_inward_halfspaces(poly):
    """Return halfspaces ax+by<=c for polygon interior (robust to orientation)."""
    pts = np.array(poly.exterior.coords[:-1])
    cx, cy = np.array(poly.representative_point().coords[0])  # guaranteed inside
    H = []
    for i in range(len(pts)):
        p = pts[i]; q = pts[(i+1)%len(pts)]
        v = q - p
        n = np.array([v[1], -v[0]], dtype=float)  # right normal
        a, b = n; c = a*p[0] + b*p[1]
        if a*cx + b*cy > c:  # flip to be inward
            a, b, c = -a, -b, -c
        H.append([a, b, c])
    return np.array(H, float)

def bisector_halfspaces(seeds, idx):
    """Halfspaces for points closer to seeds[idx] than any other seed."""
    p = np.array(seeds[idx], float); p2 = p @ p
    H = []
    for j, q in enumerate(seeds):
        if j == idx: continue
        q = np.array(q, float)
        a, b = (q - p)                 # (q-p)·x <= (|q|^2 - |p|^2)/2
        c = 0.5 * ((q @ q) - p2)
        H.append([a, b, c])
    return np.array(H, float)

def convex_poly_from_halfspaces(H, interior, eps=1e-7):
    """
    Intersect 2D halfspaces a x + b y <= c.
    SciPy expects A x + b <= 0, i.e. [a, b, -c].
    """
    if H.size == 0: return None
    nrm = np.linalg.norm(H[:, :2], axis=1)
    C = H[:, 2] - eps * nrm  # shrink inward slightly
    Aqh = np.c_[H[:,0], H[:,1], -C]
    try:
        hs = HalfspaceIntersection(Aqh, np.array(interior, float), qhull_options='QJ')  # joggle for robustness
    except Exception:
        return None
    pts = hs.intersections
    if len(pts) < 3: return None
    hull = MultiPoint(pts).convex_hull
    return hull if isinstance(hull, SPoly) else None

def voronoi_cells_in_polygon(poly, seeds):
    base = polygon_inward_halfspaces(poly)
    cells = []
    for i, s in enumerate(seeds):
        H = np.vstack([base, bisector_halfspaces(seeds, i)])
        cells.append(convex_poly_from_halfspaces(H, s))
    return [c for c in cells if c is not None and c.area > 1e-12]

def random_points_in_polygon(poly, n, rng, max_tries=20000, min_sep=1e-6):
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(SPoint(p)):
            if not pts or np.min(np.linalg.norm(np.array(pts) - p, axis=1)) > min_sep:
                pts.append(p)
        tries += 1
    return pts

# ---------- hierarchy build (static, level = 3 leaves) ----------
def collect_leaves(poly, level, max_levels, rng, seeds_per_cell=3):
    """Recursively subdivide with Voronoi; return polygons at max_levels."""
    if level > max_levels: return []
    if level == max_levels:
        return [poly]
    seeds = random_points_in_polygon(poly, seeds_per_cell, rng)
    if len(seeds) < 2:
        return []
    leaves = []
    for cell in voronoi_cells_in_polygon(poly, seeds):
        leaves.extend(collect_leaves(cell, level+1, max_levels, rng, seeds_per_cell))
    return leaves

# ---------- polygon bouncing helpers ----------
def polygon_edges_with_inward_normals(poly):
    """Return list of (p,q,n_hat) for each edge with inward unit normal."""
    pts = np.array(poly.exterior.coords[:-1], float)
    cx, cy = np.array(poly.representative_point().coords[0])
    edges = []
    for i in range(len(pts)):
        p = pts[i]; q = pts[(i+1) % len(pts)]
        v = q - p
        n = np.array([v[1], -v[0]], float)  # right normal
        a, b = n; c = a*p[0] + b*p[1]
        # flip if the representative interior point is not inside
        if a*cx + b*cy > c:
            n = -n
        nrm = np.linalg.norm(n)
        if nrm == 0: continue
        edges.append((p, q, n / nrm))
    return edges

def first_edge_hit(p, p_next, edges, tol=1e-12):
    """Return (hit_point, edge_idx, t_seg) for segment p->p_next; t_seg in [0,1]."""
    seg = LineString([tuple(p), tuple(p_next)])
    best = None
    for i, (a, b, _) in enumerate(edges):
        eln = LineString([tuple(a), tuple(b)])
        inter = seg.intersection(eln)
        if inter.is_empty: continue
        if inter.geom_type == 'Point':
            ip = np.array(inter.coords[0], float)
            # parameter t along p->p_next
            full = np.linalg.norm(p_next - p)
            if full < tol: t = 0.0
            else: t = np.linalg.norm(ip - p) / full
            if t < -tol or t > 1+tol: continue
            if best is None or t < best[2]:
                best = (ip, i, max(0.0, min(1.0, t)))
    return best  # or None

def step_bounce_in_polygon(p, v, poly, edges, max_reflects=2, eps_in=1e-9):
    """
    Try to move from p by vector v inside poly.
    Reflect off boundary edges (specular) if needed.
    Returns (new_p, new_v).
    """
    p_curr = p.copy()
    v_curr = v.copy()
    for _ in range(max_reflects + 1):
        p_next = p_curr + v_curr
        if poly.contains(SPoint(p_next)):
            return p_next, v_curr
        hit = first_edge_hit(p_curr, p_next, edges)
        if hit is None:
            # couldn't resolve intersection; nudge inward and stop
            return p_curr, -v_curr
        ip, idx, t = hit
        # reflect velocity around inward normal of hit edge
        n_hat = edges[idx][2]
        v_ref = v_curr - 2.0 * np.dot(v_curr, n_hat) * n_hat
        # remaining distance after hit
        rem = (1.0 - t) * np.linalg.norm(v_curr)
        if rem <= 0:
            return ip + n_hat * eps_in, v_ref
        dir_ref = v_ref / (np.linalg.norm(v_ref) + 1e-15)
        p_curr = ip + n_hat * eps_in  # push slightly inside
        v_curr = dir_ref * rem
    return p_curr, v_curr

# ---------- animation ----------
def animate_hierarchical(width=10, height=8, max_levels=3, leaf_seed_count=3, speed=0.05, seed=42):
    rng = np.random.default_rng(seed)
    rect = rect_polygon(width, height)

    # Build the level-3 leaves once
    leaves = collect_leaves(rect, level=1, max_levels=max_levels, rng=rng, seeds_per_cell=3)
    # For stability, drop vanishing leaves
    leaves = [L for L in leaves if isinstance(L, SPoly) and L.area > 1e-9]

    # Initialize seeds/velocities inside each leaf
    leaf_data = []  # list of dicts per leaf
    for L in leaves:
        seeds = np.array(random_points_in_polygon(L, leaf_seed_count, rng))
        if len(seeds) < 2:
            # ensure at least 2 to get cells; if not, reseed from centroid jitter
            c = np.array(L.representative_point().coords[0])
            jitter = rng.normal(scale=1e-3, size=(leaf_seed_count,2))
            seeds = np.tile(c, (leaf_seed_count,1)) + jitter
        vel = rng.uniform(-1, 1, size=(len(seeds), 2))
        vel = (vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)) * speed
        edges = polygon_edges_with_inward_normals(L)
        leaf_data.append({
            "poly": L,
            "edges": edges,
            "seeds": seeds,
            "vel": vel,
            "artists": [],  # line artists for this leaf
            "scat": None,   # scatter for seed dots (optional)
        })

    # Matplotlib setup
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal'); ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis('off')

    # draw outer border once
    c = np.array(rect.exterior.coords); ax.plot(c[:,0], c[:,1], color='black', lw=3.0)

    # seed scatters per leaf (optional for debugging / aesthetics)
    for ld in leaf_data:
        s = ld["seeds"]
        ld["scat"] = ax.scatter(s[:,0], s[:,1], s=8, color='black', alpha=0.6)

    # initial draw of cells
    def draw_leaf_cells(ld):
        # remove old artists
        for ln in ld["artists"]:
            ln.remove()
        ld["artists"].clear()

        # compute cells for this leaf
        cells = voronoi_cells_in_polygon(ld["poly"], ld["seeds"])
        for cell in cells:
            coords = np.array(cell.exterior.coords)
            (ln,) = ax.plot(coords[:,0], coords[:,1], lw=1.2, color='navy')
            ld["artists"].append(ln)

    for ld in leaf_data:
        draw_leaf_cells(ld)

    def update(_frame):
        for ld in leaf_data:
            L = ld["poly"]; edges = ld["edges"]
            seeds = ld["seeds"]; vel = ld["vel"]

            # bounce each seed
            for i in range(len(seeds)):
                p_new, v_new = step_bounce_in_polygon(seeds[i], vel[i], L, edges)
                seeds[i] = p_new
                vel[i] = v_new
            ld["seeds"] = seeds
            ld["vel"] = vel

            # update scatter
            if ld["scat"] is not None:
                ld["scat"].set_offsets(ld["seeds"])

            # redraw Voronoi cells
            draw_leaf_cells(ld)

        # return updated artists (FuncAnimation doesn't require strict blit here)
        artists = []
        for ld in leaf_data:
            artists.extend(ld["artists"])
            if ld["scat"] is not None:
                artists.append(ld["scat"])
        return artists

    ani = FuncAnimation(fig, update, frames=800, interval=16, blit=False)
    plt.tight_layout(); plt.show()

# ---- entry point ----
if __name__ == "__main__":
    # Outer rectangle stays fixed; depth limited to 3 as requested
    animate_hierarchical(width=10, height=8, max_levels=3, leaf_seed_count=3, speed=0.06, seed=123)
