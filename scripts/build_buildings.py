# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mapbox-earcut", "pillow", "scipy", "shapely", "tifffile", "imagecodecs", "scikit-image"]
# ///
"""Turn 3D BAG LoD2.2 solids into one triangle soup for the renderer.

Output (data/build/):
  buildings.bin   float32 per vertex: x, y, z, nx, ny, nz, u, v, kind, building index, atlas u, atlas v (-1: none)
  roof_atlas.png  aerial photo crops for the roofs
  buildings.json  per building: id, attributes, wall and roof colors
  footprints.json per building: footprint ring in local meters, for the ground and tree steps

Local coordinates: x east, y up, z south, origin at the area center, y=0 at the building's ground level.
"""

import hashlib
import json
from collections import defaultdict

import mapbox_earcut as earcut
import numpy as np
from PIL import Image, ImageDraw

import roof_atlas
from aerial import load_aerial, world_to_px
from config import BUILD, CX, CY, ERA_CONF, RAW, XMAX, XMIN, YMAX, YMIN, write_area
from era import built_after

KIND = {"GroundSurface": 0, "WallSurface": 1, "RoofSurface": 2}
COLS = 12

# Roof vertex blocks and footprints per building index, for texturing roofs from the aerial photo afterwards.
ROOF_BLOCKS = defaultdict(list)
FOOTPRINT = {}
ATLAS_KINDS = {"house", "barn", "shed", "church", "school", "tank", "silo", "canopy", "factory", "stoer"}

# BGT structures that are often missing from the BAG: (height, type).
EXTRAS = {
    "opslagtank": (4.5, "tank"),
    "voedersilo": (11.0, "silo"),
    "schuur": (2.6, "shed"),
    "open loods": (5.0, "shed"),
    "overkapping": (3.0, "canopy"),
    "lage trafo": (2.2, "shed"),
}


def ring_area(xy):
    return 0.5 * (np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))


def extrude(rings_xy, z0, z1):
    """Prism from a footprint (rings in RD x/y, first exterior). Yields (kind, rings in 3D)."""
    out = []
    for i, r in enumerate(rings_xy):
        r = np.asarray(r, dtype=np.float64)[:, :2]
        if np.allclose(r[0], r[-1], rtol=0, atol=1e-6):  # absolute: RD meters are large numbers
            r = r[:-1]
        # Exterior counter-clockwise, holes clockwise, so wall normals point outward.
        if (ring_area(r) > 0) != (i == 0):
            r = r[::-1]
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            if np.linalg.norm(b - a) < 0.05:
                continue
            out.append((1, [np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]])]))
    top = []
    for i, r in enumerate(rings_xy):
        r = np.asarray(r, dtype=np.float64)[:, :2]
        if np.allclose(r[0], r[-1], rtol=0, atol=1e-6):  # absolute: RD meters are large numbers
            r = r[:-1]
        if (ring_area(r) > 0) != (i == 0):
            r = r[::-1]
        top.append(np.column_stack([r, np.full(len(r), z1)]))
    out.append((2, top))
    return out


def long_axis(xy):
    """Direction (radians, RD x/y) of the longest footprint edge."""
    xy = np.asarray(xy)[:, :2]
    d = np.roll(xy, -1, axis=0) - xy
    i = int(np.argmax(np.hypot(d[:, 0], d[:, 1])))
    return float(np.arctan2(d[i, 1], d[i, 0]))


def newell_normal(pts):
    n = np.zeros(3)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else None


def triangulate(rings):
    """rings: list of (N,3) arrays in RD meters, first is the exterior. Returns (M,3,3) triangles."""
    n = newell_normal(rings[0])
    if n is None:
        return np.zeros((0, 3, 3)), None
    drop = int(np.argmax(np.abs(n)))
    keep = [i for i in range(3) if i != drop]
    pts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(pts[:, keep], ends)
    tris = pts[idx].reshape(-1, 3, 3)
    # earcut does not keep the winding; flip triangles that face away from the surface normal.
    tn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = (tn @ n) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris, n


def stable_rand(key, salt=""):
    return int(hashlib.md5((key + salt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def emit(verts, surfaces, bi, ground, center_u=False):
    """surfaces: iterable of (kind, rings in RD meters with absolute z). Appends vertices; returns roof triangles.

    center_u puts u=0 in the middle of each wall, so towers can draw a clock or louvres centered on a face."""
    roof_tris = []
    for kind, rings in surfaces:
        if kind == 0:
            continue  # the floor is never visible
        tris, n = triangulate(rings)
        if n is None or len(tris) == 0:
            continue
        if kind == 2:
            roof_tris.append(tris)
        # Wall pattern coordinates: u runs along the wall, v is height above ground.
        if kind == 1:
            h = np.array([n[1], -n[0]])
            h = h / (np.linalg.norm(h) or 1)
            u = tris[..., :2] @ h
            if center_u:
                u = u - 0.5 * (u.min() + u.max())
            uv = np.stack([u, tris[..., 2] - ground], axis=-1)
        else:
            uv = tris[..., :2] - [CX, CY]
        local = np.stack([tris[..., 0] - CX, tris[..., 2] - ground, CY - tris[..., 1]], axis=-1)
        normal = [n[0], n[2], -n[1]]
        block = np.concatenate(
            [local, np.broadcast_to(normal, local.shape), uv, np.full(local.shape[:2] + (1,), kind), np.full(local.shape[:2] + (1,), bi),
             np.full(local.shape[:2] + (2,), -1.0)],
            axis=-1,
        ).reshape(-1, COLS)
        verts.append(block)
        if kind == 2:
            ROOF_BLOCKS[bi].append((block, tris))
    return roof_tris


def ccw(ring):
    ring = np.asarray(ring, dtype=np.float64)
    return ring if ring_area(ring) > 0 else ring[::-1]


def rect_ring(cx, cy, w, d, ang):
    c, s = np.cos(ang), np.sin(ang)
    pts = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    return ccw([(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts])


def ngon_ring(cx, cy, r, n, ang):
    return ccw([(cx + r * np.cos(ang + 2 * np.pi * (i + 0.5) / n), cy + r * np.sin(ang + 2 * np.pi * (i + 0.5) / n)) for i in range(n)])


def pyramid(ring, z0, apex):
    """Roof triangles from a footprint ring up to a point above its center."""
    ring = ccw(ring)
    cx, cy = ring.mean(0)
    return [(2, [np.array([[a[0], a[1], z0], [b[0], b[1], z0], [cx, cy, apex]])]) for a, b in zip(ring, np.roll(ring, -1, axis=0))]


def frustum(bottom, top, z0, z1):
    """Walls between two rings with the same vertex count, for tapered masts."""
    bottom, top = ccw(bottom), ccw(top)
    out = []
    for i in range(len(bottom)):
        j = (i + 1) % len(bottom)
        out.append((1, [np.array([[*bottom[i], z0], [*bottom[j], z0], [*top[j], z1], [*top[i], z1]])]))
    out.append((2, [np.column_stack([top, np.full(len(top), z1)])]))
    return out


def part(meta, pid, kind, axis, height, wall, roof, param=0.0):
    meta.append({"id": f"{pid}-{kind}", "kind": kind, "axis": round(float(axis), 4), "param": param, "height": height,
                 "wall_fixed": list(wall), "roof_fixed": list(roof), "rand": 0.5, "roof_rgb": None})
    return len(meta) - 1


def add_landmark_parts(verts, meta, pid, lm, ground, axis, roof_tris):
    """Towers, roof turrets and crosses that 3D BAG smooths away."""
    wall, roof = lm.get("wall", (150, 80, 62)), lm.get("roof", (70, 72, 80))
    if "tower" in lm:
        t = lm["tower"]
        ring = rect_ring(t["x"], t["y"], t["w"], t["w"], axis)
        bi = part(meta, pid, "tower", axis, t["top"], wall, roof, param=t["body"])
        emit(verts, extrude([ring], ground, ground + t["body"]), bi, ground, center_u=True)
        emit(verts, pyramid(ring, ground + t["body"], ground + t["top"]), bi, ground)
        if t.get("cross"):
            bi = part(meta, pid, "cross", axis, t["top"] + 2, (196, 170, 96), (196, 170, 96))
            top = ground + t["top"]
            emit(verts, extrude([rect_ring(t["x"], t["y"], 0.3, 0.3, axis)], top - 0.4, top + 2.2), bi, ground)
            emit(verts, extrude([rect_ring(t["x"], t["y"], 1.4, 0.3, axis)], top + 1.2, top + 1.5), bi, ground)
    if "turret" in lm:
        t = lm["turret"]
        ring = ngon_ring(t["x"], t["y"], t["r"], 8, axis)
        bi = part(meta, pid, "turret", axis, t["top"], (240, 238, 230), (62, 64, 72), param=t["body"])
        emit(verts, extrude([ring], ground + t["base"], ground + t["body"]), bi, ground, center_u=True)
        emit(verts, pyramid(ring, ground + t["body"], ground + t["top"]), bi, ground)
    if lm.get("bell") and roof_tris:
        # A small white bell turret on the highest point of the roof, as on the school.
        pts = np.concatenate(roof_tris).reshape(-1, 3)
        x, y, zmax = pts[np.argmax(pts[:, 2])]
        ring = rect_ring(x, y, 1.5, 1.5, axis)
        bi = part(meta, pid, "turret", axis, zmax - ground + 3.4, (240, 238, 230), (70, 74, 82), param=zmax - ground + 2.0)
        emit(verts, extrude([ring], zmax - 0.4, zmax + 2.0), bi, ground, center_u=True)
        emit(verts, pyramid(ring, zmax + 2.0, zmax + 3.4), bi, ground)


def add_masts(verts, meta, masts):
    for m in masts:
        bot = rect_ring(m["x"], m["y"], m["base"], m["base"], 0.3)
        top = rect_ring(m["x"], m["y"], m["tip"], m["tip"], 0.3)
        bi = part(meta, "mast", "mast", 0.3, m["height"], (184, 188, 192), (150, 154, 158), param=m["height"])
        emit(verts, frustum(bot, top, 0.0, m["height"]), bi, 0.0, center_u=True)


def add_chimneys(verts, meta, footprints, chimneys):
    """Round brick chimneys, tapering to the top, with a dark opening."""
    for c in chimneys:
        bot = ngon_ring(c["x"], c["y"], c["r0"], 16, 0.0)
        top = ngon_ring(c["x"], c["y"], c["r1"], 16, 0.0)
        bi = part(meta, f"chimney-{c['x']:.0f}", "chimney", 0.0, c["height"], c["wall"], (44, 38, 36), param=c["height"])
        emit(verts, frustum(bot, top, 0.0, c["height"]), bi, 0.0, center_u=True)
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in bot])


def on_stilts(items, ground, st):
    """A flat-roofed building rebuilt from its roof parts, each a block from the ground up to its roof, except the
    parts whose roof lies between `lift` meters: those float, open below `under`, on columns at their corners. For the
    old Enka canteen, whose end blocks overhang on columns."""
    from shapely.geometry import Polygon

    out = []
    lo, hi = st["lift"]
    for kind, rings in items:
        if kind != 2:
            continue
        ring = rings[0]
        h = float(ring[:, 2].mean()) - ground
        poly = Polygon(ring[:, :2]).buffer(0)
        if poly.geom_type != "Polygon" or poly.area < 1:
            continue
        pts = np.asarray(poly.exterior.coords)
        if lo <= h <= hi:
            out += extrude([pts], ground + st["under"], ground + h)
            ang = long_axis(pts[:-1])
            c = np.asarray(poly.centroid.coords[0])
            for x, y in poly.minimum_rotated_rectangle.exterior.coords[:-1]:
                inward = (c - (x, y)) / (np.hypot(*(c - (x, y))) or 1)
                out += extrude([rect_ring(x + inward[0] * 0.7, y + inward[1] * 0.7, 0.45, 0.45, ang)], ground, ground + st["under"])
        else:
            out += extrude([pts], ground, ground + h)
    return out


def add_station(verts, meta, canopy, ring):
    """The station roof: folded grey metal panels, a field of triangles over the platforms with some of them glass
    skylights, on slender columns. Where `hall` cuts the canopy (an RD box), the station hall: a closed block with a
    flat green roof. And the tall slender tower beside it."""
    from shapely.geometry import Point, Polygon, box

    poly = Polygon(ring).buffer(0)
    H, fold = canopy["height"], canopy.get("fold", 1.4)
    hall = poly.intersection(box(*canopy["hall"])) if canopy.get("hall") else Polygon()
    roof = poly.difference(hall.buffer(0.01)) if not hall.is_empty else poly
    ang = long_axis(ring)
    du, dv = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
    c0 = np.asarray(roof.centroid.coords[0])
    pts = np.asarray(ring) - c0
    u0, u1 = (pts @ du).min() - 8, (pts @ du).max() + 8
    v0, v1 = (pts @ dv).min() - 8, (pts @ dv).max() + 8
    cell = canopy.get("cell", 6.0)
    # A lattice of triangles: every other row of points raised by `fold`, so each cell folds along its diagonal; the
    # roof steps down a little in bands along the tracks, as the real one does.
    nu, nv = int((u1 - u0) / cell) + 1, int((v1 - v0) / cell) + 1
    band = canopy.get("band", 60.0)

    def vertex(i, j):
        u, v = u0 + i * cell, v0 + j * cell
        xy = c0 + du * u + dv * v
        z = H + fold * ((i + j) % 2) - 0.6 * (int((u - u0) // band) % 2)
        return np.array([xy[0], xy[1], z])

    inner = roof.buffer(-0.5)
    metal, glass, edges = [], [], {}
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)
            for k, tri in enumerate(((a, b, c), (a, c, d)) if (i + j) % 2 == 0 else ((a, b, d), (b, c, d))):
                t = np.array(tri)
                if not inner.contains(Point(*t[:, :2].mean(0))):
                    continue
                if ring_area(t[:, :2]) < 0:
                    t = t[::-1]
                (glass if stable_rand(f"{i},{j},{k}") < canopy.get("skylights", 0.14) else metal).append((2, [t]))
                for p, q in zip(t, np.roll(t, -1, axis=0)):
                    key = tuple(sorted((tuple(np.round(p[:2], 2)), tuple(np.round(q[:2], 2)))))
                    edges.setdefault(key, []).append((p, q))
    # The fascia: wherever a triangle's edge has no neighbour, a thin grey band down from the fold.
    fascia = []
    for (p, q), *rest in edges.values():
        if rest:
            continue
        fascia.append((1, [np.array([(*q[:2], H - 0.7), (*p[:2], H - 0.7), p, q])]))
        fascia.append((1, [np.array([(*p[:2], H - 0.7), (*q[:2], H - 0.7), q, p])]))
    bi = part(meta, "station-roof", "station", ang, H + fold, canopy["edge"], canopy["roof"])
    emit(verts, metal + fascia, bi, 0.0)
    gi = part(meta, "station-skylights", "station", ang, H + fold, canopy["edge"], canopy["glass"])
    meta[gi]["skylight"] = True
    emit(verts, glass, gi, 0.0)
    footprint = np.asarray(roof.exterior.coords)[:-1] if roof.geom_type == "Polygon" else ring
    FOOTPRINT[bi] = footprint

    col = part(meta, "station-columns", "bridge", 0.0, H, canopy["columns"], canopy["columns"])
    cols = []
    for gx in np.arange(u0, u1, canopy["grid"]):
        for gy in np.arange(v0, v1, canopy["grid"] / 2):
            x, y = c0 + du * gx + dv * gy
            if roof.buffer(-2.0).contains(Point(x, y)):
                cols += extrude([ngon_ring(x, y, 0.3, 8, 0.0)], 0.0, H - 0.7)
    emit(verts, cols, col, 0.0)

    for part_ in getattr(hall, "geoms", [hall]):
        if part_.geom_type != "Polygon" or part_.area < 10:
            continue
        hi = part(meta, "station-hall", "station_hall", ang, canopy["hall_height"], canopy["hall_wall"], canopy["hall_roof"],
                  param=canopy["hall_height"])
        emit(verts, extrude([np.asarray(part_.exterior.coords)[:-1]], 0.0, canopy["hall_height"]), hi, 0.0)
    if canopy.get("tower"):
        t = canopy["tower"]
        bot = rect_ring(*t["at"], t["w"], t["w"] * 1.3, ang)
        top = rect_ring(*t["at"], t["w"] * 0.8, t["w"] * 1.05, ang)
        ti = part(meta, "station-tower", "station_tower", ang, t["h"], canopy["tower_colour"], canopy["tower_colour"], param=t["h"])
        emit(verts, frustum(bot, top, 0.0, t["h"]), ti, 0.0, center_u=True)


def add_sphere(verts, meta, pid, lm, x, y, n=18):
    """A ball of radius r whose top is at `top`, on a short ring of a plinth where it meets the ground."""
    r, top = lm["sphere"]["r"], lm["sphere"]["top"]
    cz = top - r
    bi = part(meta, pid, "dome", 0.0, top, lm["wall"], lm["roof"])
    meta[bi].update({"name": lm["name"], "photo": False})
    tris = []
    lats = np.linspace(np.pi / 2, -np.pi / 2, n // 2 + 1)
    lons = np.linspace(0, 2 * np.pi, n + 1)
    for a0, a1 in zip(lats, lats[1:]):
        if np.sin(a1) * r + cz < -0.01 and np.sin(a0) * r + cz < -0.01:
            continue  # under the ground
        for b0, b1 in zip(lons, lons[1:]):
            p = lambda a, b: (x + r * np.cos(a) * np.cos(b), y + r * np.cos(a) * np.sin(b), max(0.0, cz + r * np.sin(a)))
            q = [p(a0, b0), p(a1, b0), p(a1, b1), p(a0, b1)]
            # The upper half is roof (lit like a roof), the lower half wall.
            kind = 2 if a0 > 0.2 else 1
            tris.append((kind, [np.array(q[::-1])]))
    emit(verts, tris, bi, 0.0)


def add_monuments(verts, meta, footprints, monuments):
    """Monuments from landmarks.json: the Mausoleum's brick wall, with the bronze statue on top at its north end and
    the flagpole beside its south end."""
    for m in monuments:
        ring = ccw(np.asarray(m["ring"], dtype=np.float64))
        ang = long_axis(ring)
        bi = part(meta, f"monument-{m['kind']}", m["kind"], ang, m["h"], (138, 70, 52), (110, 62, 50), param=m["h"])
        emit(verts, extrude([ring], 0.0, m["h"]), bi, 0.0)
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in ring])
        d = np.array([np.cos(ang), np.sin(ang)])
        c = ring.mean(0)
        along = (ring - c) @ d
        ends = [c + d * along.max() * 0.75, c + d * along.min() * 0.75]
        north, south = sorted(ends, key=lambda p: -p[1])
        bronze = part(meta, f"monument-{m['kind']}-statue", "bridge", ang, m["h"] + 2.2, (72, 84, 70), (72, 84, 70))
        top = m["h"]
        emit(verts, extrude([rect_ring(*north, 1.2, 0.8, ang)], top, top + 0.5)
             + extrude([rect_ring(*north, 0.5, 0.45, ang)], top + 0.5, top + 1.7)
             + extrude([rect_ring(*north, 0.3, 0.3, ang)], top + 1.7, top + 2.1), bronze, 0.0)
        pole = part(meta, f"monument-{m['kind']}-pole", "bridge", 0.0, 9.0, (236, 236, 232), (236, 236, 232))
        foot = south + np.array([-d[1], d[0]]) * 3.0
        emit(verts, extrude([ngon_ring(*foot, 0.08, 6, 0.0)], 0.0, 9.0), pole, 0.0)


def add_new_buildings(verts, meta, footprints, new):
    """Buildings newer than 3D BAG and AHN: their BGT footprint (found by BAG id) extruded to an estimated height."""
    if not new:
        return
    bgt = {f["properties"].get("bag_pnd"): f["geometry"] for f in json.load(open(RAW / "bgt_pand.geojson"))["features"] if f["geometry"]}
    for b in new:
        g = bgt.get(b["bag"])
        if not g:
            print(f"{b['name']}: no BGT footprint for BAG {b['bag']}")
            continue
        rings = [np.asarray(r)[:, :2] for r in (g["coordinates"] if g["type"] == "Polygon" else g["coordinates"][0])]
        bi = part(meta, f"NL.IMBAG.Pand.{b['bag']}", b["style"], long_axis(rings[0]), b["h"], b["wall"], b["roof"], param=b["h"])
        meta[bi]["photo"] = False
        emit(verts, extrude(rings, 0.0, b["h"]), bi, 0.0)
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in rings[0]])


def add_walls(verts, meta, walls):
    """Free-standing factory walls from landmarks.json: a thin slab along each segment of the path, broken where an arch
    or gate is (`gaps`: RD x, y and width). render/scene.js cuts the window openings ("ruin")."""
    for w in walls:
        path = np.asarray(w["path"], dtype=np.float64)
        for a, b in zip(path, path[1:]):
            add_wall_segment(verts, meta, w, a, b)


def add_wall_segment(verts, meta, w, a, b):
    d = (b - a) / np.linalg.norm(b - a)
    length = np.linalg.norm(b - a)
    # Gaps that lie on this segment.
    cuts = sorted((float((np.asarray(g[:2]) - a) @ d), g[2]) for g in w.get("gaps", [])
                  if 0 < (np.asarray(g[:2]) - a) @ d < length and abs(d[0] * (g[1] - a[1]) - d[1] * (g[0] - a[0])) < 3)
    spans, s0 = [], 0.0
    for at, width in cuts:
        spans.append((s0, at - width / 2))
        s0 = at + width / 2
    spans.append((s0, length))
    ang = float(np.arctan2(d[1], d[0]))
    bi = part(meta, f"wall-{a[0]:.0f}-{a[1]:.0f}", "ruin", ang, w["h"], w["wall"], w["wall"], param=w["h"])
    items = []
    for s, e in spans:
        if e - s > 0.5:
            c = a + d * (s + e) / 2
            items += extrude([rect_ring(*c, e - s, w["thick"], ang)], 0.0, w["h"])
    # Over a gap, the arch: a lintel band across the top.
    for at, width in cuts:
        items += extrude([rect_ring(*(a + d * at), width + 0.2, w["thick"], ang)], w["h"] - 2.6, w["h"])
    emit(verts, items, bi, 0.0)


def add_dock(verts, meta, p):
    """A loading dock along a facade, from `from` to `to` (RD, with the building to the left): a raised concrete
    platform `depth` meters out at `deck` meters, and over it a steel canopy that reaches further out, at `roof` meters,
    on red struts from the wall every `bay` meters. Where the trains stopped at the Bitterzoutloods."""
    a, b = np.asarray(p["from"], dtype=np.float64), np.asarray(p["to"], dtype=np.float64)
    d = (b - a) / np.linalg.norm(b - a)
    out = np.array([d[1], -d[0]])  # to the right of from→to: away from the building
    length = float(np.linalg.norm(b - a))
    ang = float(np.arctan2(d[1], d[0]))
    mid = (a + b) / 2
    concrete, steel, red = (186, 184, 176), (150, 156, 162), (184, 30, 28)
    deck = part(meta, f"dock-{a[0]:.0f}", "bridge", ang, p["deck"], concrete, concrete)
    emit(verts, extrude([rect_ring(*(mid + out * p["depth"] / 2), length, p["depth"], ang)], 0.0, p["deck"]), deck, 0.0)
    roof = part(meta, f"dock-roof-{a[0]:.0f}", "bridge", ang, p["roof"], steel, steel)
    reach = p["depth"] + p["overhang"]
    emit(verts, extrude([rect_ring(*(mid + out * reach / 2), length, reach, ang)], p["roof"] - 0.35, p["roof"]), roof, 0.0)
    struts = part(meta, f"dock-struts-{a[0]:.0f}", "bridge", ang, p["roof"], red, red)
    items = []
    for s in np.arange(p["bay"] / 2, length, p["bay"]):
        foot = a + d * s
        # A brace from the wall up and out to under the canopy's edge, and the beam under the canopy back to the wall.
        items += beam(foot, p["roof"] - 1.9, foot + out * reach * 0.85, p["roof"] - 0.25, d, 0.2)
        items += beam(foot, p["roof"] - 0.25, foot + out * reach, p["roof"] - 0.25, d, 0.2)
    emit(verts, items, struts, 0.0)


def beam(p0, z0, p1, z1, side, w):
    """A square beam of width w from (p0, z0) to (p1, z1) in RD meters; `side` is a horizontal unit vector across it."""
    side = np.asarray(side) * w / 2
    up = np.array([0, 0, w / 2])
    a0, a1 = np.array([*p0, z0]), np.array([*p1, z1])
    s3 = np.array([*side, 0.0])
    corners = lambda c: [c - s3 - up, c + s3 - up, c + s3 + up, c - s3 + up]
    b, t = corners(a0), corners(a1)
    faces = [(1, [np.array([b[i], b[(i + 1) % 4], t[(i + 1) % 4], t[i]])]) for i in range(4)]
    return faces + [(1, [np.array(t)]), (1, [np.array(b[::-1])])]


def add_props(verts, meta, props):
    """Playground things from landmarks.json: a swing set (two A-frames and a beam, 3.6 m long) and an obstacle
    course (balance beams, stepping posts and a climbing frame round a ring)."""
    wood, steel = (150, 108, 70), (178, 52, 44)
    for p in props:
        x, y = p.get("at", p.get("from"))
        if p["kind"] == "swings":
            a = np.radians(p["angle"])
            d = np.array([np.cos(a), np.sin(a)])
            n = np.array([-d[1], d[0]])
            bi = part(meta, f"swings-{x:.0f}", "bridge", float(a), 2.4, steel, steel)
            items = []
            for s in (-1, 1):
                for k in (-1, 1):  # the legs of each A-frame splay out to the sides
                    foot = np.array([x, y]) + d * 1.9 * s + n * 0.9 * k
                    top = np.array([x, y]) + d * 1.9 * s
                    items += frustum(rect_ring(*foot, 0.14, 0.14, a), rect_ring(*top, 0.14, 0.14, a), 0.0, 2.3)
            items += extrude([rect_ring(x, y, 4.0, 0.16, a)], 2.25, 2.42)
            emit(verts, items, bi, 0.0)
        elif p["kind"] == "dock":
            add_dock(verts, meta, p)
        else:
            bi = part(meta, f"obstacles-{x:.0f}", "shed", 0.0, 1.6, wood, wood)
            meta[bi]["photo"] = False
            items, r = [], p["r"]
            for i in range(10):
                t = 2 * np.pi * i / 10
                cx, cy = x + r * np.cos(t), y + r * np.sin(t)
                if i % 3 == 0:  # a balance beam along the ring
                    items += extrude([rect_ring(cx, cy, 3.2, 0.25, t + np.pi / 2)], 0.3, 0.5)
                elif i % 3 == 1:  # stepping posts
                    for k in (-0.6, 0.0, 0.6):
                        items += extrude([ngon_ring(cx + k * np.cos(t + np.pi / 2), cy + k * np.sin(t + np.pi / 2), 0.2, 6, 0.0)], 0.0, 0.35 + 0.2 * (k + 0.6))
                else:  # a little climbing frame
                    items += extrude([rect_ring(cx, cy, 1.4, 1.4, t)], 1.3, 1.45)
                    for kx in (-0.6, 0.6):
                        for ky in (-0.6, 0.6):
                            items += extrude([rect_ring(cx + kx * np.cos(t) - ky * np.sin(t), cy + kx * np.sin(t) + ky * np.cos(t), 0.12, 0.12, t)], 0.0, 1.3)
            emit(verts, items, bi, 0.0)


def add_railings(verts, meta, play):
    """A railing 1.1 m high round every raised play area (a roof terrace), so nobody falls off."""
    for a in play:
        if a["h"] <= 0:
            continue
        ring = ccw(np.asarray(a["area"], dtype=np.float64))
        bi = part(meta, f"railing-{a['name']}", "railing", long_axis(ring), a["h"] + 1.1, (64, 70, 76), (64, 70, 76), param=a["h"] + 1.1)
        walls = []
        for p0, p1 in zip(ring, np.roll(ring, -1, axis=0)):
            walls.append((1, [np.array([(*p0, a["h"]), (*p1, a["h"]), (*p1, a["h"] + 1.1), (*p0, a["h"] + 1.1)])]))
            walls.append((1, [np.array([(*p1, a["h"]), (*p0, a["h"]), (*p0, a["h"] + 1.1), (*p1, a["h"] + 1.1)])]))
        emit(verts, walls, bi, a.get("ground", 0.0))


def ribbon(path, half):
    """Left and right edges of a polyline widened by `half` meters on each side, mitred at the bends."""
    p = np.asarray(path, dtype=np.float64)
    seg = np.diff(p, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    nrm = np.stack([-seg[:, 1], seg[:, 0]], axis=1)  # to the left in RD (y north)
    vn = np.vstack([nrm[:1], nrm[:-1] + nrm[1:], nrm[-1:]])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    cos = np.r_[1.0, np.einsum("ij,ij->i", vn[1:-1], nrm[1:]), 1.0]
    off = vn * (half / np.clip(cos, 0.5, 1.0))[:, None]
    return p + off, p - off


def add_bridges(verts, meta, bridges):
    """Bridges for cyclists and walkers: a deck that ramps up from the ground at both ends, railings on both sides,
    columns under it, open over the tracks."""
    tri = lambda a, b, c: [np.array([a, b, c])]
    for b in bridges:
        path, hs = np.asarray(b["path"], dtype=np.float64), np.asarray(b["heights"])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        s0, s1, H = cum[b["span"][0]], cum[b["span"][1]], b["height"]
        L, R = ribbon(path, b["width"] / 2)
        deck = part(meta, f"bridge-{b['name']}", "bridge", 0.0, H, b["wall"], b["deck"])
        rail = part(meta, f"bridge-{b['name']}-rail", "bridge", 0.0, H + 1.1, b["rail"], b["rail"])
        surf, rails, bank = [], [], []
        # "embankment": "end": from the end of the span on, the ramp lies on a bank of earth with sloping sides, not on columns.
        dike = b["span"][1] if b.get("embankment") == "end" else len(path)
        bank_bi = part(meta, f"bridge-{b['name']}-bank", "bridge", 0.0, H, (120, 150, 80), (120, 150, 80)) if dike < len(path) else None
        for i in range(len(path) - 1):
            h0, h1 = hs[i], hs[i + 1]
            l0, l1, r0, r1 = (*L[i], h0), (*L[i + 1], h1), (*R[i], h0), (*R[i + 1], h1)
            surf += [(2, tri(l0, r0, r1)), (2, tri(l0, r1, l1))]
            if i >= dike:
                # The bank under the deck: each side slopes out 1 m for every 1.1 m of height; the rail stays on the deck.
                for P, Q, a_i, c_i, ha, hc, sign in ((L, R, i + 1, i, h1, h0, 1), (R, L, i, i + 1, h0, h1, -1)):
                    out_dir = lambda k: (P[k] - Q[k]) / (np.linalg.norm(P[k] - Q[k]) or 1)
                    a, c = P[a_i], P[c_i]
                    a_o, c_o = a + out_dir(a_i) * ha / 1.1, c + out_dir(c_i) * hc / 1.1
                    if max(ha, hc) < 0.05:
                        continue
                    bank.append((1, [np.array([(*a_o, 0.0), (*c_o, 0.0), (*c, hc), (*a, ha)])]))
                    rails.append((1, [np.array([(*a, ha), (*c, hc), (*c, hc + 1.1), (*a, ha + 1.1)])]))
                continue
            # The edge of the deck, 0.8 m thick, down to the ground where the ramp is low. Each wall runs with its
            # outside on the right, as in extrude().
            for a, c, ha, hc in ((L[i + 1], L[i], h1, h0), (R[i], R[i + 1], h0, h1)):
                lo_a, lo_c = (0.0 if ha < 2.0 else ha - 0.8), (0.0 if hc < 2.0 else hc - 0.8)
                surf.append((1, [np.array([(*a, lo_a), (*c, lo_c), (*c, hc), (*a, ha)])]))
                rails.append((1, [np.array([(*a, ha), (*c, hc), (*c, hc + 1.1), (*a, ha + 1.1)])]))
        # Columns every 12 m where the deck is high, but none on the tracks.
        for s in np.arange(6, cum[-1], 12.0):
            h = float(np.interp(s, cum, hs))
            if h < 2.0 or s0 - 4 < s < s1 + 4 or (dike < len(path) and s > cum[dike] - 0.1):
                continue
            x, y = np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])
            surf += extrude([rect_ring(x, y, 0.8, 0.8, 0.0)], 0.0, h - 0.8)
        if bank:
            # The end of the bank against the abutment, seen from both sides.
            e = dike
            cap = [np.array([(*L[e], 0.0), (*R[e], 0.0), (*R[e], hs[e]), (*L[e], hs[e])])]
            bank += [(1, cap), (1, [cap[0][::-1]])]
            emit(verts, bank, bank_bi, 0.0)
        emit(verts, surf, deck, 0.0)
        emit(verts, rails, rail, 0.0)


def greenhouse_height(year):
    # Older Venlo greenhouses are lower; modern ones have a 5-6 m gutter.
    return 3.6 if (year or 0) < 1985 else 5.2


def main():
    write_area()
    aerial = load_aerial()
    landmarks = json.loads((BUILD / "landmarks.json").read_text()) if (BUILD / "landmarks.json").exists() else {}
    named = landmarks.get("buildings", {})
    verts = []
    meta = []
    footprints = []
    for line in open(RAW / "3dbag.jsonl"):
        feat = json.loads(line)
        t = feat["transform"]
        V = np.array(feat["vertices"], dtype=np.float64) * t["scale"] + t["translate"]
        parent_id = feat["id"]
        parent = feat["CityObjects"][parent_id]
        attrs = parent["attributes"]
        # A demolition permit doesn't take a building down: it stands until it is gone from the BAG.
        standing = ("Pand in gebruik", "Pand in gebruik (niet ingemeten)", "Verbouwing pand", "Sloopvergunning verleend")
        if attrs.get("status") not in standing or built_after(attrs):
            continue
        ground = attrs.get("b3_h_maaiveld") or 0.0
        lod0 = parent["geometry"][0]["boundaries"][0][0]
        fp = V[lod0][:, :2]
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in fp])

        if "sphere" in named.get(parent_id, {}):
            # A ball instead of the block 3D BAG makes of it (the Stingerbol).
            lm = named[parent_id]
            c = fp.mean(0)
            add_sphere(verts, meta, parent_id, lm, c[0], c[1])
            # The ball is wider than its footprint: keep the tree finder off all of it.
            footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in ngon_ring(c[0], c[1], lm["sphere"]["r"] + 1, 24, 0.0)])
            continue
        geom = None
        for child_id in parent.get("children", []):
            for g in feat["CityObjects"][child_id].get("geometry", []):
                if g["lod"] == "2.2":
                    geom = g
        bi = len(meta)
        FOOTPRINT[bi] = fp
        kas = bool(attrs.get("b3_kas_warenhuis"))
        height = (attrs.get("b3_h_dak_max") or ground) - ground
        kind = "house"
        if kas or geom is None or height < 1.0:
            # 3D BAG cannot reconstruct glass roofs (lidar passes through), so build those ourselves.
            kind = "greenhouse" if kas else "shed"
            height = greenhouse_height(attrs.get("oorspronkelijkbouwjaar")) if kas else 3.0
            rings = [V[r][:, :2] for r in parent["geometry"][0]["boundaries"][0]]
            roof_tris = emit(verts, extrude(rings, ground, ground + height), bi, ground)
        else:
            surfaces = geom["semantics"]["surfaces"]
            values = geom["semantics"]["values"]
            items = []
            for shell, shell_vals in zip(geom["boundaries"], values):
                for surf, sv in zip(shell, shell_vals):
                    k = KIND.get(surfaces[sv]["type"], 1) if sv is not None else 1
                    items.append((k, [V[r] for r in surf]))
            if "stilts" in named.get(parent_id, {}):
                items = on_stilts(items, ground, named[parent_id]["stilts"])
            roof_tris = emit(verts, items, bi, ground)
            area = attrs.get("b3_opp_grond") or 0
            if area > 250 and height < 10:
                kind = "barn"
        lm = named.get(parent_id)
        if lm:
            kind = lm["style"]
        roof_color = sample_roof(aerial, roof_tris)
        meta.append(
            {
                "id": parent_id,
                "kind": kind,
                "axis": round(long_axis(fp), 4),
                "glass": bool(attrs.get("b3_is_glas_dak")),
                "year": attrs.get("oorspronkelijkbouwjaar"),
                "floors": attrs.get("b3_bouwlagen"),
                "roof": attrs.get("b3_dak_type"),
                "area": attrs.get("b3_opp_grond"),
                "height": round(height, 2),
                "roof_rgb": roof_color,
                "rand": round(stable_rand(parent_id), 4),
            }
        )
        if lm:
            meta[-1].update({"name": lm["name"], "wall_fixed": lm.get("wall"), "roof_fixed": lm.get("roof"), "frame_fixed": lm.get("frame")})
            add_landmark_parts(verts, meta, parent_id, lm, ground, long_axis(fp), roof_tris)
    if not ERA_CONF:  # the BGT's sheds and canopies are today's
        add_extras(verts, meta, footprints, aerial, landmarks.get("stations", []))
    if ERA_CONF.get("ahn2"):
        add_demolished(verts, meta, footprints)
    add_masts(verts, meta, landmarks.get("masts", []))
    add_chimneys(verts, meta, footprints, landmarks.get("chimneys", []))
    add_bridges(verts, meta, landmarks.get("bridges", []))
    add_railings(verts, meta, landmarks.get("play", []))
    add_props(verts, meta, landmarks.get("props", []))
    add_walls(verts, meta, landmarks.get("walls", []))
    add_new_buildings(verts, meta, footprints, landmarks.get("new_buildings", []))
    add_monuments(verts, meta, footprints, landmarks.get("monuments", []))
    texture_roofs(aerial, meta)
    arr = np.concatenate(verts).astype(np.float32)
    arr.tofile(BUILD / "buildings.bin")
    (BUILD / "buildings.json").write_text(json.dumps(meta))
    (BUILD / "footprints.json").write_text(json.dumps(footprints))
    from collections import Counter

    print(f"{len(meta)} buildings, {len(arr) // 3} triangles, {Counter(m['kind'] for m in meta)}")


def texture_roofs(aerial, meta):
    roofs = [
        (bi, FOOTPRINT[bi], [t for _, t in ROOF_BLOCKS[bi]])
        for bi in ROOF_BLOCKS
        if bi in FOOTPRINT and meta[bi]["kind"] in ATLAS_KINDS and meta[bi].get("photo", True)
    ]
    fixed = {bi: meta[bi]["roof_fixed"] for bi, _, _ in roofs if meta[bi].get("roof_fixed")}
    rects, size = roof_atlas.build(aerial, roofs, fixed)
    for bi, rect in rects.items():
        for block, _ in ROOF_BLOCKS[bi]:
            u, v = roof_atlas.atlas_uv(np.column_stack([block[:, 0] + CX, CY - block[:, 2]]), rect, size)
            block[:, 10], block[:, 11] = u, v
        meta[bi]["atlas"] = True
    shifts = [np.hypot(*r[0]) for r in rects.values()]
    print(f"roof atlas {size[0]}x{size[1]} for {len(rects)} buildings, mean photo shift {np.mean(shifts):.2f} m")


OLD_STATION_WAY = 112289617  # the old station Ede-Wageningen in OpenStreetMap of 2012: a long block along the platform


def old_station():
    """The old station building (RD (174565, 448867)) as a ring, 5 m high: in both earlier eras, whatever else was found."""
    from geo import wgs_to_rd

    for w in json.loads((RAW / "osm_buildings-2012.json").read_text())["elements"]:
        if w["id"] == OLD_STATION_WAY:
            return np.array([wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]][:-1])
    return None


def osm_2012(taken_rings):
    """Buildings in OpenStreetMap as of September 2012 (data/raw/osm_buildings-2012.json) in the changed area that
    neither a building of today nor one found so far covers for 30% or more: the old station hall and its canopies.
    Their height from AHN2; where AHN2 saw only ground they were gone already, where it saw nothing 6 m, or 5 m for a
    canopy, long and narrow."""
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    import topo_buildings
    from era import changed, rail_zone
    from geo import wgs_to_rd

    from shapely.geometry import box

    # Plus the station: rebuilt since, but its new roof is no BAG building, so not in the changed area.
    near = changed().buffer(30).union(box(*ERA_CONF["osm_area"]) if ERA_CONF.get("osm_area") else Polygon())
    taken = [Polygon(r).buffer(0) for r in taken_rings if len(r) >= 3]
    tree = STRtree(taken)
    ahn = topo_buildings.heights()
    out = []
    for w in json.loads((RAW / "osm_buildings-2012.json").read_text())["elements"]:
        if w["type"] != "way" or len(w.get("geometry", [])) < 4 or w["id"] == OLD_STATION_WAY:
            continue
        p = Polygon([wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]]).buffer(0)
        if p.geom_type != "Polygon" or p.area < 12 or not near.intersects(p):
            continue
        if sum(p.intersection(taken[i]).area for i in tree.query(p)) >= 0.3 * p.area:
            continue
        if p.intersection(rail_zone()).area >= 0.3 * p.area:
            continue  # trains, platforms and sheds along the track: no building
        xy = np.asarray(p.minimum_rotated_rectangle.exterior.coords)
        short, long = sorted(np.hypot(*np.diff(xy, axis=0).T)[:2])
        h = ahn(p)
        if h < 2.5:
            continue  # AHN2 saw the bare ground there: gone by then, OpenStreetMap was behind
        if np.isnan(h):
            h = 5.0 if short <= 10 and long >= 60 else 6.0
        out.append((np.asarray(p.exterior.coords)[:-1], round(float(h), 1)))
    return out


def add_demolished(verts, meta, footprints):
    """The buildings of an earlier era that have been demolished since: from the topographic map of that year
    (scripts/topo_buildings.py) or else from its AHN (scripts/ahn2_buildings.py). Where AHN2 measured nothing (the
    military terrain round the barracks), from the topographic map of ERA_CONF["topo_blind"] instead; that area goes
    into blind.json for the trees and the ground. The aerial photo is today's, so their roofs get no photo."""
    kept = [[(x + CX, CY - z) for x, z in ring] for ring in footprints]
    import topo_buildings

    if ERA_CONF.get("topo"):
        found = [(ring, h, "topo") for ring, h in topo_buildings.find(kept, ERA_CONF["topo"])]
    else:
        import ahn2_buildings

        found = [(ring, h, "ahn2") for ring, h in ahn2_buildings.find(kept)]
        if ERA_CONF.get("topo_blind"):
            blind = ahn2_buildings.blind()
            parts = [p for p in getattr(blind, "geoms", [blind]) if p.geom_type == "Polygon"]
            (BUILD / "blind.json").write_text(json.dumps([[[round(x, 1), round(y, 1)] for x, y in p.exterior.coords] for p in parts]))
            found += [(ring, h, "barracks") for ring, h in topo_buildings.find(kept, ERA_CONF["topo_blind"], within=blind.buffer(10))]
        if ERA_CONF.get("osm") == "2012":
            found += [(ring, h, "osm") for ring, h in osm_2012(kept + [ring for ring, _, _ in found])]
    station = old_station()
    if station is not None:
        from shapely.geometry import Polygon as Poly

        sp = Poly(station).buffer(0)
        found = [f for f in found if Poly(f[0]).buffer(0).intersection(sp).area < 0.2 * Poly(f[0]).buffer(0).area]
        found.append((station, 5.0, "station"))
    for ring, h, source in found:
        bi = len(meta)
        emit(verts, extrude([ring], 0.0, h), bi, 0.0)
        area = abs(ring_area(ring))
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in ring])
        rand = stable_rand(f"ahn2-{bi}")
        # Only the factory halls have shed roofs; the barracks blocks of the military terrain have storeys of windows.
        kind = "hall" if source == "topo" and area > 1500 else "house" if source == "barracks" else "barn" if area > 250 and h < 12 else "house"
        meta.append({"id": f"ahn2-{bi}", "kind": kind, "axis": round(long_axis(ring), 4),
                     "glass": False, "year": 1935, "floors": None, "roof": "flat", "area": round(area, 1), "height": h,
                     "roof_rgb": None, "rand": round(rand, 4), "photo": False})
        if source == "station":
            meta[-1]["kind"] = "house"
            meta[-1]["roof_fixed"] = (122, 126, 130)
        elif area > 250:
            # Factory halls: red-brown tiles or grey corrugated sheets, as on the aerial photos of the Enka.
            meta[-1]["roof_fixed"] = [(156, 86, 62), (126, 130, 134), (110, 114, 120)][int(rand * 3)]
    (BUILD / "demolished.json").write_text(json.dumps([[[round(x, 2), round(y, 2)] for x, y in ring] for ring, _, _ in found]))
    print(f"{len(found)} demolished buildings, {sum(src == 'topo' for _, _, src in found)} of them from the topographic map")


def add_extras(verts, meta, footprints, aerial, station_roofs=()):
    """Tanks, silos and sheds from the BGT that do not overlap a BAG building, and the station roofs from landmarks.json."""
    from shapely.geometry import Point, Polygon

    w, h = int(XMAX - XMIN), int(YMAX - YMIN)
    ox, oz = CX - XMIN, YMAX - CY  # local (x, z) to mask column and row, at 1 m per pixel
    taken = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(taken)
    for ring in footprints:
        d.polygon([(x + ox, z + oz) for x, z in ring], fill=1)
    taken = np.array(taken)
    feats = json.load(open(RAW / "bgt_overigbouwwerk.geojson"))["features"]
    for f in feats:
        p = f["properties"]
        key = p.get("plus_type") if p.get("type") == "niet-bgt" else p.get("type")
        if key not in EXTRAS or f["geometry"] is None or f["geometry"]["type"] != "Polygon":
            continue
        rings = [np.array(r)[:, :2] for r in f["geometry"]["coordinates"]]
        c = rings[0].mean(0)
        cx, cz = int(c[0] - CX + ox), int(CY - c[1] + oz)
        if 0 <= cx < w and 0 <= cz < h and taken[cz, cx]:
            continue
        station = next((sc for sc in station_roofs if Polygon(rings[0]).contains(Point(*sc["at"]))), None)
        if station:
            add_station(verts, meta, station, rings[0])
            footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in rings[0]])
            continue
        height, kind = EXTRAS[key]
        bi = len(meta)
        FOOTPRINT[bi] = rings[0]
        roof_tris = emit(verts, extrude(rings, 0.0, height), bi, 0.0)
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in rings[0]])
        meta.append(
            {
                "id": p.get("lokaal_id"),
                "kind": kind,
                "axis": round(long_axis(rings[0]), 4),
                "glass": False,
                "year": None,
                "floors": None,
                "roof": "flat",
                "area": None,
                "height": height,
                "roof_rgb": sample_roof(aerial, roof_tris),
                "rand": round(stable_rand(str(p.get("lokaal_id"))), 4),
            }
        )


def sample_roof(aerial, roof_tris):
    """Median aerial color over the roof, shrunk a little to stay off the ground around it."""
    if not roof_tris:
        return None
    tris = np.concatenate(roof_tris)[..., :2]
    px = world_to_px(tris.reshape(-1, 2)).reshape(-1, 3, 2)
    x0, y0 = np.floor(px.min(axis=(0, 1))).astype(int) - 1
    x1, y1 = np.ceil(px.max(axis=(0, 1))).astype(int) + 2
    h, w = aerial.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(mask)
    for t in px:
        d.polygon([(float(x - x0), float(y - y0)) for x, y in t], fill=255)
    m = np.array(mask) > 0
    # Erode by ~0.75 m so lean and misalignment in the photo pull in less of the surroundings.
    for _ in range(3):
        m[1:, :] &= m[:-1, :]
        m[:-1, :] &= m[1:, :]
        m[:, 1:] &= m[:, :-1]
        m[:, :-1] &= m[:, 1:]
    crop = aerial[y0:y1, x0:x1][m]
    if len(crop) < 8:
        crop = aerial[y0:y1, x0:x1][np.array(mask) > 0]
    if len(crop) == 0:
        return None
    return [int(v) for v in np.median(crop, axis=0)]


if __name__ == "__main__":
    main()
