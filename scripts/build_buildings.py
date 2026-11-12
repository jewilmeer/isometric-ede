# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mapbox-earcut", "pillow", "scipy"]
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
from config import BUILD, CX, CY, RAW, XMAX, XMIN, YMAX, YMIN, write_area

KIND = {"GroundSurface": 0, "WallSurface": 1, "RoofSurface": 2}
COLS = 12

# Roof vertex blocks and footprints per building index, for texturing roofs from the aerial photo afterwards.
ROOF_BLOCKS = defaultdict(list)
FOOTPRINT = {}
ATLAS_KINDS = {"house", "barn", "shed", "church", "school", "tank", "silo", "canopy"}

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
        if np.allclose(r[0], r[-1]):
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
        if np.allclose(r[0], r[-1]):
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
        surf, rails = [], []
        for i in range(len(path) - 1):
            h0, h1 = hs[i], hs[i + 1]
            l0, l1, r0, r1 = (*L[i], h0), (*L[i + 1], h1), (*R[i], h0), (*R[i + 1], h1)
            surf += [(2, tri(l0, r0, r1)), (2, tri(l0, r1, l1))]
            # The edge of the deck, 0.8 m thick, down to the ground where the ramp is low. Each wall runs with its
            # outside on the right, as in extrude().
            for a, c, ha, hc in ((L[i + 1], L[i], h1, h0), (R[i], R[i + 1], h0, h1)):
                lo_a, lo_c = (0.0 if ha < 2.0 else ha - 0.8), (0.0 if hc < 2.0 else hc - 0.8)
                surf.append((1, [np.array([(*a, lo_a), (*c, lo_c), (*c, hc), (*a, ha)])]))
                rails.append((1, [np.array([(*a, ha), (*c, hc), (*c, hc + 1.1), (*a, ha + 1.1)])]))
        # Columns every 12 m where the deck is high, but none on the tracks.
        for s in np.arange(6, cum[-1], 12.0):
            h = float(np.interp(s, cum, hs))
            if h < 2.0 or s0 - 4 < s < s1 + 4:
                continue
            x, y = np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])
            surf += extrude([rect_ring(x, y, 0.8, 0.8, 0.0)], 0.0, h - 0.8)
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
        if attrs.get("status") not in ("Pand in gebruik", "Pand in gebruik (niet ingemeten)", "Verbouwing pand"):
            continue
        ground = attrs.get("b3_h_maaiveld") or 0.0
        lod0 = parent["geometry"][0]["boundaries"][0][0]
        fp = V[lod0][:, :2]
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in fp])

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
    add_extras(verts, meta, footprints, aerial)
    add_masts(verts, meta, landmarks.get("masts", []))
    add_chimneys(verts, meta, footprints, landmarks.get("chimneys", []))
    add_bridges(verts, meta, landmarks.get("bridges", []))
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
        if bi in FOOTPRINT and meta[bi]["kind"] in ATLAS_KINDS
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


def add_extras(verts, meta, footprints, aerial):
    """Tanks, silos and sheds from the BGT that do not overlap a BAG building."""
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
