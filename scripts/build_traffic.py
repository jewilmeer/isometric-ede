# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "shapely"]
# ///
"""Traffic for the viewer: lanes along the OpenStreetMap roads and paths that little cars, cyclists and walkers use.

A lane runs one way along a stretch of road between two junctions, on the right-hand side: the BGT carriageways and
cycle paths show how wide the road really is and where it lies, since the OpenStreetMap line is seldom exactly in
the middle. Cars keep to the middle of their half, cyclists to its edge, walkers to the pavement when the BGT has one
next to the road, and to the edge of the road when it doesn't. Where a building or a tree
crown stands in front of the road on screen, the lane records a hidden stretch, so a car disappears behind it instead
of being drawn on the roof; under a tree crown it shows faintly, as if seen through the leaves. Roads that leave the map end in an exit: a car that drives out comes back in at another
exit of the same road class.

Output: data/build/traffic.json
  lanes  [{"e": road stretch, "a": from node, "b": to node, "p": [[x, z], ...] in local meters (x east, z south, as in
          render/scene.js), "c": road class, "v": speed for cars (m/s), "w", "wb" and "ww": weight when a car, a
          cyclist or a walker picks its next lane (0: not for them), "d", "db" and "dw": cars, cyclists and walkers
          per km, "bo" and "wo": how much further right a cyclist and a walker go than the lane,
          "h": per view in config.VIEWS, [[s0, s1], ...] stretches in meters along the lane behind a building,
          "t": the same under tree crowns, "y": on a bridge, the height of each point of p}]
  exits  the nodes where a road leaves the map
  views  the views the hidden stretches are for, as in config.VIEWS
  play   [{"name", "h": height of the floor, "kids", "pts": [[x, z], ...] every meter inside the area, "hid": per view a
          string with a 1 for each point hidden behind a building}]: where children play (scripts/landmarks.py PLAY)
"""

import json
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity, contains_xy, prepare
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import unary_union

from era import bare, changed, underpass_ids
from era import local as to_local
from config import ERA, BUILD, CX, CY, RAW, VIEWS, XMAX, XMIN, YMAX, YMIN
from geo import screen, wgs_to_rd

RES = 0.5  # occlusion raster, screen meters per pixel
STEP = 0.5  # lane sampling for hidden stretches, meters

# Road class: default speed (km/h); for cars the weight at junctions (they prefer the bigger road) and cars per km of
# lane; the same two for cyclists, who prefer cycle paths and quiet roads, and for walkers, who prefer footpaths,
# village streets and farm tracks.
CLASSES = {
    "motorway": (120, 10, 8.0, 0, 0, 0, 0),
    "trunk": (100, 8, 4.0, 0, 0, 0, 0),
    "primary": (80, 6, 2.5, 0.5, 0.1, 0.3, 0.05),
    "secondary": (80, 5, 2.0, 1, 0.15, 0.5, 0.05),
    "tertiary": (60, 4, 1.6, 1.5, 0.3, 1, 0.25),
    "unclassified": (60, 3, 0.9, 3, 0.4, 1.5, 0.25),
    "residential": (30, 2, 1.6, 3, 0.6, 3, 1.5),
    "living_street": (15, 1, 1.0, 2, 0.5, 3, 1.5),
    "service": (15, 0.4, 0.4, 0.4, 0.05, 0.5, 0.1),
    "cycleway": (20, 0, 0, 5, 0.8, 1.5, 0.3),
    "track": (20, 0, 0, 0.5, 0.05, 2.5, 0.4),
    "footway": (5, 0, 0, 0, 0, 4, 1.5),
}
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
NO_CARS = {"no", "private", "agricultural", "forestry", "delivery"}
BIKES = {"yes", "designated", "permissive"}


def road_file():
    """Today's roads, or in an earlier era those of OpenStreetMap's oldest state, when fetched."""
    then = RAW / "osm_roads-2012.json"
    return then if ERA is not None and then.exists() else RAW / "osm_roads.json"


def local(lat, lon):
    x, y = wgs_to_rd(lat, lon)
    return x - CX, CY - y


def road_ways():
    # Ways tagged as a bridge that really run under the railway (era.underpass_ids) are on the ground.
    under = underpass_ids() if road_file() == RAW / "osm_roads.json" else set()
    for w in json.loads(road_file().read_text())["elements"]:
        t = w.get("tags", {})
        hw = t.get("highway", "")
        if w["type"] != "way" or t.get("area") == "yes" or t.get("service") in SKIP_SERVICE:
            continue
        if hw == "cycleway" or (hw in ("path", "track", "footway", "bridleway") and t.get("bicycle") in ("designated", "yes")):
            base = "cycleway"
        elif hw in ("footway", "path", "pedestrian", "steps", "bridleway"):
            base = "footway"
        else:
            base = hw.removesuffix("_link")
        if base not in CLASSES:
            continue
        kmh, weight, density, bike_weight, bike_density, walk_weight, walk_density = CLASSES[base]
        if NO_CARS & {t.get("access"), t.get("motor_vehicle"), t.get("motorcar"), t.get("vehicle")} - {None}:
            weight, density = 0, 0  # a farm road: no cars, but cyclists may
        if t.get("bicycle") in ("no", "use_sidepath") or (t.get("access") in ("no", "private") and t.get("bicycle") not in BIKES):
            bike_weight, bike_density = 0, 0
        if t.get("foot") in ("no", "use_sidepath") or (t.get("access") in ("no", "private") and t.get("foot") not in BIKES):
            walk_weight, walk_density = 0, 0
        if hw.endswith("_link"):
            walk_weight, walk_density = 0, 0
        if not weight and not bike_weight and not walk_weight:
            continue
        if hw.endswith("_link"):
            kmh, weight = min(kmh, 70), weight / 2
        digits = "".join(ch for ch in t.get("maxspeed", "") if ch.isdigit())
        if digits:
            kmh = int(digits)
        oneway = t.get("oneway") in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular")
        oneway = oneway or (base == "motorway" and t.get("oneway") != "no")
        if t.get("oneway") == "-1":
            oneway = -1
        yield {
            "nodes": w["nodes"],
            "pts": [local(g["lat"], g["lon"]) for g in w["geometry"]],
            "c": base,
            "v": round(kmh / 3.6 * 0.9, 1),
            "w": weight,
            "d": density,
            "wb": bike_weight,
            "db": bike_density,
            "ww": walk_weight,
            "dw": walk_density,
            "oneway": oneway,
            # On a bridge or viaduct: above whatever passes under it.
            "layer": max(1, int(t.get("layer", "1") or 1)) if t.get("bridge") not in (None, "no") and w["id"] not in under else 0,
        }


def stretches(ways):
    """Split the ways at junctions: nodes shared by more than one way, and the ends of every way."""
    use = Counter(n for w in ways for n in w["nodes"])
    junction = {n for n, k in use.items() if k > 1} | {w["nodes"][0] for w in ways} | {w["nodes"][-1] for w in ways}
    for w in ways:
        start, pts = 0, [w["pts"][0]]
        for i in range(1, len(w["nodes"])):
            pts.append(w["pts"][i])
            if w["nodes"][i] in junction:
                yield {**w, "a": str(w["nodes"][start]), "b": str(w["nodes"][i]), "pts": pts}
                start, pts = i, [w["pts"][i]]


def clip(edges):
    """Keep what lies on the map; a road cut off by the map's edge ends in an exit there."""
    area = box(XMIN - CX + 1, CY - YMAX + 1, XMAX - CX - 1, CY - YMIN - 1)
    exits, n = set(), 0
    for e in edges:
        line = LineString(e["pts"])
        if area.contains(line):
            yield e
            continue
        part = line.intersection(area)
        pieces = getattr(part, "geoms", [part])
        for piece in pieces:
            if piece.geom_type != "LineString" or piece.length < 1:
                continue
            c = list(piece.coords)
            ends = []
            for pt, node in ((c[0], e["a"]), (c[-1], e["b"])):
                if Point(pt).distance(Point(e["pts"][0] if node == e["a"] else e["pts"][-1])) < 0.01:
                    ends.append(node)
                else:
                    n += 1
                    ends.append(f"x{n}")
                    exits.add(f"x{n}")
            yield {**e, "a": ends[0], "b": ends[1], "pts": c}
    clip.exits = exits


def surfaces():
    """The BGT carriageways and cycle paths in local meters, to measure where each road lies and how wide it is."""
    groups = {"road": [], "cycle": [], "foot": []}
    for f in json.loads((RAW / "bgt_wegdeel.geojson").read_text())["features"]:
        fn = f["properties"].get("functie") or ""
        key = "road" if fn.startswith("rijbaan") else "cycle" if fn == "fietspad" else "foot" if fn == "voetpad" else None
        if key and f["geometry"]:
            groups[key].append(affinity.affine_transform(shape(f["geometry"]), [1, 0, 0, -1, -CX, CY]).buffer(0))
    out = {k: unary_union(v) for k, v in groups.items()}
    for g in out.values():
        prepare(g)
    return out


def measure(pts, surface, foot):
    """Where the road surface lies around a polyline: (center, half width, pavement right, pavement left), all in meters
    to the right of the line, a pavement None where there is none. Measured across every 4 m, away from the junctions
    at the ends, and the median taken."""
    p = np.asarray(pts, dtype=np.float64)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 0.05]]
    seg = np.diff(p, axis=0)
    if len(seg) == 0:
        return None
    length = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(length)]
    s = np.arange(4, cum[-1] - 4, 4.0)
    if len(s) < 2:
        return None
    xs, zs = np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])
    i = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
    d = seg[i] / length[i][:, None]
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    keep = contains_xy(surface, xs, zs)
    if keep.sum() < 2:
        return None
    xs, zs, n = xs[keep], zs[keep], n[keep]
    right, left = reach(surface, xs, zs, n, 0, 10), reach(surface, xs, zs, -n, 0, 10)
    center, half = float(np.median((right - left) / 2)), float(np.median((right + left) / 2))
    if not 1 <= half <= 6:
        return None
    return center, half, pavement(foot, xs, zs, n, right), pavement(foot, xs, zs, -n, left)


def reach(area, xs, zs, n, start, far, step=0.2):
    """Per sample, how far along n from start (a number or one per sample) the area goes on; start must be inside."""
    ds = np.arange(0.1, far, step)
    at = np.broadcast_to(np.asarray(start, dtype=np.float64), xs.shape)[:, None] + ds
    inside = contains_xy(area, xs[:, None] + n[:, :1] * at, zs[:, None] + n[:, 1:] * at)
    first = np.where(inside.all(axis=1), len(ds) - 1, np.argmax(~inside, axis=1))
    return np.broadcast_to(np.asarray(start, dtype=np.float64), xs.shape) + ds[first] - 0.1


def pavement(foot, xs, zs, n, edge):
    """The middle of a pavement beside the road, in meters from the line, when most of the way has one within 4 m of
    the road's edge."""
    ds = np.arange(0.1, 4, 0.2)
    on = contains_xy(foot, xs[:, None] + n[:, :1] * (edge[:, None] + ds), zs[:, None] + n[:, 1:] * (edge[:, None] + ds))
    found = on.any(axis=1)
    if found.mean() < 0.6:
        return None
    near = edge[found] + ds[np.argmax(on[found], axis=1)]
    far = reach(foot, xs[found], zs[found], n[found], near, 6)
    return float(np.median((near + far) / 2))


def offset(pts, d):
    """The polyline moved d meters to the right of its direction (x east, z south: right of east is south)."""
    p = np.asarray(pts, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 0.05]
    p = p[keep]
    if len(p) < 2 or d == 0:
        return p
    seg = np.diff(p, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    nrm = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    vn = np.vstack([nrm[:1], nrm[:-1] + nrm[1:], nrm[-1:]])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    # Miter: a corner moves out further than d, up to twice as far for sharp bends.
    cos = np.r_[1.0, np.einsum("ij,ij->i", vn[1:-1], nrm[1:]), 1.0]
    return p + vn * (d / np.clip(cos, 0.5, 1.0))[:, None]


def lanes(edges):
    surface = surfaces()
    out, measured = [], 0
    for i, e in enumerate(edges):
        dirs = [(e["pts"], e["a"], e["b"], 1)]
        if e["oneway"] == -1:
            dirs = [(e["pts"][::-1], e["b"], e["a"], -1)]
        elif not e["oneway"]:
            dirs.append((e["pts"][::-1], e["b"], e["a"], -1))
        kind = {"motorway": None, "cycleway": "cycle", "footway": None, "track": None}.get(e["c"], "road")
        fit = kind and measure(e["pts"], surface[kind], surface["foot"])
        measured += bool(fit)
        for pts, a, b, sign in dirs:
            if e["c"] == "motorway":
                off = bike = walk = 1.6
            elif fit:
                center, half = fit[0] * sign, fit[1]
                side = fit[2] if sign > 0 else fit[3]
                if side is not None:
                    side *= sign
                if e["c"] == "cycleway":
                    off = center if e["oneway"] else center + half / 2
                    bike = off
                    walk = side if side is not None else center + half - 0.3
                else:
                    off = center if e["oneway"] else center + min(half / 2, 1.9)
                    bike = center + half - 0.45
                    walk = side if side is not None else center + half - 0.2
            else:
                width = {"cycleway": 0.6, "footway": 0.35, "track": 0.8}.get(e["c"])
                width = width or (1.3 if e["c"] in ("residential", "living_street", "service", "unclassified") else 1.7)
                off = 0.0 if e["oneway"] else width
                bike = off if e["c"] in ("cycleway", "track") else off + 0.75
                walk = off if e["c"] in ("footway", "track") else bike + 0.25
            p = offset(pts, off)
            if len(p) >= 2:
                lane = {"e": i, "a": a, "b": b, "p": p, **{k: e[k] for k in ("c", "v", "w", "d", "wb", "db", "ww", "dw", "layer")}}
                lane["bo"], lane["wo"] = round(bike - off, 2), round(walk - off, 2)
                out.append(lane)
    lanes.measured = f"{measured} of {len(edges)} stretches of road measured on the BGT"
    return out


def occluders(bounds, look):
    """Rasters in screen meters of what stands up from the ground: one of the buildings, one of the tree crowns.

    A ground point whose screen position falls inside one is behind it, or under it."""
    u0, v0, u1, v1 = bounds
    w, h = int((u1 - u0) / RES) + 1, int((v1 - v0) / RES) + 1
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    trees = Image.new("1", (w, h), 0)
    px = lambda u, v: ((u - u0) / RES, (v1 - v) / RES)

    b = np.fromfile(BUILD / "buildings.bin", dtype=np.float32).reshape(-1, 12)
    u, v = screen(b[:, 0], b[:, 2], b[:, 1], look)
    tri = np.stack([(u - u0) / RES, (v1 - v) / RES], axis=1).reshape(-1, 3, 2)
    for t in tri:
        draw.polygon([tuple(q) for q in t], fill=1)

    buildings = np.array(img, dtype=bool)
    draw = ImageDraw.Draw(trees)
    t = np.fromfile(BUILD / "trees.bin", dtype=np.float32).reshape(-1, 10)
    for x, z, ht, r, base, kind in t[:, :6]:
        if kind > 0.5:  # conifer: a cone from half the crown base up to the top
            bottom, rr = base * 0.5, r * 0.9
            cu, cv = screen(x, z, bottom, look)
            _, tv = screen(x, z, ht, look)
            ring = [px(cu + rr * math.cos(a), cv + 0.5 * rr * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 7)]
            draw.polygon([*ring, px(cu, tv)], fill=1)
        else:  # round crown, as render/scene.js scales it: an ellipsoid r * 1.12 wide and `half` high
            half = max(1, (ht - base) / 2, min(r * 0.9, ht * 0.42))
            cu, cv = screen(x, z, ht - half, look)
            a = r * 1.12
            bv = math.hypot(0.5 * a, math.cos(math.radians(30)) * half)
            (l, top), (rgt, bot) = px(cu - a, cv + bv), px(cu + a, cv - bv)
            draw.ellipse([l, top, rgt, bot], fill=1)
    return buildings, np.array(trees, dtype=bool)


def stretches_where(hit, s):
    """Runs of samples where hit is set, as [from, to] in meters; small gaps closed so nothing flickers."""
    runs, start = [], None
    for i, h in enumerate(np.r_[hit, False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            runs.append([s[start] - STEP / 2, s[i - 1] + STEP / 2])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 1.5:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [[round(a, 1), round(b, 1)] for a, b in merged if b - a >= 1]


def on_bridges(lanes_):
    """Heights for the lanes that run over a bridge from landmarks.json: the deck height under each point."""
    marks = json.loads((BUILD / "landmarks.json").read_text()) if (BUILD / "landmarks.json").exists() else {}
    for b in marks.get("bridges", []):
        deck = LineString([(x - CX, CY - y) for x, y in b["path"]])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(deck.coords), axis=0), axis=1))]
        for ln in lanes_:
            # On the deck: near its line, and running along it (a road that passes under it crosses it).
            p = np.asarray(ln["p"])
            heading = (p[-1] - p[0]) / (np.linalg.norm(p[-1] - p[0]) or 1)
            near = []
            for q in p:
                at = deck.project(Point(*q))
                p0, p1 = deck.interpolate(max(0, at - 2)), deck.interpolate(min(deck.length, at + 2))
                along = np.array([p1.x - p0.x, p1.y - p0.y]) / (np.hypot(p1.x - p0.x, p1.y - p0.y) or 1)
                near.append(deck.distance(Point(*q)) < max(3.0, b["width"] / 2) and abs(heading @ along) > 0.8)
            if sum(near) < 2:
                continue
            ys = [float(np.interp(deck.project(Point(*q)), cum, b["heights"])) if n else 0.0 for q, n in zip(ln["p"], near)]
            ln["y"] = [round(y, 2) for y in np.maximum(ys, ln.get("y", 0))]


def on_sunken(lanes_):
    """Negative heights for the lanes in the sunken underpasses and on their ramps (scripts/build_sunken.py): the depth of
    the floor under each point of p. A lane that touches one gets a point every 2 m first, as the floor slopes."""
    f = BUILD / "sunken.json"
    for s in json.loads(f.read_text()) if f.exists() else []:
        g = s.get("grid")
        if not g:
            continue
        z = np.asarray(g["z"], dtype=float).reshape(g["h"], g["w"])
        area = Polygon([(x - CX, CY - y) for x, y in s["area"]]).buffer(1.5)
        prepare(area)

        def depth(p):  # bilinear, p in local meters
            fi = np.clip((p[:, 0] + CX - g["x0"]) / g["res"] - 0.5, 0, g["w"] - 1)
            fj = np.clip((CY - p[:, 1] - g["y0"]) / g["res"] - 0.5, 0, g["h"] - 1)
            i, j = np.minimum(fi.astype(int), g["w"] - 2), np.minimum(fj.astype(int), g["h"] - 2)
            a, b = fi - i, fj - j
            return (z[j, i] * (1 - a) + z[j, i + 1] * a) * (1 - b) + (z[j + 1, i] * (1 - a) + z[j + 1, i + 1] * a) * b

        for ln in lanes_:
            p = np.asarray(ln["p"], dtype=float)
            if not area.intersects(LineString(p)):
                continue
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            keep = np.unique(np.r_[np.arange(0, cum[-1], 2.0), cum])
            q = np.c_[np.interp(keep, cum, p[:, 0]), np.interp(keep, cum, p[:, 1])]
            h = np.where(contains_xy(area, q[:, 0], q[:, 1]), -depth(q), 0.0)
            if (h > -0.3).all():
                continue
            if "y" in ln:
                h = np.minimum(h, np.interp(keep, cum, ln["y"]))
            ln["p"] = q
            ln["y"] = [round(float(v), 2) for v in h]


def decks():
    """Where a bridge deck or a roof lies over the ground, in local meters: the roads and paths that OpenStreetMap marks
    as a bridge, as wide as a road or a path, the station roofs, and the bridges in landmarks.json where they are more
    than 2 m up."""
    from shapely.ops import unary_union

    parts = []
    for ln in DECK_LANES:
        parts.append(LineString(ln["p"]).buffer(2.0 if ln["c"] in ("cycleway", "footway", "track") else 3.5, cap_style="flat"))
    marks = json.loads((BUILD / "landmarks.json").read_text()) if (BUILD / "landmarks.json").exists() else {}
    # The station roofs: the BGT canopy each one stands on.
    from shapely.geometry import Point as P

    for st in marks.get("stations", []):
        for f in json.loads((RAW / "bgt_overigbouwwerk.geojson").read_text())["features"]:
            g = f["geometry"]
            if g and g["type"] == "Polygon" and shape(g).contains(P(*st["at"])):
                parts.append(affinity.affine_transform(shape(g), [1, 0, 0, -1, -CX, CY]))
    for b in marks.get("bridges", []):
        pts = [(x - CX, CY - y) for (x, y), h in zip(b["path"], b["heights"]) if h > 2.0]
        if len(pts) >= 2:
            parts.append(LineString(pts).buffer(b["width"] / 2, cap_style="flat"))
    # Roads in a short tunnel under the railway: the BGT roads at relative height -1 (trains run over them, not in them).
    from era import local as to_local_geom, under_polygons

    parts.append(to_local_geom(under_polygons(False)))
    return unary_union(parts) if parts else None


DECK_LANES = []


def hidden(lanes_):
    """Per view, the stretches of each lane behind a building ("h") and under tree crowns ("t"), and once the stretches
    under a bridge deck ("u"): there a mover is out of sight whatever the view, until it comes out the other side."""
    samples = []
    for ln in lanes_:
        p = ln["p"]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        s = np.arange(0, cum[-1], STEP)
        samples.append((s, np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])))
    # On a bridge deck a mover is in plain view: the deck itself is one of the buildings it would hide behind.
    raised = [np.interp(s, np.r_[0, np.cumsum(np.linalg.norm(np.diff(ln["p"], axis=0), axis=1))], ln["y"]) > 0.3 if "y" in ln
              else np.zeros(len(s), bool) for ln, (s, _, _) in zip(lanes_, samples)]
    deck = decks()
    if deck is not None:
        prepare(deck)
    for ln, (s, xs, zs), up in zip(lanes_, samples, raised):
        on_top = ln.get("layer", 0) > 0 or up.any()
        ln["u"] = [] if deck is None or on_top else stretches_where(contains_xy(deck, xs, zs), s)
    for ln in lanes_:
        ln["h"], ln["t"] = [], []
    for look in VIEWS:
        uv = [screen(xs, zs, 0, look) for _, xs, zs in samples]
        us, vs = np.concatenate([u for u, _ in uv]), np.concatenate([v for _, v in uv])
        bounds = (us.min() - 2, vs.min() - 2, us.max() + 2, vs.max() + 2)
        buildings, trees = occluders(bounds, look)
        u0, _, _, v1 = bounds
        for ln, (s, _, _), (u, v), up in zip(lanes_, samples, uv, raised):
            at = ((v1 - v) / RES).astype(int), ((u - u0) / RES).astype(int)
            ln["h"].append(stretches_where(buildings[at] & ~up, s))
            ln["t"].append(stretches_where(trees[at] & ~buildings[at] & ~up, s))


def mills():
    marks = json.loads((BUILD / "landmarks.json").read_text()) if (BUILD / "landmarks.json").exists() else {}
    return [{"name": m["name"], "x": m["at"][0] - CX, "z": CY - m["at"][1], "hub": m["hub"], "span": m["span"], "facing": m["facing"]}
            for m in marks.get("mills", [])]


def play_areas():
    """Points every meter inside each play area, and per view which of them a building hides: a depth test against the
    building triangles nearby, so children on a roof terrace disappear behind the higher part of the building too."""
    marks = json.loads((BUILD / "landmarks.json").read_text()) if (BUILD / "landmarks.json").exists() else {}
    tris = np.fromfile(BUILD / "buildings.bin", dtype=np.float32).reshape(-1, 3, 12)[..., :3].astype(np.float64)  # x, up, z
    out = []
    for a in marks.get("play", []):
        poly = Polygon([(x - CX, CY - y) for x, y in a["area"]])
        x0, z0, x1, z1 = poly.bounds
        gx, gz = np.meshgrid(np.arange(x0 + 0.5, x1, 1.0), np.arange(z0 + 0.5, z1, 1.0))
        inside = contains_xy(poly, gx, gz)
        px, pz = gx[inside], gz[inside]
        c = np.array(poly.centroid.coords[0])
        near = tris[np.hypot(tris[..., 0] - c[0], tris[..., 2] - c[1]).min(1) < 80]
        hid = []
        for look in VIEWS:
            depth = lambda x, z, y: (x * math.sin(math.radians(look)) - z * math.cos(math.radians(look))) * math.cos(math.radians(30)) - y * 0.5
            pu, pv = screen(px, pz, a["h"] + 0.8, look)
            pd = depth(px, pz, a["h"] + 0.8)
            tu, tv = screen(near[..., 0], near[..., 2], near[..., 1], look)
            td = depth(near[..., 0], near[..., 2], near[..., 1])
            # Barycentric coordinates of every point in every triangle on screen.
            (ua, ub, uc), (va, vb, vc) = tu.T, tv.T
            det = (vb - vc) * (ua - uc) + (uc - ub) * (va - vc)
            det = np.where(np.abs(det) < 1e-9, np.nan, det)
            l1 = ((vb - vc)[None] * (pu[:, None] - uc[None]) + (uc - ub)[None] * (pv[:, None] - vc[None])) / det[None]
            l2 = ((vc - va)[None] * (pu[:, None] - uc[None]) + (ua - uc)[None] * (pv[:, None] - vc[None])) / det[None]
            l3 = 1 - l1 - l2
            cover = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            tdepth = l1 * td[:, 0][None] + l2 * td[:, 1][None] + l3 * td[:, 2][None]
            hid.append("".join("1" if h else "0" for h in (cover & (tdepth < pd[:, None] - 0.4)).any(1)))
        swing = a.get("swing") and {"x": round(a["swing"]["at"][0] - CX, 2), "z": round(CY - a["swing"]["at"][1], 2), "angle": a["swing"]["angle"]}
        out.append({"name": a["name"], "h": a["h"], "kids": a["kids"], **({"swing": swing} if swing else {}), "pts": [[round(float(x), 1), round(float(z), 1)] for x, z in zip(px, pz)], "hid": hid})
        print(f"{a['name']}: {len(px)} m², hidden per view {[h.count('1') for h in hid]}")
    return out


def main():
    ways = list(road_ways())
    edges = list(clip(stretches(ways)))
    out = lanes(edges)
    if ERA is not None:
        # The roads are those of 2012 (scripts/fetch.py fetch_attic), the oldest OpenStreetMap keeps; on the building
        # site (scripts/era.py) none were there yet. Without them, no streets in the whole changed area.
        gone = to_local(bare() if road_file() != RAW / "osm_roads.json" else changed())
        prepare(gone)
        out = [ln for ln in out if not contains_xy(gone, *np.asarray(ln["p"]).mean(0))]
    DECK_LANES[:] = [ln for ln in out if ln.get("layer", 0) > 0]
    on_sunken(out)
    on_bridges(out)
    hidden(out)
    for ln in out:
        ln["p"] = [[round(float(x), 1), round(float(z), 1)] for x, z in ln["p"]]
    (BUILD / "traffic.json").write_text(json.dumps({"views": list(VIEWS), "lanes": out, "exits": sorted(clip.exits), "play": play_areas(), "mills": mills()}, separators=(",", ":")))
    km = sum(np.linalg.norm(np.diff(np.array(ln["p"]), axis=0), axis=1).sum() for ln in out) / 1000
    share = sum(sum(b - a for a, b in ln["h"][0]) for ln in out) / 1000 / km
    leafy = sum(sum(b - a for a, b in ln["t"][0]) for ln in out) / 1000 / km
    print(f"{len(out)} lanes, {km:.0f} km, {len(clip.exits)} exits, {share:.0%} behind buildings, {leafy:.0%} under trees; {lanes.measured}")


if __name__ == "__main__":
    main()
