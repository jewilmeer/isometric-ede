# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pillow", "shapely"]
# ///
"""Paint the ground: BGT polygons in a pixel-art palette, with baked sun shadows.

Output, in 2000x2000 px chunks of 500 m (0.25 m/px), row 0 north:
  data/build/ground/{ix}_{iy}.png   the ground, with the roads filled in by the surfaces around them
  data/build/roads/{ix}_{iy}.png    the roads' own colours (shadows included), running on past their edges
  data/build/roads.json             the road polygons, which render/scene.js draws with hard edges
"""

import hashlib
import json
import math
from functools import cache

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from aerial import load_aerial
from era import bare, built_after, changed
from config import ERA, ERA_CONF, BUILD, CHUNK, CX, CY, GROUND_RES, NX as CHUNKS_X, NY as CHUNKS_Y, RAW, XMAX, XMIN, YMAX, YMIN
from sun import SUN_AZIMUTH, SUN_ELEVATION

NX, NY = int((XMAX - XMIN) / GROUND_RES), int((YMAX - YMIN) / GROUND_RES)

# Style categories. Order matters only for readability.
STYLES = {
    "gap": (150, 150, 120),
    "grass": (116, 160, 70),
    "meadow": (124, 166, 74),
    "berm": (120, 158, 74),
    "shrub": (84, 122, 56),
    "forest": (68, 98, 50),
    "field": (150, 118, 82),
    "orchard": (118, 152, 72),
    "heath": (146, 118, 96),
    "marsh": (104, 132, 92),
    "water": (70, 128, 172),
    "ditch": (98, 128, 70),
    "bank": (108, 144, 68),
    "yard": (160, 155, 140),
    "sand": (222, 204, 160),
    "asphalt": (98, 100, 106),
    "motorway": (84, 86, 92),
    "concrete_pavers": (160, 156, 148),
    "tiles": (180, 177, 170),
    "clinkers": (164, 104, 84),
    "concrete": (178, 175, 164),
    "gravel": (176, 164, 138),
    "grass_pavers": (138, 160, 98),
    "dirt": (178, 156, 114),
    "building": (168, 162, 150),
    "bridge": (150, 140, 126),
    "cyclepath": (212, 118, 60),
    "bare": (206, 184, 140),
    "tartan": (192, 88, 58),  # the athletics track  # a building site in an earlier era (scripts/era.py)
}
CAT = {k: i for i, k in enumerate(STYLES)}
WATERY = {CAT["water"]}
SOFT = {CAT[k] for k in ("bare", "grass", "meadow", "berm", "shrub", "forest", "field", "orchard", "heath", "marsh", "ditch", "bank", "yard", "dirt", "sand", "gap")}


def style_for(coll, p):
    fv = p.get("fysiek_voorkomen") or ""
    plus = p.get("plus_fysiek_voorkomen") or ""
    func = p.get("functie") or ""
    if coll == "waterdeel":
        return "ditch" if fv.startswith("greppel") else "water"
    if coll == "ondersteunendwaterdeel":
        return "bank"
    if coll in ("pand", "overigbouwwerk"):
        return "building" if p.get("type") != "bassin" else "water"
    if coll == "overbruggingsdeel":
        return "bridge"
    if coll == "park":
        return "grass"
    if coll == "begroeidterreindeel" or fv == "groenvoorziening":
        return {
            "grasland agrarisch": "meadow",
            "grasland overig": "meadow",
            "bouwland": "field",
            "loofbos": "forest",
            "gemengd bos": "forest",
            "naaldbos": "forest",
            "houtwal": "forest",
            "struiken": "shrub",
            "fruitteelt": "orchard",
            "boomteelt": "orchard",
            "heide": "heath",
            "moeras": "marsh",
        }.get(fv, "shrub" if plus in ("heesters", "bosplantsoen", "planten") else ("berm" if func == "berm" else "grass"))
    if func == "rijbaan autosnelweg":
        return "motorway"
    if plus == "asfalt" or (fv == "gesloten verharding" and plus != "cementbeton"):
        return "asphalt"
    if plus in ("cementbeton", "beton element"):
        return "concrete"
    if plus == "betonstraatstenen":
        return "concrete_pavers"
    if plus in ("tegels", "sierbestrating"):
        return "tiles"
    if plus == "gebakken klinkers":
        return "clinkers"
    if plus == "grasklinkers":
        return "grass_pavers"
    if plus in ("grind", "puin") or fv == "half verhard":
        return "gravel"
    if fv == "open verharding":
        return "concrete_pavers"
    if fv == "erf":
        return "yard"
    if fv == "zand" or plus == "zand":
        return "sand"
    if fv == "onverhard":
        return "dirt"
    return "yard"


# Surfaces that are drawn as crisp polygons over the ground (see render/scene.js, makeRoads): the whole road
# network, bridges, and paved squares and yards. Grass verges and plain yards stay part of the soft ground.
ROAD_STYLES = {"asphalt", "motorway", "concrete_pavers", "tiles", "clinkers", "concrete", "gravel", "grass_pavers", "dirt", "sand", "bridge",
               "cyclepath"}

# What changed after the BGT inside an area of landmarks.json "ground": bare ground and old asphalt become park grass.
TO_PARK = {"asphalt", "sand", "dirt", "yard", "gravel", "gap"}
PAVED_STYLES = {"asphalt", "concrete_pavers", "tiles", "clinkers", "concrete", "gravel"}


def is_road(coll, style):
    if coll in ("wegdeel", "overbruggingsdeel"):
        return style in ROAD_STYLES
    if coll in ("ondersteunendwegdeel", "onbegroeidterreindeel"):
        return style in PAVED_STYLES
    return False


def load_polygons():
    colls = [
        "begroeidterreindeel",
        "onbegroeidterreindeel",
        "ondersteunendwaterdeel",
        "ondersteunendwegdeel",
        "waterdeel",
        "wegdeel",
        "overigbouwwerk",
        "pand",
        "overbruggingsdeel",
    ]
    polys = []
    for coll in colls:
        for f in json.load(open(RAW / f"bgt_{coll}.geojson"))["features"]:
            g = f["geometry"]
            if g is None:
                continue
            parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"] if g["type"] == "MultiPolygon" else []
            for rings in parts:
                ext = np.array(rings[0])[:, :2]
                area = 0.5 * abs(np.dot(ext[:, 0], np.roll(ext[:, 1], 1)) - np.dot(ext[:, 1], np.roll(ext[:, 0], 1)))
                # Bridges sit over water and are drawn last; everything else by size, large first.
                prio = 2 if coll == "overbruggingsdeel" else (1 if coll in ("pand", "overigbouwwerk") else 0)
                polys.append((prio, -area, coll, f["properties"], rings))
    polys += changed_ground()
    polys.sort(key=lambda t: (t[0], t[1]))
    return polys


def ground_changes():
    lm = BUILD / "landmarks.json"
    return json.loads(lm.read_text()).get("ground", []) if lm.exists() else []


def changed_ground():
    """Polygons for landmarks.json "ground": each area as a lawn under everything else, and the OpenStreetMap cycle
    paths through it, 3 m wide, which the BGT doesn't have yet."""
    from shapely.geometry import LineString, Polygon

    from geo import wgs_to_rd

    out = []
    roads = json.load(open(RAW / "osm_roads.json"))["elements"] if (RAW / "osm_roads.json").exists() else []
    for g in ground_changes():
        area = Polygon(g["area"])
        out.append((-1, -area.area, "park", {"lokaal_id": g["name"]}, [list(area.exterior.coords)]))
        for w in roads:
            t = w.get("tags", {})
            if w["type"] != "way" or not w.get("geometry") or t.get("highway") != "cycleway" or t.get("bridge"):
                continue
            line = LineString([wgs_to_rd(p["lat"], p["lon"]) for p in w["geometry"]])
            part = line.intersection(area).buffer(1.5, cap_style="flat")
            for poly in getattr(part, "geoms", [part]):
                if poly.geom_type == "Polygon" and not poly.is_empty:
                    out.append((0, -poly.area, "wegdeel", {"functie": "fietspad", "lokaal_id": f"osm{w['id']}"}, [list(poly.exterior.coords)]))
    return out


def era_style(coll, props, rings, style):
    """In an earlier era: what lay where today's polygon lies, when that is in the changed area (scripts/era.py). A
    building that stood then keeps its own."""
    from shapely.geometry import Point

    if ERA is None or not changed().contains(Point(np.asarray(rings[0])[:, :2].mean(0))):
        return style
    if coll == "pand" and props.get("bag_pnd") in standing():
        return style
    return "bare" if bare().contains(Point(np.asarray(rings[0])[:, :2].mean(0))) else "grass"


@cache
def standing():
    """BAG ids of today's buildings that stood in the era too."""
    out = set()
    for line in open(RAW / "3dbag.jsonl"):
        f = json.loads(line)
        if not built_after(f["CityObjects"][f["id"]]["attributes"]):
            out.add(f["id"].rsplit(".", 1)[-1])
    return out


def changed_style(coll, props, rings, style, areas):
    """The style of a BGT polygon that lies mostly inside a changed area, else its own."""
    from shapely.geometry import Polygon

    if coll == "park":
        return "grass"
    poly = Polygon(np.asarray(rings[0])[:, :2]).buffer(0)
    if not poly.area or not any(Polygon(a["area"]).intersection(poly).area > 0.5 * poly.area for a in areas):
        return style
    if props.get("functie") == "spoorbaan":
        return style
    if coll == "wegdeel" and props.get("functie") == "fietspad":
        # The paths from OpenStreetMap are the ones there now; the BGT's is the old straight road through the park.
        return "cyclepath" if str(props.get("lokaal_id", "")).startswith("osm") else "grass"
    return "grass" if style in TO_PARK else style


def to_px(ring):
    a = np.asarray(ring)[:, :2]
    return list(zip(((a[:, 0] - XMIN) / GROUND_RES).tolist(), ((YMAX - a[:, 1]) / GROUND_RES).tolist()))


def value_noise(shape, cell, seed, octaves=1):
    """Smooth noise in [0,1] with features of about `cell` pixels, tileable over the whole raster."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(1, int(cell / (2**o)))
        g = rng.random((shape[0] // c + 2, shape[1] // c + 2)).astype(np.float32)
        z = ndi.zoom(g, c, order=1)[: shape[0], : shape[1]]
        out += amp * z
        total += amp
        amp *= 0.5
    return out / total


def main():
    aerial = load_aerial()
    polys = load_polygons()
    print(f"{len(polys)} polygons")

    ids = Image.new("I", (NX, NY), 0)
    d = ImageDraw.Draw(ids)
    for i, (_, _, _, _, rings) in enumerate(polys, start=1):
        d.polygon(to_px(rings[0]), fill=i)
        for hole in rings[1:]:
            d.polygon(to_px(hole), fill=0)
    ids = np.array(ids, dtype=np.int32)

    # Per polygon: median aerial color, ignoring tree canopy that hides the surface.
    chm = np.load(BUILD / "chm.npy").astype(np.float32)
    chm_hi = np.repeat(np.repeat(chm, 2, 0), 2, 1)
    lab = ids[::4, ::4].copy()
    lab[chm_hi[::4, ::4] > 2.0] = 0
    small = aerial[::4, ::4].astype(np.float32)
    idx = np.arange(1, len(polys) + 1)
    med = np.stack([ndi.median(small[..., c], lab, idx) for c in range(3)], -1)
    med = np.nan_to_num(med, nan=128.0)

    cats = np.zeros(len(polys) + 1, np.int32)
    roads = np.zeros(len(polys) + 1, bool)
    areas = ground_changes()
    base = np.zeros((len(polys) + 1, 3), np.float32)
    stripes_dir = np.zeros(len(polys) + 1, np.float32)
    for i, (_, _, coll, props, rings) in enumerate(polys, start=1):
        # Seeded per polygon, so a field keeps its crop when the map area changes.
        rng = np.random.default_rng(int(hashlib.md5(str(props.get("lokaal_id", i)).encode()).hexdigest()[:8], 16))
        s = tartan(coll, props, rings) or era_style(coll, props, rings, changed_style(coll, props, rings, style_for(coll, props), areas))
        cats[i] = CAT[s]
        roads[i] = is_road(coll, s)
        c = np.array(STYLES[s], np.float32)
        m = med[i - 1]
        if s == "field":
            # Pick a crop look per field; the photo decides whether it is green or bare.
            green = m[1] > m[0] + 6
            choice = rng.random()
            if green:
                c = np.array((128, 166, 72) if choice < 0.6 else (150, 176, 84), np.float32)
            elif choice < 0.55:
                c = np.array((152, 116, 80), np.float32) * (0.9 + 0.2 * rng.random())
            elif choice < 0.8:
                c = np.array((206, 184, 108), np.float32)
            else:
                c = np.array((136, 160, 76), np.float32)
        elif s == "meadow" and m[0] >= m[1] - 2:
            # Brown in the spring photo: the moor grass of the Peel, not a green pasture.
            s = "heath"
            cats[i] = CAT[s]
            c = 0.45 * np.array(STYLES["heath"], np.float32) + 0.55 * m * np.array((1.08, 1.02, 0.9), np.float32)
        elif s in ("meadow", "grass", "berm"):
            c = c * (0.94 + 0.12 * rng.random())
        elif s == "yard":
            c = 0.55 * m + 0.45 * c
        elif s == "heath":
            c = 0.5 * c + 0.5 * m
        base[i] = c
        ext = np.asarray(rings[0])[:, :2]
        ext = ext - ext.mean(0)
        w, v = np.linalg.eigh(ext.T @ ext)
        stripes_dir[i] = math.atan2(v[1, -1], v[0, -1])
    base[0] = STYLES["gap"]

    cat = cats[ids]
    img = base[ids]  # (N, N, 3) float32
    road = roads[ids]
    # Bare ground inside a changed area that no polygon of its own made park, and the grass the spring photo still
    # shows as bare sand: lawn as well.
    if areas:
        mask = Image.new("1", (NX, NY), 0)
        for g in areas:
            ImageDraw.Draw(mask).polygon(to_px(g["area"]), fill=1)
        bare = np.array(mask) & np.isin(cat, [CAT[k] for k in (*TO_PARK, "heath", "meadow")]) & ~road
        cat[bare], img[bare] = CAT["grass"], np.array(STYLES["grass"], np.float32)
        del mask, bare
    # The outer metre of a road stays clean: the photo's road edge never lines up exactly with the BGT polygon,
    # and its detail there turns a straight road edge into a blotchy one.
    road_rim = road & ~ndi.binary_erosion(road, iterations=4)

    yy, xx = np.ogrid[0:NY, 0:NX]
    yy, xx = yy.astype(np.float32), xx.astype(np.float32)
    # Field furrows along the long axis of each parcel.
    ang = stripes_dir[ids]
    field = cat == CAT["field"]
    along = xx * np.sin(ang) + yy * np.cos(ang)
    furrow = (np.sin(along * (2 * np.pi / 7.0)) > 0.3).astype(np.float32)
    img[field] *= (0.96 + 0.05 * furrow[field])[:, None]
    del along, furrow, ang

    # Orchards and tree nurseries: rows of small dark shrubs.
    orch = cat == CAT["orchard"]
    rows = ((xx.astype(np.int32) // 6) % 3 == 0) & ((yy.astype(np.int32) // 6) % 2 == 0)
    img[orch & rows] *= 0.72
    del rows

    n1 = value_noise((NY, NX), 20, 1, octaves=2)
    n2 = value_noise((NY, NX), 40, 2, octaves=2)
    soft = np.isin(cat, list(SOFT))
    img[soft] *= (0.96 + 0.08 * n1[soft])[:, None]
    grassy = np.isin(cat, [CAT[k] for k in ("grass", "meadow", "berm", "bank", "orchard")])
    img[grassy] *= (0.95 + 0.1 * n2[grassy])[:, None]
    shrub = np.isin(cat, [CAT["shrub"], CAT["forest"], CAT["marsh"]])
    img[shrub & (n1 > 0.62)] *= 0.9
    img[shrub & (n1 < 0.3)] *= 1.05
    heath = cat == CAT["heath"]
    purple = np.array((132, 104, 118), np.float32)
    straw = np.array((168, 150, 102), np.float32)
    img[heath & (n2 > 0.6)] = img[heath & (n2 > 0.6)] * 0.5 + purple * 0.5
    img[heath & (n1 < 0.35)] = img[heath & (n1 < 0.35)] * 0.6 + straw * 0.4
    del soft, grassy, shrub, heath

    # Yards: lawn where the photo is green, light paving elsewhere.
    yard = cat == CAT["yard"]
    sm = ndi.gaussian_filter(aerial.astype(np.float32), (6, 6, 0))
    greenish = ndi.median_filter((sm[..., 1] > sm[..., 0] + 2) & (sm[..., 1] >= sm[..., 2]), size=7)
    lawn = np.array((112, 156, 68), np.float32)
    paving = np.array((176, 170, 156), np.float32)
    img[yard & greenish] = lawn * (0.96 + 0.08 * n1[yard & greenish])[:, None]
    img[yard & ~greenish] = paving * (0.97 + 0.05 * n1[yard & ~greenish])[:, None]
    del sm, greenish, yard

    # Real objects from the aerial photo: parked cars, road markings, garden paths, sheds' surroundings.
    # Only distinct things come through (a strong local contrast), not the photo's grain.
    paved = np.isin(cat, [CAT[k] for k in ("asphalt", "motorway", "concrete_pavers", "tiles", "clinkers", "concrete",
                                           "gravel", "grass_pavers", "bridge")]) & ~road_rim
    yard = cat == CAT["yard"]
    lum = aerial.astype(np.float32).mean(-1)
    strong = np.abs(lum - ndi.gaussian_filter(lum, 5)) > 14
    strong = ndi.binary_opening(strong, iterations=1)  # objects, not single noisy pixels
    del lum
    for k in range(3):
        ch = aerial[..., k].astype(np.float32)
        hp = np.where(strong, np.clip(ch - ndi.gaussian_filter(ch, 5), -60, 60), 0)
        img[..., k][paved] += 0.7 * hp[paved]
        img[..., k][yard] += 0.3 * hp[yard]
        del ch, hp
    del paved, yard, strong

    # Keep a little of the real ground texture (tracks, paths, field patterns) on soft surfaces.
    lum = aerial.astype(np.float32).mean(-1)
    detail = lum - ndi.uniform_filter(lum, 9)
    detail = np.clip(detail, -18, 18)
    keep = np.isin(cat, [CAT[k] for k in ("field", "heath", "meadow", "marsh", "dirt", "sand", "gap")]) & ~road_rim
    img[keep] += (0.2 * detail[keep])[:, None]
    del lum, detail, keep

    # Pavement patterns.
    # Not on the roads: their layer is sampled by sharp-edged polygons, where a 1 m grid turns into moiré.
    pav = np.isin(cat, [CAT["tiles"], CAT["concrete_pavers"], CAT["clinkers"]]) & ~road
    grid = ((xx.astype(np.int32) % 4 == 0) | (yy.astype(np.int32) % 4 == 0))
    img[pav & grid] *= 0.97
    asph = np.isin(cat, [CAT["asphalt"], CAT["motorway"]])
    img[asph] *= (0.985 + 0.03 * n1[asph])[:, None]
    del pav, grid, asph

    # Water: ripples and a darker far bank, which is the side of the canal you look into.
    water = np.isin(cat, list(WATERY))
    ripple = (np.sin((xx * 0.7 + yy * 0.7) * 0.9 + n2 * 20) > 0.93) & (n1 > 0.55)
    img[water & ripple] = img[water & ripple] * 0.5 + np.array((140, 186, 214), np.float32) * 0.5
    land = ~water
    # The camera looks north-east, so the visible bank is the land to the north and east of the water.
    far = np.zeros_like(water)
    for k in range(1, 4):
        far |= np.roll(land, -k, axis=1) | np.roll(land, k, axis=0)
    bank = water & far
    img[bank] = np.array((92, 84, 62), np.float32)
    edge = water & ~bank & ndi.binary_dilation(land, iterations=1)
    img[edge] *= 0.8
    del ripple, far, bank, edge, xx, yy

    paint_osm(img, cat)
    # The lanes on the athletics track: a white line every 1.22 m in from its edge.
    track = cat == CAT["tartan"]
    if track.any():
        dist = ndi.distance_transform_edt(track) * GROUND_RES
        img[track & (dist > 0.4) & (np.mod(dist, 1.22) < 0.12)] = np.array((236, 232, 224), np.float32)
        del dist
    if ERA is not None:
        paint_era(img, cat, n1, n2)

    # Darken edges between unlike surfaces so plots read as crisp shapes. Roads get their edge from their polygon.
    hard = ~np.isin(cat, list(SOFT)) & ~road
    border = (cat != np.roll(cat, 1, 0)) | (cat != np.roll(cat, 1, 1))
    img[border & hard] *= 0.86
    del hard, border, road_rim

    # Baked shadows from buildings and trees, plus a contact shadow along every wall.
    shadow, contact = compute_shadows()
    img *= np.where(shadow[..., None], np.array((0.62, 0.66, 0.8), np.float32), 1.0)
    img[contact] *= 0.8
    del shadow, contact

    img = np.clip(img, 0, 255).astype(np.uint8)
    full_preview = img[::8, ::8].copy()
    # Two layers. The roads are drawn as polygons with hard edges and sample their own layer, in which each road
    # colour runs on outward, so a pixel on the edge never picks up the verge. The ground layer has the roads
    # filled in with the colours around them, so its soft texture doesn't smear grey past those hard edges.
    ri = ndi.distance_transform_edt(~road, return_distances=False, return_indices=True)
    road_img = img[ri[0], ri[1]]
    del ri
    gi = ndi.distance_transform_edt(road, return_distances=False, return_indices=True)
    img = img[gi[0], gi[1]]
    del gi
    save_chunks(img, BUILD / "ground")
    save_chunks(road_img, BUILD / "roads")
    del road_img
    Image.fromarray(full_preview).save(BUILD / "ground_preview.png")
    save_road_polygons(polys, roads, cats)
    print("ground done")


def save_chunks(img, out):
    if out.exists():
        for f in out.glob("*.png"):
            f.unlink()
    out.mkdir(exist_ok=True)
    px = int(CHUNK / GROUND_RES)
    for ix in range(CHUNKS_X):
        for iy in range(CHUNKS_Y):
            row = (CHUNKS_Y - 1 - iy) * px
            Image.fromarray(img[row : row + px, ix * px : ix * px + px]).save(out / f"{ix}_{iy}.png", optimize=False, compress_level=1)


def save_road_polygons(polys, roads, cats):
    """The road polygons and their kerbs in scene coordinates (x east, z south, from the center), for render/scene.js.

    The kerbs are the outlines of the road surfaces merged per kind of surface: so a road gets an edge where it
    meets the verge or a pavement of another kind, but not where the BGT happens to cut it into segments.
    """
    from shapely import STRtree
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    local = lambda ring: [(round(x - CX, 2), round(CY - y, 2)) for x, y, *_ in ring]
    rounded = lambda ring: [[round(x, 2), round(z, 2)] for x, z in ring.coords]
    extent = box(XMIN - CX, CY - YMAX, XMAX - CX, CY - YMIN)
    names = list(STYLES)
    shapes = [Polygon(local(rings[0]), [local(r) for r in rings[1:]]).buffer(0) for _, _, _, _, rings in polys]
    tree = STRtree(shapes)
    out, kinds = [], {}
    for i, (_, _, coll, _, rings) in enumerate(polys, start=1):
        if not roads[i]:
            continue
        shape = shapes[i - 1]
        if not shape.intersects(extent):
            continue
        # The BGT has overlaps (lawns lying on a footpath, roads on a viaduct over the motorway). The ground
        # raster paints them in list order, later ones on top, so the road keeps only what the raster shows:
        # minus every later polygon that isn't a road. Buildings stay out of it; they cover the road anyway.
        later = [shapes[j] for j in tree.query(shape) if j > i - 1 and not roads[j + 1] and names[cats[j + 1]] != "building"]
        if later:
            shape = shape.difference(unary_union(later))
        # Drop slivers thinner than 10 cm that the cut can leave along shared borders.
        shape = shape.buffer(-0.05, join_style="mitre").buffer(0.05, join_style="mitre")
        for part in getattr(shape, "geoms", [shape]):
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            out.append({"rings": [rounded(part.exterior), *map(rounded, part.interiors)], "bridge": coll == "overbruggingsdeel"})
            kinds.setdefault(names[cats[i]], []).append(part)

    edges = []
    for shapes in kinds.values():
        # Closing by a few centimetres fuses neighbours whose shared border doesn't match exactly. Not clipped to
        # the map: the renderer clips per chunk, and a clip here would draw a kerb along the edge of the map.
        merged = unary_union(shapes).buffer(0.05, join_style="mitre").buffer(-0.05, join_style="mitre")
        for part in getattr(merged, "geoms", [merged]):
            if part.geom_type != "Polygon":
                continue
            for ring in [part.exterior, *part.interiors]:
                edges.append([[round(x, 2), round(z, 2)] for x, z in ring.coords])
    (BUILD / "roads.json").write_text(json.dumps({"polygons": out, "edges": edges}, separators=(",", ":")))
    print(f"{len(out)} road polygons, {len(edges)} kerb lines")


def paint_era(img, cat, n1, n2):
    """The changed area of an earlier era, painted afresh so nothing of today's photo shows through: sand with tracks
    on the building site, and elsewhere lawn with paving round the buildings that stood there then."""
    def raster(geom, grow=0):
        mask = Image.new("1", (NX, NY), 0)
        d = ImageDraw.Draw(mask)
        for p in getattr(geom, "geoms", [geom]):
            if p.geom_type == "Polygon" and not p.is_empty:
                d.polygon(to_px(p.exterior.coords), fill=1)
                for hole in p.interiors:
                    d.polygon(to_px(hole.coords), fill=0)
        m = np.array(mask)
        return ndi.binary_dilation(m, iterations=grow) if grow else m

    ch = raster(changed()) & (cat != CAT["building"])
    site = ch & raster(bare())
    lawn = ch & ~site
    demolished = BUILD / "demolished.json"
    if demolished.exists():
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        rings = [Polygon(r).buffer(0) for r in json.loads(demolished.read_text())]
        paving = raster(unary_union(rings), grow=int(ERA_CONF.get("paving", 6) / GROUND_RES))
        blind = BUILD / "blind.json"
        if blind.exists():
            # Round the barracks where AHN2 is blind, the wider paving of a military terrain.
            area = unary_union([Polygon(r).buffer(0) for r in json.loads(blind.read_text())])
            paving |= raster(unary_union([r for r in rings if r.intersects(area)]), grow=int(15 / GROUND_RES))
        paved = lawn & paving
        lawn &= ~paving
        cat[paved], img[paved] = CAT["concrete"], np.array(STYLES["concrete"], np.float32) * (0.97 + 0.05 * n1[paved])[:, None]
    cat[lawn], img[lawn] = CAT["grass"], np.array(STYLES["grass"], np.float32) * (0.94 + 0.1 * n2[lawn])[:, None]
    # Sand worked over by machines: darker damp patches and the tracks of the shovels.
    tracks = (np.sin(np.arange(NX, dtype=np.float32)[None, :] * 0.35 + n2 * 40) > 0.9)
    cat[site] = CAT["bare"]
    img[site] = np.array(STYLES["bare"], np.float32) * (0.9 + 0.14 * n1[site])[:, None]
    img[site & (n2 > 0.62)] *= 0.86
    img[site & tracks] *= 0.92


@cache
def athletics():
    """Where OpenStreetMap has an athletics ground (leisure=track), grown by 40 m, in RD meters."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from geo import wgs_to_rd

    path = RAW / "osm_landmarks.json"
    parts = []
    for el in json.load(open(path))["elements"] if path.exists() else []:
        t, g = el.get("tags", {}), el.get("geometry")
        if t.get("leisure") == "track" and g and len(g) >= 4:
            parts.append(Polygon([wgs_to_rd(p["lat"], p["lon"]) for p in g]).buffer(40))
    return unary_union(parts) if parts else Polygon()


def tartan(coll, props, rings):
    """The BGT's synthetic surfaces on an athletics ground are the track: orange."""
    from shapely.geometry import Point

    if props.get("plus_fysiek_voorkomen") != "kunststof" or ERA is not None:
        return None
    return "tartan" if athletics().contains(Point(np.asarray(rings[0])[:, :2].mean(0))) else None


def paint_osm(img, cat):
    """Sports pitches, riding arenas and graveyards from OpenStreetMap, drawn onto the ground."""
    path = RAW / "osm_landmarks.json"
    if not path.exists():
        return
    from geo import wgs_to_rd

    for el in json.load(open(path))["elements"]:
        t = el.get("tags", {})
        g = el.get("geometry")
        if el["type"] != "way" or not g:
            continue
        rd = np.array([wgs_to_rd(p["lat"], p["lon"]) for p in g])
        px = [((x - XMIN) / GROUND_RES, (YMAX - y) / GROUND_RES) for x, y in rd]
        x0, y0 = np.floor(np.min(px, 0)).astype(int) - 4
        x1, y1 = np.ceil(np.max(px, 0)).astype(int) + 4
        if x1 < 0 or y1 < 0 or x0 >= NX or y0 >= NY:
            continue
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, NX), min(y1, NY)
        w, h = x1 - x0, y1 - y0
        area = Image.new("L", (w, h), 0)
        ImageDraw.Draw(area).polygon([(a - x0, b - y0) for a, b in px], fill=1)
        area = np.array(area) > 0
        lines = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(lines)
        sub = img[y0:y1, x0:x1]
        # The pitch's own axes, from its outline.
        pts = np.array(px) - (x0, y0)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]  # OSM repeats the first node to close the ring; it would pull the center toward it
        c = pts.mean(0)
        _, vecs = np.linalg.eigh((pts - c).T @ (pts - c))
        ua, va = vecs[:, -1], vecs[:, 0]
        hu = np.abs((pts - c) @ ua).max()
        hv = np.abs((pts - c) @ va).max()
        P = lambda a, b: tuple(c + a * hu * ua + b * hv * va)
        sport = t.get("sport")
        if t.get("landuse") == "cemetery":
            sub[area] = np.array((108, 150, 70), np.float32)
            gy, gx = np.mgrid[0:h, 0:w]
            along = (gx - c[0]) * ua[0] + (gy - c[1]) * ua[1]
            across = (gx - c[0]) * va[0] + (gy - c[1]) * va[1]
            stones = (np.mod(along, 7) < 2) & (np.mod(across, 12) < 3) & area
            sub[stones] = np.array((176, 176, 170), np.float32)
            path_m = (np.abs(along) < 3) | (np.abs(across) < 3)
            sub[area & path_m] = np.array((196, 186, 160), np.float32)
        elif sport == "tennis":
            sub[area] = np.array((190, 104, 72), np.float32)
            d.polygon([P(-0.8, -0.7), P(0.8, -0.7), P(0.8, 0.7), P(-0.8, 0.7)], outline=1, width=2)
            d.line([P(0, -0.8), P(0, 0.8)], fill=1, width=2)
            d.line([P(-0.45, -0.55), P(0.45, -0.55)], fill=1, width=1)
            d.line([P(-0.45, 0.55), P(0.45, 0.55)], fill=1, width=1)
        elif sport == "equestrian":
            sub[area] = np.array((214, 196, 154), np.float32)
            d.polygon([tuple(p) for p in pts], outline=1, width=2)
        elif t.get("leisure") == "pitch":
            # Football: mowing stripes and the full set of lines, scaled to the pitch.
            gy, gx = np.mgrid[0:h, 0:w]
            along = (gx - c[0]) * ua[0] + (gy - c[1]) * ua[1]
            stripe = (np.mod(along, 40) < 20) & area
            sub[area] = np.array((96, 156, 64), np.float32)
            sub[stripe] = np.array((110, 170, 72), np.float32)
            m = 0.94
            d.polygon([P(-m, -m), P(m, -m), P(m, m), P(-m, m)], outline=1, width=2)
            d.line([P(0, -m), P(0, m)], fill=1, width=2)
            r = 9.15 / GROUND_RES
            d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=1, width=2)
            for s_ in (-1, 1):
                d.polygon([P(s_ * m, -0.6), P(s_ * (m - 0.3), -0.6), P(s_ * (m - 0.3), 0.6), P(s_ * m, 0.6)], outline=1, width=2)
                d.polygon([P(s_ * m, -0.28), P(s_ * (m - 0.1), -0.28), P(s_ * (m - 0.1), 0.28), P(s_ * m, 0.28)], outline=1, width=2)
        else:
            continue
        sub[np.array(lines) > 0] = np.array((240, 240, 236), np.float32)


def compute_shadows(res=0.5):
    """Hard shadows at 0.5 m from building roofs and tree crowns, upsampled to the ground raster."""
    nx, ny = int((XMAX - XMIN) / res), int((YMAX - YMIN) / res)
    top = np.zeros((ny, nx), np.float32)
    bot = np.full((ny, nx), np.inf, np.float32)

    v = np.fromfile(BUILD / "buildings.bin", dtype=np.float32).reshape(-1, 12)
    tris = v.reshape(-1, 3, 12)
    tris = tris[tris[:, 0, 8] == 2]  # roofs
    for t in tris:
        col = (t[:, 0] + CX - XMIN) / res
        row = (YMAX - (CY - t[:, 2])) / res
        c0, c1 = int(max(col.min(), 0)), int(min(col.max() + 1, nx))
        r0, r1 = int(max(row.min(), 0)), int(min(row.max() + 1, ny))
        if c1 <= c0 or r1 <= r0:
            continue
        gy, gx = np.mgrid[r0:r1, c0:c1].astype(np.float32) + 0.5
        (x0, x1, x2), (y0, y1, y2) = col, row
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        c = 1 - a - b
        inside = (a >= -0.01) & (b >= -0.01) & (c >= -0.01)
        h = a * t[0, 1] + b * t[1, 1] + c * t[2, 1]
        sub = top[r0:r1, c0:c1]
        np.maximum(sub, np.where(inside, h, 0), out=sub)
        sb = bot[r0:r1, c0:c1]
        np.minimum(sb, np.where(inside, 0, np.inf), out=sb)

    trees = np.fromfile(BUILD / "trees.bin", dtype=np.float32).reshape(-1, 10)
    for x, z, h, r, base, *_ in trees:
        col, row = (x + CX - XMIN) / res, (YMAX - (CY - z)) / res
        rr = r / res
        c0, c1 = int(max(col - rr, 0)), int(min(col + rr + 1, nx))
        r0, r1 = int(max(row - rr, 0)), int(min(row + rr + 1, ny))
        if c1 <= c0 or r1 <= r0:
            continue
        gy, gx = np.mgrid[r0:r1, c0:c1].astype(np.float32) + 0.5
        dd = ((gx - col) ** 2 + (gy - row) ** 2) / (rr * rr)
        inside = dd < 1
        mid, half = (h + base) / 2, (h - base) / 2
        s = np.sqrt(np.clip(1 - dd, 0, 1)) * half
        sub = top[r0:r1, c0:c1]
        np.maximum(sub, np.where(inside, mid + s, 0), out=sub)
        sb = bot[r0:r1, c0:c1]
        np.minimum(sb, np.where(inside, mid - s, np.inf), out=sb)

    az, el = math.radians(SUN_AZIMUTH), math.radians(SUN_ELEVATION)
    dx, dy = math.sin(az), -math.cos(az)  # toward the sun, in columns and rows
    shadow = np.zeros((ny, nx), bool)
    maxh = float(top.max())
    steps = int(maxh / math.tan(el) / res) + 1
    for k in range(1, steps + 1):
        ox, oy = int(round(dx * k)), int(round(dy * k))
        z = k * res * math.tan(el)
        if z > maxh:
            break
        src_t = shift(top, ox, oy, 0)
        src_b = shift(bot, ox, oy, np.inf)
        shadow |= (src_t >= z) & (src_b <= z)
    occupied = top > 0.3
    shadow &= ~occupied
    contact = ndi.binary_dilation(occupied & (bot < 0.5) & (top > 1.5), iterations=1) & ~occupied
    up = lambda m: np.repeat(np.repeat(m, 2, 0), 2, 1)
    return up(shadow), up(contact)


def shift(a, ox, oy, fill):
    """out[r, c] = a[r + oy, c + ox]"""
    out = np.full_like(a, fill)
    h, w = a.shape
    rs, re = max(0, -oy), min(h, h - oy)
    cs, ce = max(0, -ox), min(w, w - ox)
    out[rs:re, cs:ce] = a[rs + oy : re + oy, cs + ox : ce + ox]
    return out


if __name__ == "__main__":
    main()
