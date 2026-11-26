"""Buildings that stood in an earlier era and are gone since, from the BAG itself (Kadaster, CC0): the panden with status
"Pand gesloopt" (data/raw/bag_gesloopt.json, scripts/fetch.py fetch_bag_demolished) with their footprint and year of
construction. Neither AHN2 nor the topographic map see the barracks of Ede well (the military terrain has no AHN2, the map
shows hardly any building on it), but the BAG has every shed and block there, as exact as the buildings of today.

What the BAG does not have is a height. None was measured on the military terrain, and the shadows in the aerial photo
(tried: the sun at azimuth 45 degrees and 43 degrees up, from the Enka chimney's shadow) gave worse heights than a constant
on 99 standing buildings of known height. So a height comes from what the roof looks like in the photo of the era:
  tile red or green roof, or two roof slopes lit differently  -> a pitched roof: a house, or a barracks block of 3 storeys
  anything else, wide enough                                 -> a flat shed (barn), 4.5-6.5 m by its width
  under 40 m2                                                -> a hut (garage, guard post), 2.8 m
and where AHN2 did measure it, that height within a band round the default (the lidar sees tree crowns on a garage).

A building that today's 3D BAG still has, or that AHN2, the topographic map or OpenStreetMap found before, is not added
twice: on the barracks terrains the BAG outline replaces what the map traced (the map draws a blur of a block), elsewhere
what AHN2 measured stays. Its roof colour is the median of the era's own aerial photo over the footprint.
"""

import json

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from aerial import load_aerial, world_to_px
from config import ERA_CONF, GROUND_RES, OLD_TERRAIN, RAW, XMAX, XMIN, YMAX, YMIN

# What replaces a shape that was found from the map: sources that trace outlines, not measure them.
REPLACED = ("topo", "barracks", "osm")
PITCHED = 0.07  # the two halves of a pitched roof differ by this much in brightness (a flat roof by 0.01-0.03)

_known = []  # [(polygon, bouwjaar, roof rgb)] of what merge() added, for info()


def photo_year():
    return ERA_CONF.get("photo") or 2009


def _photo():
    return load_aerial(f"aerial-{photo_year()}")


def _mask(poly, shape):
    img = Image.new("1", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for g in getattr(poly, "geoms", [poly]):
        if g.geom_type == "Polygon" and not g.is_empty:
            d.polygon([(x, y) for x, y in world_to_px(np.asarray(g.exterior.coords)[:, :2])], fill=1)
    return np.array(img)


def _inside(poly, photo):
    """The photo's pixels under a polygon, shrunk 1 m so a roof that leans in the photo takes less of the ground round it."""
    small = poly.buffer(-1.0)
    if small.is_empty or small.area < 6:
        small = poly.buffer(-0.4)
    if small.is_empty:
        small = poly
    x0, y0, x1, y1 = small.bounds
    c0, r0 = world_to_px([(x0, y1)])[0].astype(int)
    c1, r1 = world_to_px([(x1, y0)])[0].astype(int)
    c0, r0, c1, r1 = max(c0 - 1, 0), max(r0 - 1, 0), min(c1 + 2, photo.shape[1]), min(r1 + 2, photo.shape[0])
    if c1 <= c0 or r1 <= r0:
        return None
    m = _mask(small, photo.shape)[r0:r1, c0:c1]
    return photo[r0:r1, c0:c1][m] if m.sum() >= 6 else None


def roof_rgb(poly, photo=None):
    """The median colour of the era's photo over the footprint, or None."""
    px = _inside(poly, _photo() if photo is None else photo)
    return None if px is None else [int(v) for v in np.median(px, axis=0)]


def pitch(poly, photo):
    """How much the two halves of the roof, either side of its long axis, differ in brightness: a pitched roof lies lit
    on one side and in shade on the other, a flat one is even. None where the photo has too little of it."""
    xy = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    e = [xy[1] - xy[0], xy[2] - xy[1]]
    v = max(e, key=lambda q: np.hypot(*q))
    v = v / np.hypot(*v)
    n = np.array([-v[1], v[0]])
    c, big = np.array(poly.centroid.coords[0]), 1000.0
    halves = [Polygon([c + v * big + n * big * s, c - v * big + n * big * s, c - v * big, c + v * big]) for s in (1, -1)]
    lum = []
    for h in halves:
        px = _inside(poly.intersection(h), photo)
        if px is None:
            return None
        lum.append(np.median(px @ np.float32([0.299, 0.587, 0.114])))
    return abs(lum[0] - lum[1]) / max((lum[0] + lum[1]) / 2, 1)


def is_tile(rgb):
    """Red or brown roof tiles, or the green of the barracks' painted roofs."""
    r, g, b = rgb
    return (r - g >= 12 and r - b >= 22) or (g - r >= 5 and g - b >= 8)


def style(poly, rgb, p):
    """(kind, height in meters) for a footprint from its shape and roof."""
    xy = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    short, long = sorted(np.hypot(*np.diff(xy, axis=0).T)[:2])
    area = poly.area
    if area < 40:
        return "hut", 2.8
    pitched = (rgb is not None and is_tile(rgb)) or (p is not None and p >= PITCHED)
    if pitched:
        if area < 250:
            return "house", 6.5
        return "house", 9.0 if short >= 11 else 7.5
    if area < 100 or short < 7:
        return "shed", 3.5
    return "barn", round(float(np.clip(4.5 + 0.12 * (short - 8), 4.5, 6.5)), 1)


def measured():
    """AHN2's height above the ground under a polygon (RD meters), nan where it has no data for most of it: on the military
    terrain it has only scattered ground cells, which would read as a height of 0."""
    import ahn2_buildings
    from scipy import ndimage as ndi

    dsm, dtm = ahn2_buildings.mosaic("dsm_05m"), ahn2_buildings.mosaic("dtm_05m")
    dtm = dtm[tuple(ndi.distance_transform_edt(np.isnan(dtm), return_distances=False, return_indices=True))]
    ndsm, res = dsm - dtm, ahn2_buildings.RES

    def at(poly):
        x0, y0, x1, y1 = poly.bounds
        c0, r0 = max(int((x0 - XMIN) / res), 0), max(int((YMAX - y1) / res), 0)
        c1, r1 = int((x1 - XMIN) / res) + 1, int((YMAX - y0) / res) + 1
        win = ndsm[r0:r1, c0:c1]
        if not win.size:
            return np.nan
        img = Image.new("1", (win.shape[1], win.shape[0]), 0)
        ImageDraw.Draw(img).polygon([((x - XMIN) / res - c0, (YMAX - y) / res - r0) for x, y in poly.exterior.coords], fill=1)
        inside = np.array(img)
        vals = win[inside & np.isfinite(win)]
        return float(np.median(vals)) if vals.size >= 0.6 * inside.sum() and vals.size else np.nan

    return at


def zone():
    """The barracks terrains of config.OLD_TERRAIN (the second and third); the Enka terrain, the first, has its tall halls
    measured and traced already."""
    return unary_union([Polygon(t) for t in OLD_TERRAIN[1:]]).buffer(20)


def candidates(kept_polys):
    """The demolished BAG panden of the era that no building of today still covers: [(polygon, bouwjaar)]."""
    from topo_buildings import solid_parts

    built = ERA_CONF["built"]
    tree = STRtree(kept_polys) if kept_polys else None
    out = []
    for b in json.loads((RAW / "bag_gesloopt.json").read_text()):
        if not b.get("bouwjaar") or b["bouwjaar"] > built:
            continue
        p = Polygon(b["ring"]).buffer(0)
        if p.geom_type != "Polygon" or p.area < 12 or not (XMIN <= p.centroid.x <= XMAX and YMIN <= p.centroid.y <= YMAX):
            continue
        if tree is not None and sum(p.intersection(kept_polys[i]).area for i in tree.query(p)) >= 0.3 * p.area:
            continue
        for q in solid_parts(p):  # one polygon without a hole per piece
            out.append((q, b["bouwjaar"]))
    # The BAG has the same building twice now and then (a temporary one on the same ground): the larger one stays.
    out.sort(key=lambda c: -c[0].area)
    kept_out = []
    for p, y in out:
        if not any(p.intersection(q).area >= 0.5 * p.area for q, _ in kept_out if p.intersects(q)):
            kept_out.append((p, y))
    return kept_out


def merge(found, kept_rings):
    """`found` ([(ring, height, source)] of what AHN2, the map and OpenStreetMap saw) with the demolished BAG panden added.
    On the barracks terrains the BAG outline replaces what was traced from a map (a tenth overlap is enough); a pand that something measured already
    is not added. `kept_rings` are the buildings of today that stood then."""
    kept = [Polygon(r).buffer(0) for r in kept_rings if len(r) >= 3]
    cands = candidates(kept)
    barracks = zone()
    ahn = measured()
    photo = _photo()
    old = [(Polygon(ring).buffer(0), (ring, h, src)) for ring, h, src in found]
    # Traced from a map, on the barracks terrains: gone where the BAG covers a tenth of it (a traced blur next to a real outline is a ghost).
    cover = unary_union([p for p, _ in cands]) if cands else Polygon()
    drop = {i for i, (p, (_, _, src)) in enumerate(old)
            if src in REPLACED and barracks.intersects(p.representative_point()) and p.area > 0 and p.intersection(cover).area >= 0.1 * p.area}
    rest = [(p, f) for i, (p, f) in enumerate(old) if i not in drop]
    tree = STRtree([p for p, _ in rest]) if rest else None
    added = []
    _known.clear()
    for p, year in cands:
        # What AHN2 or OpenStreetMap measured stays.
        if tree is not None and sum(p.intersection(rest[i][0]).area for i in tree.query(p)) >= 0.3 * p.area:
            continue
        rgb = roof_rgb(p, photo)
        kind, h = style(p, rgb, pitch(p, photo) if p.area >= 40 else None)
        m = ahn(p)
        if np.isfinite(m) and 3 <= m <= 30:
            h = round(float(np.clip(m, 0.7 * h, 1.5 * h)), 1)
        elif np.isfinite(m) and m < 0.8 and p.area >= 60:
            continue  # AHN2 saw bare ground there: gone before its flight
        ring = np.asarray(p.exterior.coords)[:-1]
        added.append((ring, h, f"bag-{kind}"))
        _known.append((p, year, rgb))
    print(f"{len(cands)} demolished BAG panden of the era, {len(added)} added, {len(drop)} traced outlines replaced")
    return [f for _, f in rest] + added


def info(ring):
    """(bouwjaar, roof colour) of the BAG pand that merge() made this ring of, or None."""
    pt = Polygon(ring).buffer(0).representative_point()
    return next(((y, rgb) for p, y, rgb in _known if p.contains(pt)), None)
