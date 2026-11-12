"""Buildings that stood in an earlier era but have been demolished since, traced from the Kadaster's topographic map
of that year (scripts/topo_fetch.py, Topotijdreis, CC BY 4.0 Kadaster).

The map draws houses and barracks black and the big industrial complexes grey. Only near the changed area
(era.changed), where today's buildings replaced the old ones, so that the names and roads printed on the map elsewhere
are left alone; and only what is solid, has straight sides and some width, so that letters, railway lines and the
casings of roads drop out. A building that AHN2 still saw keeps its measured height; the others get one by size.
"""

import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from skimage import measure

from config import RAW
from era import changed


def heights():
    """AHN2's height above the ground under a polygon in RD meters (nan where it has none)."""
    import ahn2_buildings
    from config import XMIN, YMAX

    dsm, dtm = ahn2_buildings.mosaic("dsm_05m"), ahn2_buildings.mosaic("dtm_05m")
    bad = np.isnan(dtm)
    dtm = dtm[tuple(ndi.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    ndsm = dsm - dtm
    res = ahn2_buildings.RES

    def at(poly):
        x0, y0, x1, y1 = poly.bounds
        c0, c1 = int((x0 - XMIN) / res), int((x1 - XMIN) / res) + 1
        r0, r1 = int((YMAX - y1) / res), int((YMAX - y0) / res) + 1
        win = ndsm[max(r0, 0) : max(r1, 0), max(c0, 0) : max(c1, 0)]
        win = win[np.isfinite(win)]
        return float(np.median(win)) if win.size else np.nan

    return at


def solid_parts(geom):
    """Polygons without holes: a polygon around a building that stands today is cut in two through each hole, as the
    renderer extrudes outlines only."""
    todo, parts = list(getattr(geom, "geoms", [geom])), []
    while todo:
        p = todo.pop()
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        if not p.interiors:
            parts.append(p)
            continue
        cx = p.interiors[0].centroid.x
        x0, y0, x1, y1 = p.bounds
        for half in (box(x0 - 1, y0 - 1, cx, y1 + 1), box(cx, y0 - 1, x1 + 1, y1 + 1)):
            piece = p.intersection(half)
            todo += [q for q in getattr(piece, "geoms", [piece]) if q.geom_type == "Polygon"]
    return parts


def halls(p, width=30.0, alley=3.0):
    """A big complex as the map draws it, one grey patch, cut into halls of `width` meters along its long side with
    alleys between them, as the factory was built: hall after hall."""
    if p.area < 3000:
        return [p]
    from shapely import affinity

    rect = p.minimum_rotated_rectangle
    xy = np.asarray(rect.exterior.coords)
    edge = xy[1] - xy[0] if np.hypot(*(xy[1] - xy[0])) < np.hypot(*(xy[2] - xy[1])) else xy[2] - xy[1]
    ang = np.degrees(np.arctan2(edge[1], edge[0]))
    c = p.centroid
    flat = affinity.rotate(p, -ang, origin=c)
    x0, y0, x1, y1 = flat.bounds
    out = []
    for x in np.arange(x0, x1, width + alley):
        piece = flat.intersection(box(x, y0 - 1, x + width, y1 + 1))
        piece = affinity.rotate(piece, ang, origin=c)
        out += [q for q in getattr(piece, "geoms", [piece]) if q.geom_type == "Polygon"]
    return out


# Per map: the grey of its industrial buildings (the mean over 7 pixels; 1987 prints a dot screen, 2008 a flat grey),
# and the height of a building that no AHN measured: black ones by size, or one height for all (the barracks blocks of
# 2008, two or three storeys).
STYLE = {
    1987: {"grey": (95, 172), "black": None},
    2008: {"grey": (150, 186), "black": 10.0},
}


def default_height(area, grey, year):
    if grey:
        return 9.0 if area > 1500 else 7.0
    if STYLE[year]["black"]:
        return STYLE[year]["black"]
    return 9.0 if area > 1500 else 7.0 if area > 300 else 4.5


def find(kept_rings, year=1987, within=None):
    """[(ring in RD meters, height), ...] of the buildings on the map of `year` that are gone today, near the changed
    area or, given, within that RD geometry."""
    d = RAW / f"topo{year}"
    meta = json.loads((d / "mosaic.json").read_text())
    a = np.asarray(Image.open(d / "mosaic.png").convert("RGB")).astype(np.int16)
    x0, y0, res = meta["x0"], meta["y0"], meta["res"]
    v, sat = a.mean(-1), a.max(-1) - a.min(-1)
    dark = ndi.binary_opening((v < 95) & (sat < 35), iterations=1)
    # The grey fill is a dot screen: grey on average over a few pixels. A road's grey casing lies between white road
    # and white ground, so its average is lighter.
    mean = ndi.uniform_filter(v.astype(np.float32), 7)
    lo, hi = STYLE[year]["grey"]
    grey = (mean >= lo) & (mean < hi) & (ndi.uniform_filter(sat.astype(np.float32), 7) < 18)
    grey = ndi.binary_opening(grey, iterations=3)

    near = changed().buffer(30) if within is None else within
    kept = unary_union([Polygon(r).buffer(2) for r in kept_rings if len(r) >= 3])
    ahn = heights()
    found = []
    for mask, is_grey, min_area in ((dark, False, 40), (grey, True, 800)):
        blobs, n = ndi.label(mask)
        for sl, k in zip(ndi.find_objects(blobs), range(1, n + 1)):
            blob = blobs[sl] == k
            filled = ndi.binary_fill_holes(blob)
            # A road's casing encloses the road: hollow. A building is solid.
            if blob.sum() < 0.75 * filled.sum():
                continue
            contours = measure.find_contours(np.pad(filled, 1).astype(np.float32), 0.5)
            if not contours or len(max(contours, key=len)) < 4:
                continue
            contour = max(contours, key=len)
            r0, c0 = sl[0].start - 1, sl[1].start - 1
            xy = np.stack([x0 + (contour[:, 1] + c0 + 0.5) * res, y0 - (contour[:, 0] + r0 + 0.5) * res], 1)
            poly = Polygon(xy).simplify(res * 0.8).buffer(0)
            if is_grey:
                # Off with the ribbons of track and road that cling to a complex.
                poly = poly.buffer(-6, join_style="mitre").buffer(6, join_style="mitre")
                poly = max(getattr(poly, "geoms", [poly]), key=lambda q: q.area) if not poly.is_empty else poly
            if poly.is_empty or poly.geom_type != "Polygon" or poly.area < min_area or not near.intersects(poly):
                continue
            rect = poly.minimum_rotated_rectangle
            sides = sorted(np.hypot(*np.diff(np.asarray(rect.exterior.coords), axis=0).T)[:2])
            fill = poly.area / rect.area
            # Letters, rails and road casings: thin, long, or far from rectangular. A large grey complex may have
            # any shape, as long as it is not a ring of road.
            if sides[0] < 6 or fill < 0.45:
                continue
            if poly.area < 3000 and (sides[1] > 5 * sides[0] or fill < (0.7 if is_grey else 0.75)):
                continue
            if fill > 0.85:
                poly = rect
            for whole in solid_parts(poly.intersection(near).difference(kept)):
                for i, p in enumerate(halls(whole) if is_grey else [whole]):
                    # What is left of a building of today after cutting it out: slivers, not buildings.
                    xy = np.asarray(p.minimum_rotated_rectangle.exterior.coords)
                    if p.area < 12 or min(np.hypot(*np.diff(xy, axis=0).T)[:2]) < 3:
                        continue
                    h = ahn(p)
                    if not (3 <= h <= 30):
                        h = default_height(p.area, is_grey, year) + (1.5 if i % 3 == 1 else -1.0 if i % 3 == 2 else 0.0)
                    found.append((np.asarray(p.exterior.coords)[:-1], round(h, 1)))
    return found
