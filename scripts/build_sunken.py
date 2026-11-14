# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-image", "tifffile", "imagecodecs", "pillow", "shapely"]
# ///
"""Roads and paths that run sunken under the railway, with the ramps down to them.

The BGT marks road parts that lie under something with relatieve_hoogteligging -1. From those (near the railway), the
area is grown along the connected road polygons for as long as the AHN terrain model (DTM) lies more than 0.4 m below
the local ground level (median DTM in a ring of 15 to 30 m around, leaving out roads, rails and buildings): the ramps.
The floor follows the DTM, as depth below the local ground level, which render/scene.js draws at y = 0.

Output: data/build/sunken.json (today's map only)
  [{"name", "area": [[X, Y], ...] RD polygon of the floor (without 8 m round the rails, where the rails stay visible),
    "grid": {"x0", "y0", "res", "w", "h", "z"}: depths in m below ground for the cells (x0 + (i + .5) res,
    y0 + (j + .5) res), z[j * w + i], j counting north; the polygon plus a few meters round it, 0 elsewhere}]
"""

import json
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import unary_union
from shapely import prepare

import tifffile

from config import BUILD, CHUNK, ERA, RAW
from geo import wgs_to_rd

RES = 0.5  # DTM resolution
G = 1.0  # floor grid resolution
THRESHOLD = 0.4  # m below the local ground that still counts as sunken
RAIL_BUFFER = 8.0
MAX_VERTS = 64
DEBUG = os.environ.get("SUNKEN_DEBUG")
SEED_RAIL = 60.0  # BGT "under" parts count when a railway is within this many meters
SEED_AT = [(173246.8, 449235.7), (173502.4, 449149.9), (174509.6, 448857.8)]  # the known underpasses, within 80 m


AHN = RAW / "ahn"
OLD_AREA = (172750, 447000)  # the corner of the map that the chunks were once numbered from


def dtm_window(x0, y0, w, h):
    """The DTM (0.5 m) of the RD box from (x0, y0), w x h cells, rows counting from the north. The chunks are named after
    their RD corner (scripts/config.py chunk_name) or, as before, after their index from the old corner."""
    out = np.full((h, w), np.nan, dtype=np.float32)
    px = int(CHUNK / RES)
    for cx in range(int((x0 - OLD_AREA[0]) // CHUNK * CHUNK + OLD_AREA[0]), int(x0 + w * RES), CHUNK):
        for cy in range(int((y0 - OLD_AREA[1]) // CHUNK * CHUNK + OLD_AREA[1]), int(y0 + h * RES), CHUNK):
            for name in (f"{cx}_{cy}", f"{(cx - OLD_AREA[0]) // CHUNK}_{(cy - OLD_AREA[1]) // CHUNK}"):
                if (AHN / f"dtm_05m_{name}.tif").exists():
                    break
            else:
                continue
            a = tifffile.imread(AHN / f"dtm_05m_{name}.tif").astype(np.float32)[:px, :px]
            a[(a > 1000) | (a < -100)] = np.nan
            # a: rows from the north edge of the chunk (cy + CHUNK); copy the overlap with the window.
            top = cy + CHUNK  # RD y of the chunk's north edge
            col0 = round((cx - x0) / RES)
            row0 = round(((y0 + h * RES) - top) / RES)
            r0, r1 = max(row0, 0), min(row0 + px, h)
            c0, c1 = max(col0, 0), min(col0 + px, w)
            if r1 > r0 and c1 > c0:
                out[r0:r1, c0:c1] = a[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0]
    return out


def features(name):
    return json.loads((RAW / f"bgt_{name}.geojson").read_text())["features"]


# Underpasses that the DTM doesn't show well, by hand: centreline in RD from the zero depth at the one end to the other,
# the depth profile as (distance along the line in m, depth in m) breakpoints given by the points of the line, and the
# half-width of the corridor of road surfaces that sinks. Points from OpenStreetMap: the Klinkenbergerweg between the
# crossroads with the Emmalaan/Akulaan and the Keetmolenweg, the new cycle tunnel from the end of the Oude
# Bennekomseweg to the Ariënsweg.
MANUAL = [
    {"name": "Klinkenbergerweg-tunnel", "half": 16.0,
     "line": [(174586, 448627), (174619, 448800), (174625.5, 448869), (174617, 449100)],
     "depths": [0.0, 5.0, 5.0, 0.0]},
    {"name": "Fietstunnel station", "half": 12.0,
     "line": [(174514, 448769), (174507, 448860), (174505, 448898), (174497, 449042)],
     "depths": [0.0, 3.0, 3.0, 0.0]},
]


def manual_areas(rails, parts):
    """The MANUAL underpasses as sunken.json entries: the road surfaces within the corridor round the centreline, without
    RAIL_BUFFER round the rails, the floor depth by the position along the centreline."""
    road = unary_union([g for g, _ in parts])
    rail_zone = rails.buffer(RAIL_BUFFER)
    out = []
    for m in MANUAL:
        line = LineString(m["line"])
        cum = [0.0]
        for a, b in zip(m["line"], m["line"][1:]):
            cum.append(cum[-1] + float(np.hypot(b[0] - a[0], b[1] - a[1])))
        corridor = line.buffer(m["half"], cap_style=2)
        floor = road.intersection(corridor).difference(rail_zone).buffer(0.3).buffer(-0.3)
        for p in [p for p in getattr(floor, "geoms", [floor]) if p.geom_type == "Polygon" and p.area >= 20]:
            tol = 0.5
            s = Polygon(p.exterior).simplify(tol)
            while len(s.exterior.coords) - 1 > MAX_VERTS:
                tol += 0.25
                s = Polygon(p.exterior).simplify(tol)
            s = s.buffer(0)
            if s.is_empty or s.geom_type != "Polygon":
                continue
            bx0, by0, bx1, by1 = s.buffer(4).bounds
            gx0, gy0 = np.floor(bx0), np.floor(by0)
            gw, gh = int(np.ceil(bx1 - gx0)), int(np.ceil(by1 - gy0))
            z = np.zeros((gh, gw), dtype=np.float32)
            near = s.buffer(3)
            prepare(near)
            for j in range(gh):
                for i in range(gw):
                    q = Point(gx0 + i + 0.5, gy0 + j + 0.5)
                    if near.contains(q):
                        z[j, i] = np.interp(line.project(q), cum, m["depths"])
            print(f"  {m['name']}: {len(s.exterior.coords) - 1} vertices, {s.area:.0f} m2, depth up to {z.max():.1f} m")
            out.append({
                "name": m["name"],
                "area": [[round(x, 1), round(y, 1)] for x, y in list(s.exterior.coords)[:-1]],
                "grid": {"x0": float(gx0), "y0": float(gy0), "res": G, "w": gw, "h": gh,
                         "z": [round(float(v), 2) for v in z.ravel()]},
            })
    return out


def drop_overlapping(auto, manual):
    """The automatically found underpasses without those that lie (partly) over a manual one."""
    zone = unary_union([Polygon(e["area"]) for e in manual]) if manual else Polygon()
    keep = [e for e in auto if Polygon(e["area"]).buffer(0).intersection(zone).area < 10]
    if len(keep) < len(auto):
        print(f"dropped {len(auto) - len(keep)} automatic underpass(es) overlapping a manual one")
    return keep


def main():
    if ERA is not None:
        (BUILD / "sunken.json").write_text("[]")
        return
    rails = unary_union([LineString([wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]])
                         for w in json.loads((RAW / "osm_rail.json").read_text())["elements"]
                         if w["type"] == "way" and w.get("geometry") and w["tags"].get("railway") == "rail"])
    parts = []  # (geometry, below?) of everything road-like
    for name in ("wegdeel", "ondersteunendwegdeel", "onbegroeidterreindeel"):
        for f in features(name):
            p = f["properties"]
            if f["geometry"] is None or p.get("eind_registratie"):
                continue
            if name == "onbegroeidterreindeel" and "verharding" not in (p.get("fysiek_voorkomen") or ""):
                continue
            if name == "ondersteunendwegdeel" and (p.get("functie") or "") not in ("berm", "") and (p.get("relatieve_hoogteligging") or 0) >= 0:
                continue
            parts.append((shape(f["geometry"]).buffer(0), (p.get("relatieve_hoogteligging") or 0) < 0))
    seeds = [g for g, below in parts if below and (g.distance(rails) < SEED_RAIL or any(g.distance(Point(s)) < 80 for s in SEED_AT))]
    print(f"{len(seeds)} BGT parts under something near the railway")
    manual = manual_areas(rails, parts)
    if not seeds:
        (BUILD / "sunken.json").write_text(json.dumps(manual, separators=(",", ":")))
        return
    roads = unary_union([g for g, _ in parts])


    def raster(geom, x0, y0, w, h):  # RD geometry to a mask on a RES grid with the top left at (x0, y0 + h * RES)
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        for g in getattr(geom, "geoms", [geom]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            d.polygon([((x - x0) / RES, (y0 + h * RES - y) / RES) for x, y in g.exterior.coords], fill=255)
            for r in g.interiors:
                d.polygon([((x - x0) / RES, (y0 + h * RES - y) / RES) for x, y in r.coords], fill=0)
        return np.array(img) > 0

    # Work per cluster of seeds, in a window of their bounds plus 150 m.
    clusters = []
    for g in sorted(seeds, key=lambda g: g.centroid.x):
        for c in clusters:
            if c["box"].buffer(60).intersects(g):
                c["geoms"].append(g)
                c["box"] = unary_union(c["geoms"]).envelope
                break
        else:
            clusters.append({"geoms": [g], "box": g.envelope})
    out = []
    for ci, c in enumerate(clusters):
        bx0, by0, bx1, by1 = c["box"].buffer(150).bounds
        x0, y0 = np.floor(bx0), np.floor(by0)
        w, h = int((bx1 - x0) / RES), int((by1 - y0) / RES)
        w, h = int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2)
        win = dtm_window(x0, y0, w, h)
        if np.isnan(win).all():
            print(f"cluster {ci}: no DTM")
            continue
        # 1 m raster: the DTM, the road mask, the rails and buildings.
        small = np.nanmean(win.reshape(h // 2, 2, w // 2, 2), axis=(1, 3))
        hh, ww = h // 2, w // 2
        sx0, sy0 = x0, y0
        wbox = box(x0, y0, x0 + w * RES, y0 + h * RES).buffer(5)
        road = raster(roads.intersection(wbox).buffer(0.3), x0, y0, w, h).reshape(hh, 2, ww, 2).any(axis=(1, 3))
        near_rail = raster(rails.intersection(wbox).buffer(RAIL_BUFFER), x0, y0, w, h).reshape(hh, 2, ww, 2).any(axis=(1, 3))
        seed = raster(unary_union(c["geoms"]), x0, y0, w, h).reshape(hh, 2, ww, 2).any(axis=(1, 3))
        # Ground level: median of the DTM in a ring 15 to 30 m round, without roads, rails and the seed parts.
        masked = np.where(road | near_rail | np.isnan(small), np.nan, small)
        step = 4
        ch, cw = hh // step, ww // step
        blocks = masked[: ch * step, : cw * step].reshape(ch, step, cw, step)
        with np.errstate(all="ignore"):
            cm = np.nanmedian(blocks, axis=(1, 3))
        R = int(30 / (step * G)) + 1
        padded = np.pad(cm, R, constant_values=np.nan)
        shifts = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if 15 <= np.hypot(dy, dx) * step * G <= 30:
                    shifts.append(padded[R + dy : R + dy + ch, R + dx : R + dx + cw])
        ground_c = np.nanmedian(np.stack(shifts), axis=0)
        # Where the ring is empty the cells stay NaN: take the nearest value.
        bad = np.isnan(ground_c)
        if bad.all():
            continue
        idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        ground_c = ground_c[tuple(idx)]
        ground = ndi.zoom(np.pad(ground_c, ((0, 1), (0, 1)), mode="edge"), step, order=1, mode="nearest")[:hh, :ww]
        ground = np.pad(ground, ((0, hh - ground.shape[0]), (0, ww - ground.shape[1])), mode="edge")
        depth = np.nan_to_num(ground - small, nan=0.0)
        depth = ndi.median_filter(depth, size=3)
        # Grow: road cells where the DTM lies more than THRESHOLD below the ground, connected to the seed parts.
        print(f'cluster {ci}: window {w}x{h}, ground {np.nanmedian(ground):.1f}, depth>0.4 on road {int((road & (depth > THRESHOLD)).sum())} cells, seed {int(seed.sum())}')
        cand = (road & (depth > THRESHOLD)) | seed
        lab, _ = ndi.label(cand, structure=np.ones((3, 3)))
        keep = np.unique(lab[seed & cand])
        region = np.isin(lab, keep[keep > 0])
        # Don't let it run off along a long road that merely lies low: stay within 120 m of a seed part.
        region &= ndi.distance_transform_edt(~seed) < 120
        region = ndi.binary_closing(region, iterations=1) | region
        region = ndi.binary_fill_holes(ndi.binary_opening(region, iterations=1) | seed)
        if not region.any():
            continue
        # Depth grid: DTM depth in the region (also under the rails), a few meters further out by the nearest value.
        d = np.where(region, np.clip(depth, 0, 8), 0.0)
        idx = ndi.distance_transform_edt(~region, return_distances=True, return_indices=True)
        near = d[tuple(idx[1])]
        d = np.where(idx[0] < 4, near, 0.0)
        d = ndi.gaussian_filter(d, 1.5)
        d[d < 0.05] = 0
        d = d[::-1]  # rows count north from here on
        # Polygons of the floor: the region without the rails, one entry per piece.
        floor_mask = region & ~near_rail
        if DEBUG:
            rgb = np.stack([np.clip(depth * 40, 0, 255), road * 120, near_rail * 255], -1).astype(np.uint8)
            rgb[region] = (rgb[region] * 0.4 + np.array([0, 150, 0]) * 0.6).astype(np.uint8)
            rgb[seed] = (255, 255, 0)
            Image.fromarray(rgb).save(DEBUG + f'/c{ci}.png')
            print(f'cluster {ci} origin {x0} {y0}, region {int(region.sum())}, floor {int(floor_mask.sum())}')
        lab, n = ndi.label(floor_mask)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < 40:
                continue
            m = ndi.binary_dilation(m, iterations=1) & region
            cells = [Polygon([(sx0 + i, sy0 + j), (sx0 + i + 1, sy0 + j), (sx0 + i + 1, sy0 + j + 1), (sx0 + i, sy0 + j + 1)])
                     for j, i in zip(*np.nonzero(m[::-1]))]
            poly = unary_union(cells)
            polys = [p for p in getattr(poly, "geoms", [poly]) if p.area >= 40]
            for p in polys:
                p = Polygon(p.exterior).buffer(1.2, join_style=1).buffer(-1.2, join_style=1)
                if p.is_empty or p.geom_type != "Polygon":
                    continue
                tol = 0.7
                s = p.simplify(tol)
                while len(s.exterior.coords) - 1 > MAX_VERTS:
                    tol += 0.25
                    s = p.simplify(tol)
                s = s.buffer(0)
                if s.is_empty or s.geom_type != "Polygon":
                    continue
                bx0_, by0_, bx1_, by1_ = s.buffer(4).bounds
                gx0, gy0 = np.floor(bx0_), np.floor(by0_)
                gw, gh = int(np.ceil(bx1_ - gx0)), int(np.ceil(by1_ - gy0))
                z = np.zeros((gh, gw), dtype=np.float32)
                j0, i0 = int(gy0 - sy0), int(gx0 - sx0)
                sub = d[max(j0, 0) : j0 + gh, max(i0, 0) : i0 + gw]
                z[max(-j0, 0) : max(-j0, 0) + sub.shape[0], max(-i0, 0) : max(-i0, 0) + sub.shape[1]] = sub
                cx, cy = s.centroid.x, s.centroid.y
                print(f"  piece at ({cx:.0f}, {cy:.0f}): {len(s.exterior.coords) - 1} vertices, {s.area:.0f} m2, depth up to {z.max():.1f} m")
                out.append({
                    "name": f"Onderdoorgang {len(out) + 1}",
                    "area": [[round(x, 1), round(y, 1)] for x, y in list(s.exterior.coords)[:-1]],
                    "grid": {"x0": float(gx0), "y0": float(gy0), "res": G, "w": gw, "h": gh,
                             "z": [round(float(v), 2) for v in z.ravel()]},
                })
    out = drop_overlapping(out, manual) + manual
    (BUILD / "sunken.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out)} sunken areas")


if __name__ == "__main__":
    main()
