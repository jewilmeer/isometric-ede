# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-image", "tifffile", "imagecodecs", "pillow", "shapely"]
# ///
"""Find individual trees in the AHN canopy height model.

Output (data/build/):
  trees.bin   float32 per tree: x, z, height, crown radius, crown base, kind, r, g, b, seed
  chm.npy     canopy height model at 0.5 m (row 0 north), used for hedges and shadows
"""

import json

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from aerial import load_aerial
from era import AHN
from config import ERA, chunk_name, BUILD, CHUNK, CX, CY, GROUND_RES, NX, NY, RAW, XMIN, YMAX

RES = 0.5


def mosaic(cov, source=AHN):
    px = int(CHUNK / RES)
    out = np.full((NY * px, NX * px), np.nan, dtype=np.float32)
    for ix in range(NX):
        for iy in range(NY):
            a = tifffile.imread(source / f"{cov}_{chunk_name(ix, iy)}.tif").astype(np.float32)
            a[(a > 1000) | (a < -100)] = np.nan
            row = (NY - 1 - iy) * px
            out[row : row + px, ix * px : ix * px + px] = a[:px, :px]
    return out


def fill_nearest(a):
    bad = np.isnan(a)
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def footprint_mask(shape, grow_m):
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for ring in json.load(open(BUILD / "footprints.json")):
        d.polygon([((x + CX - XMIN) / RES, (YMAX - (CY - z)) / RES) for x, z in ring], fill=255)
    m = np.array(img) > 0
    return ndi.binary_dilation(m, iterations=max(1, int(grow_m / RES)))


def main():
    dsm = mosaic("dsm_05m")
    dtm = fill_nearest(mosaic("dtm_05m"))
    chm = np.nan_to_num(dsm - dtm, nan=0.0)
    chm[chm < 0] = 0
    # Where the era's AHN measured nothing (blind.json, written by build_buildings.py: the barracks terrain in AHN2),
    # the trees of today stand in for those of then, except where the place has been rebuilt since (era.changed).
    blind = BUILD / "blind.json"
    if ERA is not None and blind.exists():
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        from era import changed

        area = unary_union([Polygon(r).buffer(0) for r in json.loads(blind.read_text())]).difference(changed())
        img = Image.new("L", (chm.shape[1], chm.shape[0]), 0)
        for p in getattr(area, "geoms", [area]):
            if p.geom_type == "Polygon" and not p.is_empty:
                ImageDraw.Draw(img).polygon([((x - XMIN) / RES, (YMAX - y) / RES) for x, y in p.exterior.coords], fill=255)
        inside = np.array(img) > 0
        now = mosaic("dsm_05m", RAW / "ahn") - fill_nearest(mosaic("dtm_05m", RAW / "ahn"))
        chm[inside] = np.clip(np.nan_to_num(now[inside], nan=0.0), 0, None)
    chm[footprint_mask(chm.shape, 1.0)] = 0
    np.save(BUILD / "chm.npy", chm.astype(np.float16))

    smooth = ndi.gaussian_filter(chm, 1.2)
    canopy = smooth > 2.5
    peaks = (smooth == ndi.maximum_filter(smooth, size=5)) & (smooth > 3.5)
    ys, xs = np.nonzero(peaks)
    hs = smooth[ys, xs]

    # Greedy suppression: taller trees claim a radius that grows with their height.
    order = np.argsort(-hs)
    taken = np.zeros_like(peaks)
    keep = []
    for i in order:
        y, x, h = ys[i], xs[i], hs[i]
        if taken[y, x]:
            continue
        keep.append(i)
        r = int(np.clip(0.12 * h + 1.0, 1.5, 4.5) / RES)
        taken[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = True
    ys, xs, hs = ys[keep], xs[keep], hs[keep]

    markers = np.zeros(chm.shape, dtype=np.int32)
    markers[ys, xs] = np.arange(1, len(ys) + 1)
    labels = watershed(-smooth, markers, mask=canopy)
    areas = ndi.sum_labels(np.ones_like(labels), labels, index=np.arange(1, len(ys) + 1)) * RES * RES
    radius = np.clip(np.sqrt(areas / np.pi), 1.0, 7.5)

    # Crown color from the aerial photo: evergreens stay green in the (spring) photo.
    aerial = load_aerial().astype(np.float32)
    scale = int(RES / GROUND_RES)
    small = aerial.reshape(aerial.shape[0] // scale, scale, aerial.shape[1] // scale, scale, 3).mean(axis=(1, 3))
    cols = np.stack([ndi.median(small[..., c], labels, index=np.arange(1, len(ys) + 1)) for c in range(3)], -1)
    cols = np.nan_to_num(cols, nan=90.0)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    evergreen = (g > r + 4) & (g > b + 4) & (g < 110)

    rng = np.random.default_rng(7)
    wx = XMIN + (xs + 0.5) * RES
    wy = YMAX - (ys + 0.5) * RES
    kind = evergreen.astype(np.float32)
    base = np.clip(hs * rng.uniform(0.25, 0.4, len(hs)), 1.2, 6)
    trees = np.stack(
        [wx - CX, CY - wy, hs, radius, base, kind, r / 255, g / 255, b / 255, rng.random(len(hs))], -1
    ).astype(np.float32)

    # Municipal trees from the BGT that the canopy model missed (young street trees); today's, so not in an earlier era.
    bgt = json.load(open(RAW / "bgt_vegetatieobject_punt.geojson"))["features"] if ERA is None else []
    pts = np.array([f["geometry"]["coordinates"][:2] for f in bgt]).reshape(-1, 2)
    have = np.zeros(chm.shape, bool)
    have[ys, xs] = True
    have = ndi.binary_dilation(have, iterations=int(3 / RES))
    extra = []
    for x, y in pts:
        col, row = int((x - XMIN) / RES), int((YMAX - y) / RES)
        if not (0 <= col < chm.shape[1] and 0 <= row < chm.shape[0]) or have[row, col]:
            continue
        h = max(float(chm[row, col]), 4.0 + rng.random() * 3)
        extra.append([x - CX, CY - y, h, 1.2 + h * 0.15, 1.5, 0, 0.35, 0.42, 0.25, rng.random()])
    if extra:
        trees = np.concatenate([trees, np.array(extra, dtype=np.float32)])
    # Masts and chimneys show up in the canopy model as very tall, thin "trees".
    lm = BUILD / "landmarks.json"
    marks = json.loads(lm.read_text()) if lm.exists() else {}
    for m in marks.get("masts", []) + marks.get("chimneys", []):
        trees = trees[np.hypot(trees[:, 0] - (m["x"] - CX), trees[:, 1] - (CY - m["y"])) > 6]
    # No trees on a bridge deck or on its ramps, over a road that runs under something, or over the motorway: the
    # canopy model measures cars, signs and the structure there as trees.
    from shapely import contains_xy, prepare
    from shapely.geometry import LineString
    from shapely.geometry import shape as geo_shape
    from shapely.ops import unary_union

    from era import rail_lines, under_polygons

    bar = [LineString(b["path"]).buffer(b["width"] / 2 + 1.0) for b in marks.get("bridges", [])]
    bar.append(under_polygons().buffer(1.0))
    # Nor on or by the railway: the track is kept clear, or there would be leaves on the line in autumn.
    bar.append(rail_lines().buffer(7.0))
    bar += [geo_shape(f["geometry"]).buffer(1.5) for f in json.load(open(RAW / "bgt_wegdeel.geojson"))["features"]
            if f["properties"].get("functie") == "rijbaan autosnelweg"]
    bar = unary_union(bar)
    prepare(bar)
    onbar = contains_xy(bar, trees[:, 0] + CX, CY - trees[:, 1])
    trees = trees[~onbar]
    print(f"{int(onbar.sum())} trees dropped from bridges, the motorway and the railway")
    trees.tofile(BUILD / "trees.bin")
    print(f"{len(trees)} trees ({len(extra)} from BGT), {int(kind.sum())} evergreen, mean radius {radius.mean():.1f} m")


if __name__ == "__main__":
    main()
