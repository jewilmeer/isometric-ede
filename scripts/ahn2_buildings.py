"""Buildings that stood in an earlier era but have been demolished since, found in that era's AHN (era.AHN).

Where the surface stands more than 3 m above the ground and is smooth (roofs are planes, tree crowns are not), has
straight sides (sand heaps and hedges have not), and no building of today stands, a building stood. Only where the
map changed: near today's buildings that replaced the old ones (era.changed), or where something high then is low
now in today's AHN, so that trees and lorries elsewhere are left alone. Each part of a roof within one 2.5 m band of
height becomes a block up to that height: halls with lower wings come out stepped.

AHN2 has no data over military terrain (the barracks of Ede among them): blind() gives those areas, for another
source such as the topographic map of the time.
"""

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import measure

from config import CHUNK, NX, NY, RAW, XMIN, YMAX
from era import AHN, changed

RES = 0.5
BAND = 2.5


def mosaic(cov, source=AHN):
    px = int(CHUNK / RES)
    out = np.full((NY * px, NX * px), np.nan, dtype=np.float32)
    for ix in range(NX):
        for iy in range(NY):
            a = tifffile.imread(source / f"{cov}_{ix}_{iy}.tif").astype(np.float32)
            a[(a > 1000) | (a < -100)] = np.nan
            row = (NY - 1 - iy) * px
            out[row : row + px, ix * px : ix * px + px] = a
    return out


def mask_of(polys, shape, grow=0):
    img = Image.new("1", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for ring in polys:
        d.polygon([((x - XMIN) / RES, (YMAX - y) / RES) for x, y in ring], fill=1)
    m = np.array(img)
    return ndi.binary_dilation(m, iterations=grow) if grow else m


def ground_filled(dtm):
    bad = np.isnan(dtm)
    return dtm[tuple(ndi.distance_transform_edt(bad, return_distances=False, return_indices=True))]


def blind(min_area=10_000):
    """The areas of at least min_area m² where the era's AHN has no data at all, as RD polygons."""
    from shapely.ops import unary_union

    gap = np.isnan(mosaic("dsm_05m")) & np.isnan(mosaic("dtm_05m"))
    gap = ndi.binary_opening(ndi.binary_closing(gap, iterations=8), iterations=4)
    labels, n = ndi.label(gap)
    out = []
    for sl, k in zip(ndi.find_objects(labels), range(1, n + 1)):
        part = labels[sl] == k
        if part.sum() * RES * RES < min_area:
            continue
        for contour in measure.find_contours(np.pad(part, 1).astype(np.float32), 0.5):
            xy = np.stack([XMIN + (contour[:, 1] + sl[1].start - 1 + 0.5) * RES, YMAX - (contour[:, 0] + sl[0].start - 1 + 0.5) * RES], 1)
            p = Polygon(xy).simplify(2.0).buffer(0)
            if p.area >= min_area:
                out.append(p)
    return unary_union(out) if out else Polygon()


def find(kept_rings):
    """[(ring in RD meters, height), ...] of the buildings demolished since, given today's buildings that remain."""
    dsm = mosaic("dsm_05m")
    dtm = ground_filled(mosaic("dtm_05m"))
    dsm = np.where(np.isnan(dsm), dtm, dsm)
    ndsm = ndi.median_filter(dsm, 3) - dtm
    smooth = np.abs(ndi.laplace(ndi.median_filter(dsm, 3))) < 0.35

    area = changed().buffer(30)
    rings = [np.asarray(p.exterior.coords) for p in getattr(area, "geoms", [area]) if p.geom_type == "Polygon"]
    near = mask_of(rings, dsm.shape)
    # Or where it stood high then and is low now: gone since, whatever replaced it.
    now_dtm = ground_filled(mosaic("dtm_05m", RAW / "ahn"))
    now_dsm = mosaic("dsm_05m", RAW / "ahn")
    now = np.where(np.isnan(now_dsm), now_dtm, now_dsm) - now_dtm
    gone = (ndi.median_filter(dsm, 3) - dtm > 3.0) & (ndi.median_filter(now, 3) < 1.5)
    near |= ndi.binary_dilation(ndi.binary_opening(gone, iterations=2), iterations=4)
    cand = (ndsm > 3.0) & smooth & near & ~mask_of(kept_rings, dsm.shape, grow=2)
    cand = ndi.binary_opening(cand, iterations=2)
    band = np.floor(ndsm / BAND).astype(np.int16)

    out = []
    blobs, n = ndi.label(cand)
    for sl, k in zip(ndi.find_objects(blobs), range(1, n + 1)):
        blob = blobs[sl] == k
        if blob.sum() * RES * RES < 30:
            continue
        for b in np.unique(band[sl][blob]):
            part = ndi.binary_closing(blob & (band[sl] == b), iterations=1)
            part = ndi.binary_fill_holes(part)
            if part.sum() * RES * RES < 12:
                continue
            h = float(np.median(ndsm[sl][part]))
            padded = np.pad(part, 1)
            for contour in measure.find_contours(padded.astype(np.float32), 0.5):
                if len(contour) < 4:
                    continue
                r0, c0 = sl[0].start - 1, sl[1].start - 1
                xy = np.stack([XMIN + (contour[:, 1] + c0 + 0.5) * RES, YMAX - (contour[:, 0] + r0 + 0.5) * RES], 1)
                poly = Polygon(xy).simplify(0.6).buffer(0)
                for p in getattr(poly, "geoms", [poly]):
                    if p.geom_type != "Polygon" or p.area < 12:
                        continue
                    # Buildings have straight sides: what fills less than 70% of its tightest rectangle is a heap of
                    # sand or a hedge. Nearly a rectangle becomes one, with clean corners.
                    rect = p.minimum_rotated_rectangle
                    fill = p.area / rect.area
                    if fill < 0.7:
                        continue
                    if fill > 0.85:
                        p = rect
                    out.append((np.asarray(p.exterior.coords)[:-1], round(h, 1)))
    return out
