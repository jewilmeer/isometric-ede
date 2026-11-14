"""Aerial photo mosaic at GROUND_RES, with row 0 at the north edge."""

import numpy as np
from PIL import Image

from config import BUILD, CHUNK, chunk_name, GROUND_RES, NX, NY, RAW, XMIN, XMAX, YMAX, YMIN


def load_aerial():
    cache = BUILD / f"aerial-{XMIN}_{YMIN}_{XMAX}_{YMAX}.npy"  # the area is in the name: a cache of another area is no use
    if cache.exists():
        return np.load(cache)
    px = int(CHUNK / GROUND_RES)
    out = np.zeros((NY * px, NX * px, 3), dtype=np.uint8)
    for ix in range(NX):
        for iy in range(NY):
            img = np.array(Image.open(RAW / "aerial" / f"{chunk_name(ix, iy)}.jpg").convert("RGB"))
            row = (NY - 1 - iy) * px
            out[row : row + px, ix * px : ix * px + px] = img
    BUILD.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def world_to_px(xy):
    """RD meters (N,2) to fractional pixel coordinates (col, row) in the mosaic."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([(xy[:, 0] - XMIN) / GROUND_RES, (YMAX - xy[:, 1]) / GROUND_RES], axis=-1)
