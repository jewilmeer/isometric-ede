# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "tifffile", "imagecodecs"]
# ///
"""Cut the AHN2 sheets (data/raw/ahn2/r*.tif and i*.tif, 0.5 m, flown 2007-2012) into the same 500 m chunks as the
current AHN in data/raw/ahn, so an era can use them in its place (config.ERAS "ahn2").

The r sheets hold every lidar return, objects included: the surface model. The i sheets hold the ground where the
lidar saw it, with gaps under buildings and trees: the terrain model, as dtm_05m.

Download the sheets that cover the area from GeoTiles (https://geotiles.citg.tudelft.nl, public domain), e.g.
https://basisdata.nl/hwh-ahn/AHN2/DSM_50cm/r39fn1.tif.zip and .../DTM_50cm/i39fn1.tif.zip.

Output: data/raw/ahn2/chunks/{dsm_05m,dtm_05m}_{x0}_{y0}.tif (RD of the lower-left corner)
"""

import numpy as np
import tifffile

from config import AREAS, CHUNK, ERAS, RAW

RES = 0.5
NODATA = np.float32(3.4028235e38)


def sheets(prefix):
    out = []
    for f in sorted((RAW / "ahn2").glob(f"{prefix}*.tif")):
        page = tifffile.TiffFile(f).pages[0]
        x0, y0 = page.tags[33922].value[3:5]  # top-left corner
        out.append((f, x0, y0, page.shape))
    return out


def main():
    out = RAW / "ahn2" / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    px = int(CHUNK / RES)
    # The chunks of the earlier eras' areas (config.AREAS), named after the RD of their lower-left corner.
    cells = sorted({(x0, y0) for e in ERAS for x0 in range(AREAS[e][0], AREAS[e][2], CHUNK) for y0 in range(AREAS[e][1], AREAS[e][3], CHUNK)})
    for prefix, cov in (("r", "dsm_05m"), ("i", "dtm_05m")):
        # A chunk can straddle two sheets: each takes what lies on it.
        parts = [(tifffile.memmap(f), sx, sy, h, w) for f, sx, sy, (h, w) in sheets(prefix)]
        for x0, y0 in cells:
            ytop = y0 + CHUNK
            chunk = np.full((px, px), NODATA, dtype=np.float32)
            filled = np.zeros((px, px), bool)
            for data, sx, sy, h, w in parts:
                c, r = int(round((x0 - sx) / RES)), int(round((sy - ytop) / RES))
                c0, r0, c1, r1 = max(c, 0), max(r, 0), min(c + px, w), min(r + px, h)
                if c0 >= c1 or r0 >= r1:
                    continue
                chunk[r0 - r : r1 - r, c0 - c : c1 - c] = data[r0:r1, c0:c1]
                filled[r0 - r : r1 - r, c0 - c : c1 - c] = True
            assert filled.all(), f"AHN2 does not cover chunk {x0}_{y0}: download the missing sheets"
            chunk[~(np.abs(chunk) < 1e30)] = NODATA
            tifffile.imwrite(out / f"{cov}_{x0}_{y0}.tif", chunk, compression="zlib")
        print(f"{cov}: {len(cells)} chunks")


if __name__ == "__main__":
    main()
