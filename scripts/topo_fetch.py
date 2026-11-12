"""Download the Kadaster's topographic map of a year (Topotijdreis, CC BY 4.0 Kadaster) over the area at its finest
level, and stitch it into one image with its RD position.

    uv run topo_fetch.py 1987

Output: data/raw/topo<year>/mosaic.png and mosaic.json ({"x0", "y0": RD of the top-left corner, "res": m per pixel})
"""

import io
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from config import RAW, XMAX, XMIN, YMAX, YMIN

SERVICE = "https://tiles.arcgis.com/tiles/nSZVuSZjHpEZZbRo/arcgis/rest/services/Historische_tijdreis_{year}/MapServer"
UA = {"User-Agent": "isometric-ede/0.1"}
TILE = 256


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 1987
    service = SERVICE.format(year=year)
    # The tiling scheme: origin and the finest level the service has.
    info = json.loads(get(f"{service}?f=json"))["tileInfo"]
    OX, OY = info["origin"]["x"], info["origin"]["y"]
    LEVEL, RES = info["lods"][-1]["level"], info["lods"][-1]["resolution"]
    d = RAW / f"topo{year}"
    d.mkdir(parents=True, exist_ok=True)

    def tile(rc):
        r, c = rc
        out = d / f"{LEVEL}_{r}_{c}.png"
        if not out.exists():
            out.write_bytes(get(f"{service}/tile/{LEVEL}/{r}/{c}"))
        return rc, Image.open(io.BytesIO(out.read_bytes())).convert("RGB")

    span = TILE * RES
    c0, c1 = math.floor((XMIN - OX) / span), math.floor((XMAX - OX) / span)
    r0, r1 = math.floor((OY - YMAX) / span), math.floor((OY - YMIN) / span)
    img = Image.new("RGB", ((c1 - c0 + 1) * TILE, (r1 - r0 + 1) * TILE), (255, 255, 255))
    with ThreadPoolExecutor(8) as pool:
        for (r, c), t in pool.map(tile, [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]):
            img.paste(t, ((c - c0) * TILE, (r - r0) * TILE))
    img.save(d / "mosaic.png")
    meta = {"x0": OX + c0 * span, "y0": OY - r0 * span, "res": RES}
    (d / "mosaic.json").write_text(json.dumps(meta))
    print(img.size, meta)


if __name__ == "__main__":
    main()
