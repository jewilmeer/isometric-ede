# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download the raw open data for the area: 3D BAG, BGT, aerial photos and AHN."""

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from config import CHUNK, NX, NY, PLACE, POSTCODES, RAW, XMAX, XMIN, YMAX, YMIN

UA = {"User-Agent": "isometric-ede/0.1"}
RD = "http://www.opengis.net/def/crs/EPSG/0/28992"

BGT_COLLECTIONS = [
    "begroeidterreindeel",
    "onbegroeidterreindeel",
    "wegdeel",
    "ondersteunendwegdeel",
    "waterdeel",
    "ondersteunendwaterdeel",
    "overbruggingsdeel",
    "overigbouwwerk",
    "kunstwerkdeel_vlak",
    "scheiding_vlak",
    "scheiding_lijn",
    "vegetatieobject_punt",
    "vegetatieobject_vlak",
    "vegetatieobject_lijn",
    "pand",
    "spoor",
    "weginrichtingselement_lijn",
    "weginrichtingselement_vlak",
    "straatmeubilair",
    "paal",
    "mast",
]


def get(url, binary=False, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
                return data if binary else json.loads(data)
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1} {url[:100]}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def fetch_3dbag_cell(ix, iy):
    """One CHUNK cell of 3D BAG; a building on a cell border comes back in both cells."""
    x0, y0 = XMIN + ix * CHUNK, YMIN + iy * CHUNK
    url = f"https://api.3dbag.nl/collections/pand/items?bbox={x0},{y0},{x0 + CHUNK},{y0 + CHUNK}&limit=100"
    feats = []
    while url:
        page = get(url)
        for feat in page["features"]:  # an empty page has no metadata
            feat["transform"] = page["metadata"]["transform"]
            feats.append(feat)
        url = next((l["href"] for l in page["links"] if l["rel"] == "next"), None)
    return feats


def fetch_3dbag():
    # The API pages slowly, so ask for every cell at once.
    out = RAW / "3dbag.jsonl"
    if out.exists():
        return
    seen = set()
    with ThreadPoolExecutor(16) as pool, open(out.with_suffix(".tmp"), "w") as f:
        for i, feats in enumerate(pool.map(lambda c: fetch_3dbag_cell(*c), [(i, j) for i in range(NX) for j in range(NY)])):
            for feat in feats:
                if feat["id"] not in seen:
                    seen.add(feat["id"])
                    f.write(json.dumps(feat) + "\n")
            print(f"3dbag cell {i + 1}/{NX * NY}: {len(seen)} buildings", file=sys.stderr)
    out.with_suffix(".tmp").rename(out)


def fetch_bgt(coll):
    out = RAW / f"bgt_{coll}.geojson"
    if out.exists():
        return
    q = urllib.parse.urlencode(
        {"bbox": f"{XMIN},{YMIN},{XMAX},{YMAX}", "bbox-crs": RD, "crs": RD, "limit": 1000, "f": "json"}
    )
    url = f"https://api.pdok.nl/lv/bgt/ogc/v1/collections/{coll}/items?{q}"
    feats = []
    while url:
        page = get(url)
        feats += page["features"]
        url = next((l["href"] for l in page["links"] if l["rel"] == "next"), None)
    # The API also serves retired versions of objects; keep only the current ones.
    current = [f for f in feats if not f["properties"].get("eind_registratie")]
    out.write_text(json.dumps({"type": "FeatureCollection", "features": current}))
    print(f"bgt {coll}: {len(current)} current of {len(feats)}", file=sys.stderr)


def fetch_aerial(ix, iy, chunk=CHUNK, res=0.25):
    d = RAW / "aerial"
    d.mkdir(exist_ok=True)
    out = d / f"{ix}_{iy}.jpg"
    if out.exists():
        return
    x0, y0 = XMIN + ix * chunk, YMIN + iy * chunk
    px = int(chunk / res)
    q = urllib.parse.urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": "Actueel_orthoHR",
            "STYLES": "",
            "CRS": "EPSG:28992",
            "BBOX": f"{x0},{y0},{x0 + chunk},{y0 + chunk}",
            "WIDTH": px,
            "HEIGHT": px,
            "FORMAT": "image/jpeg",
        }
    )
    out.write_bytes(get(f"https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?{q}", binary=True))


def fetch_ahn(cov, ix, iy, chunk=CHUNK):
    # The WCS refuses anything over 4000 px a side, so fetch it in chunks (1000 px at 0.5 m).
    d = RAW / "ahn"
    d.mkdir(exist_ok=True)
    out = d / f"{cov}_{ix}_{iy}.tif"
    if out.exists():
        return
    x0, y0 = XMIN + ix * chunk, YMIN + iy * chunk
    url = (
        "https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={cov}&SUBSET=x({x0},{x0 + chunk})&SUBSET=y({y0},{y0 + chunk})&FORMAT=image/tiff"
    )
    out.write_bytes(get(url, binary=True))


def fetch_addresses():
    """Every BAG address of the town inside the area, for the viewer's search box. The locatieserver pages no further
    than 10000 results, so ask per postcode area."""
    out = RAW / "addresses.json"
    if out.exists():
        return
    pts = []
    for pc in POSTCODES:
        start = 0
        while True:
            q = urllib.parse.urlencode(
                {"q": "*", "fq": [f"woonplaatsnaam:{PLACE}", "type:adres", f"postcode:{pc}*"], "fl": "weergavenaam,centroide_rd", "rows": 100, "start": start},
                doseq=True,
            )
            d = get(f"https://api.pdok.nl/bzk/locatieserver/search/v3_1/free?{q}")["response"]
            for doc in d["docs"]:
                x, y = (float(v) for v in doc["centroide_rd"][6:-1].split())
                if XMIN <= x <= XMAX and YMIN <= y <= YMAX:
                    pts.append((x, y, doc["weergavenaam"]))
            start += 100
            if start >= d["numFound"]:
                break
    out.write_text(json.dumps(pts))


def overpass(out, q, what):
    """Run an Overpass query into out (Overpass is often busy, so try a few servers)."""
    for host in ("overpass-api.de/api/interpreter", "overpass.private.coffee/api/interpreter", "maps.mail.ru/osm/tools/overpass/api/interpreter"):
        try:
            req = urllib.request.Request(f"https://{host}", data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                out.write_bytes(r.read())
            return
        except Exception as e:  # noqa: BLE001
            print(f"overpass {host}: {e}", file=sys.stderr)
    print(f"OpenStreetMap unavailable; {what} are skipped", file=sys.stderr)


def osm_bbox():
    from geo import rd_to_wgs

    lats, lons = zip(*(rd_to_wgs(x, y) for x in (XMIN, XMAX) for y in (YMIN, YMAX)))
    return f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"


def fetch_osm():
    """Pitches, graveyards and points of interest from OpenStreetMap."""
    out = RAW / "osm_landmarks.json"
    if out.exists():
        return
    bbox = osm_bbox()
    q = f"""[out:json][timeout:90];
( nwr["military"]({bbox}); way["leisure"="pitch"]({bbox}); nwr["landuse"="cemetery"]({bbox});
  nwr["amenity"="place_of_worship"]({bbox}); nwr["tourism"]({bbox}); nwr["historic"]({bbox}); );
out geom tags;"""
    overpass(out, q, "pitches and graveyards")


def fetch_routes():
    """Walking routes and the numbered walking-junction network (wandelknooppunten) from OpenStreetMap."""
    out = RAW / "osm_routes.json"
    if out.exists():
        return
    bbox = osm_bbox()
    q = f"""[out:json][timeout:120];
( relation["route"~"^(hiking|foot|walking)$"]({bbox}); node["rwn_ref"]({bbox}); );
out geom;"""
    overpass(out, q, "walking routes")


def fetch_roads():
    """The roads, cycle paths, footpaths and farm tracks that people use, from OpenStreetMap, with node ids to find
    junctions."""
    out = RAW / "osm_roads.json"
    if out.exists():
        return
    bbox = osm_bbox()
    kinds = "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service"
    q = f"""[out:json][timeout:120];
( way["highway"~"^({kinds})$"]({bbox}); way["highway"~"^(cycleway|footway|path|pedestrian|track|steps|bridleway)$"]({bbox}); );
out geom;"""
    overpass(out, q, "roads and paths for traffic")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [fetch_3dbag, fetch_addresses, fetch_osm, fetch_routes, fetch_roads, *(lambda c=c: fetch_bgt(c) for c in BGT_COLLECTIONS)]
    jobs += [lambda c=c, i=i, j=j: fetch_ahn(c, i, j) for c in ("dsm_05m", "dtm_05m") for i in range(NX) for j in range(NY)]
    jobs += [lambda i=i, j=j: fetch_aerial(i, j) for i in range(NX) for j in range(NY)]
    with ThreadPoolExecutor(8) as pool:
        for fut in [pool.submit(j) for j in jobs]:
            fut.result()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
