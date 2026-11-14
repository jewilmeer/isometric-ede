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

from config import AREAS, CHUNK, ERAS, NOW_AREA, PLACE, POSTCODES, RAW

# Everything that is downloaded by area is downloaded for today's, the largest (config.AREAS); the earlier eras use a part.
XMIN, YMIN, XMAX, YMAX = NOW_AREA
NX, NY = int((XMAX - XMIN) / CHUNK), int((YMAX - YMIN) / CHUNK)

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


# What the earlier eras need (the OpenStreetMap of the past): the union of their areas.
PAST = (min(AREAS[e][0] for e in ERAS), min(AREAS[e][1] for e in ERAS), max(AREAS[e][2] for e in ERAS), max(AREAS[e][3] for e in ERAS))


def covers(out, area=NOW_AREA):
    """Whether out was downloaded for an area that includes today's: the area is kept in out.bbox.json next to it
    (a file from before that was kept belongs to the old area, which is not enough)."""
    side = out.with_name(out.name + ".bbox.json")
    if not out.exists() or not side.exists():
        return False
    x0, y0, x1, y1 = json.loads(side.read_text())
    return x0 <= area[0] and y0 <= area[1] and x1 >= area[2] and y1 >= area[3]


def done(out, area=None):
    """Record that out was downloaded for today's area (or another)."""
    out.with_name(out.name + ".bbox.json").write_text(json.dumps(list(area or WANT.get(out) or NOW_AREA)))


WANT = {}  # file -> the area it is being downloaded for, when that is not today's


def migrate():
    """Chunks used to be named by their index from the old area's corner (172750, 447000); now by the RD of their
    lower-left corner, so that every era's area finds them."""
    import re

    for d in (RAW / "aerial", RAW / "ahn", RAW / "ahn2" / "chunks"):
        for f in d.glob("*.*") if d.exists() else []:
            m = re.fullmatch(r"(.*?)(\d{1,2})_(\d{1,2})(\.\w+)", f.name)
            if m:
                f.rename(d / f"{m[1]}{172750 + int(m[2]) * CHUNK}_{447000 + int(m[3]) * CHUNK}{m[4]}")


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
    if covers(out):
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
    done(out)


def fetch_bgt(coll):
    out = RAW / f"bgt_{coll}.geojson"
    if covers(out):
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
    done(out)
    print(f"bgt {coll}: {len(current)} current of {len(feats)}", file=sys.stderr)


def fetch_aerial(ix, iy, chunk=CHUNK, res=0.25):
    d = RAW / "aerial"
    d.mkdir(exist_ok=True)
    x0, y0 = XMIN + ix * chunk, YMIN + iy * chunk
    out = d / f"{x0}_{y0}.jpg"
    if out.exists():
        return
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
    x0, y0 = XMIN + ix * chunk, YMIN + iy * chunk
    out = d / f"{cov}_{x0}_{y0}.tif"
    if out.exists():
        return
    url = (
        "https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={cov}&SUBSET=x({x0},{x0 + chunk})&SUBSET=y({y0},{y0 + chunk})&FORMAT=image/tiff"
    )
    out.write_bytes(get(url, binary=True))


def fetch_addresses():
    """Every BAG address of the town inside the area, for the viewer's search box. The locatieserver pages no further
    than 10000 results, so ask per postcode area."""
    out = RAW / "addresses.json"
    if covers(out):
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
    done(out)


def overpass(out, q, what, timeout=180, hosts=None):
    """Run an Overpass query into out (Overpass is often busy, so try a few servers)."""
    for host in hosts or ("overpass-api.de/api/interpreter", "overpass.private.coffee/api/interpreter", "maps.mail.ru/osm/tools/overpass/api/interpreter"):
        try:
            req = urllib.request.Request(f"https://{host}", data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out.write_bytes(r.read())
            done(out)
            return
        except Exception as e:  # noqa: BLE001
            print(f"overpass {host}: {e}", file=sys.stderr)
    print(f"OpenStreetMap unavailable; {what} are skipped", file=sys.stderr)


def osm_bbox(area=NOW_AREA):
    from geo import rd_to_wgs

    lats, lons = zip(*(rd_to_wgs(x, y) for x in (area[0], area[2]) for y in (area[1], area[3])))
    return f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"


def fetch_osm():
    """Pitches, graveyards and points of interest from OpenStreetMap."""
    out = RAW / "osm_landmarks.json"
    if covers(out):
        return
    bbox = osm_bbox()
    q = f"""[out:json][timeout:90];
( nwr["military"]({bbox}); way["leisure"~"^(pitch|track)$"]({bbox}); nwr["landuse"="cemetery"]({bbox});
  nwr["amenity"="place_of_worship"]({bbox}); nwr["tourism"]({bbox}); nwr["historic"]({bbox}); );
out geom tags;"""
    overpass(out, q, "pitches and graveyards")


def fetch_routes():
    """Walking routes and the numbered walking-junction network (wandelknooppunten) from OpenStreetMap."""
    out = RAW / "osm_routes.json"
    if covers(out):
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
    if covers(out):
        return
    bbox = osm_bbox()
    kinds = "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service"
    q = f"""[out:json][timeout:120];
( way["highway"~"^({kinds})$"]({bbox}); way["highway"~"^(cycleway|footway|path|pedestrian|track|steps|bridleway)$"]({bbox}); );
out geom;"""
    overpass(out, q, "roads and paths for traffic")


def fetch_rail():
    """The railway tracks, platforms and stations, for the trains."""
    out = RAW / "osm_rail.json"
    if covers(out):
        return
    bbox = osm_bbox()
    q = f"""[out:json][timeout:120];
( way["railway"="rail"]({bbox}); way["railway"="platform"]({bbox}); node["railway"~"^(station|halt)$"]({bbox}); );
out geom tags;"""
    overpass(out, q, "railway tracks")


# OpenStreetMap as it was: the oldest state Overpass keeps (its history starts in September 2012), for the earlier
# eras (config.ERAS "osm"), when the station of Ede-Wageningen still stood west of the Klinkerbergerweg.
ATTIC = "2012-09-13T00:00:00Z"
# Not every Overpass server keeps the history; one that doesn't would answer with today's map.
HISTORY = ("overpass-api.de/api/interpreter", "overpass.private.coffee/api/interpreter")


def fetch_attic_buildings():
    """The buildings as OpenStreetMap had them at ATTIC, into osm_buildings-2012.json: what has been demolished since
    and that the height data of that time misses (scripts/build_buildings.py add_demolished)."""
    out = RAW / f"osm_buildings-{ATTIC[:4]}.json"
    WANT[out] = PAST
    if covers(out, PAST):
        return
    q = f"""[out:json][timeout:300][date:"{ATTIC}"];
( way["building"]({osm_bbox(PAST)}); );
out geom tags;"""
    for _ in range(3):  # history queries often time out; try again
        overpass(out, q, "buildings of the past", timeout=600, hosts=HISTORY)
        if out.exists():
            break


def fetch_attic():
    """The roads and the railway as OpenStreetMap had them at ATTIC, into osm_roads-2012.json and osm_rail-2012.json.
    History queries are slow, so the roads come in four parts."""
    tag = ATTIC[:4]
    rail = RAW / f"osm_rail-{tag}.json"
    WANT[rail] = PAST
    if not covers(rail, PAST):
        q = f"""[out:json][timeout:300][date:"{ATTIC}"];
( way["railway"="rail"]({osm_bbox(PAST)}); way["railway"="platform"]({osm_bbox(PAST)}); node["railway"~"^(station|halt)$"]({osm_bbox(PAST)}); );
out geom tags;"""
        overpass(rail, q, "railway tracks of the past", timeout=400, hosts=HISTORY)
    roads = RAW / f"osm_roads-{tag}.json"
    WANT[roads] = PAST
    if covers(roads, PAST):
        return
    s, w, n, e = map(float, osm_bbox(PAST).split(","))
    kinds = "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service"
    seen, ways = set(), []
    for lat0, lat1 in ((s, (s + n) / 2), ((s + n) / 2, n)):
        for lon0, lon1 in ((w, (w + e) / 2), ((w + e) / 2, e)):
            b = f"{lat0},{lon0},{lat1},{lon1}"
            q = f"""[out:json][timeout:300][date:"{ATTIC}"];
( way["highway"~"^({kinds})$"]({b}); way["highway"~"^(cycleway|footway|path|pedestrian|track|steps|bridleway)$"]({b}); );
out geom;"""
            part = roads.with_suffix(".part")
            for _ in range(3):  # history queries often time out; try again
                overpass(part, q, "roads of the past", timeout=600, hosts=HISTORY)
                if part.exists():
                    break
            for el in json.loads(part.read_text())["elements"]:
                if el["id"] not in seen:
                    seen.add(el["id"])
                    ways.append(el)
            part.unlink()
            part.with_name(part.name + ".bbox.json").unlink(missing_ok=True)
    roads.write_text(json.dumps({"elements": ways}))
    done(roads, PAST)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    migrate()
    jobs = [fetch_3dbag, fetch_addresses, fetch_osm, fetch_routes, fetch_roads, fetch_rail, *(lambda c=c: fetch_bgt(c) for c in BGT_COLLECTIONS)]
    jobs += [lambda c=c, i=i, j=j: fetch_ahn(c, i, j) for c in ("dsm_05m", "dtm_05m") for i in range(NX) for j in range(NY)]
    jobs += [lambda i=i, j=j: fetch_aerial(i, j) for i in range(NX) for j in range(NY)]
    with ThreadPoolExecutor(8) as pool:
        for fut in [pool.submit(j) for j in jobs]:
            fut.result()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
