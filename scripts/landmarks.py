# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Named places of Ede around the Enka-terrein: style overrides and extra geometry for buildings, plus the viewer's
place list.

Sources: OpenStreetMap (names and positions) and AHN. The stories still have to be written; places without a `text`
show only their name and kind.

Output: data/build/landmarks.json
"""

import json

import numpy as np

from config import BUILD, RAW
from geo import wgs_to_rd

# Buildings found by an address or point; `pand` is resolved to the 3D BAG building under that point.
BUILDINGS = [
    # The Enka-terrein: the rayon factory of the Eerste Nederlandsche Kunstzijdefabriek, now a new neighbourhood.
    {"name": "Enka-schoorsteen", "at": (175084, 448248), "kind": "Schoorsteen · 1922"},
    {"name": "De Reehorst", "at": (174497, 448446), "kind": "Congrescentrum en hotel"},
    # The kazerneterrein, from OpenStreetMap (all tagged as municipal heritage).
    {"name": "Johan Willem Friso kazerne", "at": (174746, 448948), "kind": "Kazerne"},
    {"name": "Mauritskazerne", "at": (174712, 449149), "kind": "Kazerne"},
    {"name": "Arthur Kool kazerne", "at": (174763, 449603), "kind": "Kazerne"},
    {"name": "P.L. Bergansiuskazerne", "at": (174821, 449534), "kind": "Kazerne"},
    {"name": "Van Essenkazerne", "at": (174952, 449617), "kind": "Kazerne"},
    {"name": "Elias Beeckman kazerne", "at": (175356, 449896), "kind": "Kazerne"},
    # Churches and the old centre.
    {"name": "Oude Kerk", "at": (174507, 450892), "kind": "Kerk"},
    {"name": "Sint Antonius van Padua", "at": (174422, 449193), "kind": "Kerk"},
    {"name": "Historisch Museum Ede", "at": (174213, 450948), "kind": "Museum"},
    {"name": "Kasteel Hoekelum", "at": (174878, 447502), "kind": "Kasteel"},
    {"name": "50PK", "at": (174984, 449488), "kind": "Galerie"},
]

# Points without a building of their own.
POINTS = [
    # Not in 3D BAG; AHN sees a spike of about 50 m here.
    {"name": "Enka Kleine Schoorsteen", "at": (175119, 448433), "kind": "Schoorsteen"},
    {"name": "De padden van het Enka-terrein", "at": (175126, 448100), "kind": "Informatiebord"},
    {"name": "Monument voor de bombardementsslachtoffers", "at": (173869, 448765), "kind": "Oorlogsmonument"},
    {"name": "Oorlogsmonument", "at": (174956, 450248), "kind": "Oorlogsmonument"},
    {"name": "Mausoleum verzetstrijders", "at": (174999, 450636), "kind": "Monument"},
    {"name": "Anna Maria Moens", "at": (174894, 450607), "kind": "Monument"},
    {"name": "Wilmina Lantaarn", "at": (174318, 450446), "kind": "Monument"},
    {"name": "Beeld: De Danser", "at": (174542, 448576), "kind": "Beeld"},
    {"name": "Beeld: Het Koningspaar", "at": (172890, 448367), "kind": "Kunstwerk"},
    {"name": "Beeld: Leeuw", "at": (175390, 449944), "kind": "Beeld"},
    {"name": "Beeld: De Grote Vlucht", "at": (174858, 450919), "kind": "Beeld"},
    {"name": "Beeld: Spelende beren", "at": (174485, 451001), "kind": "Beeld"},
    {"name": "Kinderboerderij De Proosdij", "at": (173196, 450270), "kind": "Kinderboerderij"},
]

# Series in the viewer's panel: a walk past these places, starting with the one nearest the Enka chimney.
CATEGORIES = [
    ("enka", "Enka-terrein", ["Enka-schoorsteen", "Enka Kleine Schoorsteen", "De padden van het Enka-terrein", "De Reehorst"]),
    ("kazernes", "Kazernes", [
        "Johan Willem Friso kazerne", "Mauritskazerne", "Arthur Kool kazerne", "P.L. Bergansiuskazerne", "Van Essenkazerne",
        "Elias Beeckman kazerne",
    ]),
    ("historisch", "Historische plekken", ["Oude Kerk", "Sint Antonius van Padua", "Historisch Museum Ede", "Kasteel Hoekelum"]),
    ("oorlog", "Oorlog en herdenking", [
        "Monument voor de bombardementsslachtoffers", "Oorlogsmonument", "Mausoleum verzetstrijders",
    ]),
    ("kunst", "Kunst", [
        "Beeld: De Danser", "Beeld: Het Koningspaar", "Beeld: Leeuw", "Beeld: De Grote Vlucht", "Beeld: Spelende beren",
        "Anna Maria Moens", "Wilmina Lantaarn", "50PK",
    ]),
]

# What the viewer shows when a place is clicked; the rest of an entry is for the render.
INFO = ("note", "kind", "text", "address", "web", "more", "photo", "stop")

# Steel lattice masts that 3D BAG leaves out: none in Ede yet.
MASTS = []


def point_in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


ADDRESSES = None


def address_xy(name):
    """RD position of a BAG address such as "Geldersestraat 3" (data/raw/addresses.json, from scripts/fetch.py)."""
    global ADDRESSES
    if ADDRESSES is None:
        ADDRESSES = {label.split(", ")[0]: (x, y) for x, y, label in json.loads((RAW / "addresses.json").read_text())}
    return ADDRESSES[name]


def building_at(rings, x, y):
    for rid, ring, a in rings:
        if point_in_ring(x, y, ring):
            return rid, a
    return None, None


def main():
    feats = [json.loads(line) for line in open(RAW / "3dbag.jsonl")]
    rings = []
    for f in feats:
        t = f["transform"]
        V = np.array(f["vertices"], dtype=np.float64) * t["scale"] + t["translate"]
        co = f["CityObjects"][f["id"]]
        rings.append((f["id"], V[co["geometry"][0]["boundaries"][0][0]][:, :2], co["attributes"]))

    out = {"buildings": {}, "places": [], "masts": [], "hotspots": [], "categories": []}
    for b in BUILDINGS:
        if "addrs" in b:
            pts = [address_xy(a) for a in b["addrs"]]
            pids = [building_at(rings, *pt)[0] for pt in pts]
            assert all(pids), f"{b['name']}: no building at {b['addrs']}"
            x, y = np.mean(pts, axis=0)
            out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": 4.0})
            out["hotspots"].append({"name": b["name"], "pids": pids, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
            print(f"{b['name']}: {', '.join(pids)}")
            continue
        x, y = b["at"] if "at" in b else wgs_to_rd(*b["latlon"])
        pid, attrs = None, None
        for rid, ring, a in rings:
            if point_in_ring(x, y, ring):
                pid, attrs = rid, a
                break
        if pid is None:
            d = [(np.hypot(*(ring.mean(0) - (x, y))), rid, a) for rid, ring, a in rings]
            dist, pid, attrs = min(d, key=lambda t: t[0])
            if dist > 20:
                pid = None
        height = ((attrs or {}).get("b3_h_dak_max") or 0) - ((attrs or {}).get("b3_h_maaiveld") or 0)
        if "tower" in b:
            height = b["tower"]["top"]
        elif "turret" in b:
            height = b["turret"]["top"]
        out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": round(height, 1)})
        if pid and "style" in b:
            out["buildings"][pid] = {k: v for k, v in b.items() if k not in ("at", "latlon", *INFO)}
        if pid:
            out["hotspots"].append({"name": b["name"], "pid": pid, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
        print(f"{b['name']}: {pid or 'no building'}")
    for p in POINTS:
        x, y = p["at"] if "at" in p else address_xy(p["addr"]) if "addr" in p else wgs_to_rd(*p["latlon"])
        out["places"].append({"name": p["name"], "note": p.get("note"), "x": x, "y": y, "h": 2.0})
        out["hotspots"].append({"name": p["name"], "x": x, "y": y, **{k: p[k] for k in INFO if k in p}})
    for m in MASTS:
        x, y = wgs_to_rd(*m["latlon"])
        out["masts"].append({**{k: v for k, v in m.items() if k != "latlon"}, "x": x, "y": y})
    names = {h["name"] for h in out["hotspots"]}
    for key, label, members in CATEGORIES:
        missing = [n for n in members if n not in names]
        assert not missing, f"{label}: no hotspot for {missing}"
        out["categories"].append({"key": key, "label": label, "names": members})
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "landmarks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
