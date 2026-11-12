# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Named places of Ede around the Enka-terrein: style overrides and extra geometry for buildings, plus the viewer's
place list.

Sources: OpenStreetMap (names and positions), AHN (heights), BOEi (boei.nl, the foundation that restored the Enka
monuments). Places without a `text` show only their name and kind.

Output: data/build/landmarks.json
"""

import json

import numpy as np

from config import BUILD, RAW
from geo import wgs_to_rd

# Buildings found by an address or point; `pand` is resolved to the 3D BAG building under that point.
BUILDINGS = [
    # The Enka-terrein: the rayon factory of the Eerste Nederlandsche Kunstzijdefabriek, now a new neighbourhood.
    {
        "name": "Enka-schoorsteen",
        "at": (175084, 448248),
        "kind": "Schoorsteen",
        "text": "Het Enka-terrein heeft twee schoorstenen: de ketelschoorsteen uit 1920 en de spinnerijschoorsteen uit 1927, door de arbeiders de dunne en de dikke genoemd. De spinnerijschoorsteen is de hoogste gemetselde schoorsteen van Nederland; hij voerde de spingassen af, door buizen over de daken en langs de gevels van de spinnerij en de bleekgebouwen. In mei 1940 zouden de schoorstenen worden gesloopt, omdat ze voor het Duitse leger een te duidelijk baken waren, maar de snelle inval maakte daar een eind aan. BOEi restaureerde ze; de spinnerijschoorsteen in 2020.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    # The old canteen, a rijksmonument: a school below, after-school care and a nursery on the top floor.
    {
        "name": "Kantinegebouw",
        "at": (175018, 448454),
        "kind": "Rijksmonument · brede school",
        "text": "Het oude kantinegebouw van de Enka. BOEi restaureerde het eerst als casco en maakte er daarna een bijna energieneutrale brede school van: in de onderste twee lagen basisschool De Ontdekking, met het schoolplein aan de westkant, en bovenin buitenschoolse opvang en kinderopvang, met een terras. Het project won de Erfgoed Duurzaamheidsprijs 2024.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    {"name": "Gymlokaal Enka", "at": (174977, 448483), "kind": "Gymzaal"},
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
    # Not in 3D BAG; drawn from CHIMNEYS below.
    {
        "name": "Enka Kleine Schoorsteen",
        "at": (175119.4, 448434.5),
        "kind": "Schoorsteen · 50 m",
        "text": "De tweede schoorsteen van het Enka-terrein, zo'n 50 meter hoog. De fabriek zette haar schoorstenen in haar logo.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    {"name": "Langzaamverkeerbrug", "at": (175295, 448655), "kind": "Fiets- en voetgangersbrug", "text": "De brug voor fietsers en voetgangers over het spoor, tussen de Enka-wijk en de Sysselt."},
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
    ("enka", "Enka-terrein", [
        "Enka-schoorsteen", "Enka Kleine Schoorsteen", "Kantinegebouw", "Gymlokaal Enka", "De padden van het Enka-terrein",
        "Langzaamverkeerbrug", "De Reehorst",
    ]),
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

# Buildings that only need another colour than the renderer would pick, by address.
STYLED = [{"addrs": ["Krimperij 3"], "style": "house", "wall": (236, 234, 226), "frame": (60, 62, 66)}]

# Brick chimneys that 3D BAG leaves out, measured in AHN: height above the ground, radius at the foot and the top.
CHIMNEYS = [{"at": (175119.4, 448434.5), "height": 49.6, "r0": 2.6, "r1": 1.9, "wall": (150, 84, 62)}]

# The bridge for cyclists and walkers over the railway, along the Sysseltsepad in OpenStreetMap: from the ground on
# the Sysselt side up to the span over the tracks, and down the hairpin ramp on the Enka side. `span` is the stretch
# (indices into `path`) at full height; the ramps slope from there to the ground at both ends. The render is flat,
# so the deck clears the tracks by its full height.
BRIDGES = [
    {
        "name": "Langzaamverkeerbrug",
        "path": [
            (175167.7, 448771.2), (175185.4, 448766.8), (175199.1, 448762.3), (175217.1, 448753.8), (175233.3, 448744.4),
            (175243.8, 448736.2), (175251.7, 448730.2), (175276.2, 448706.1), (175282.0, 448698.2), (175285.6, 448690.2),
            (175289.0, 448679.4), (175301.7, 448630.6), (175304.3, 448626.7), (175308.0, 448624.0), (175312.7, 448622.6),
            (175319.9, 448620.4), (175382.9, 448604.0), (175385.4, 448602.6), (175387.2, 448600.4), (175387.6, 448598.2),
            (175387.5, 448595.3), (175386.9, 448591.9), (175385.0, 448589.3), (175382.1, 448587.5), (175376.4, 448587.1),
            (175315.5, 448598.7), (175313.1, 448599.3), (175279.9, 448607.2), (175223.0, 448625.2),
        ],
        "span": (10, 11),
        "height": 7.5,
        "width": 4.6,
        "deck": (212, 118, 60),  # the orange of the cycle path
        "wall": (200, 198, 190),
        "rail": (70, 74, 80),
    },
]

# Ground that changed after the BGT and the aerial photo: the strip between the railway and the Enka neighbourhood is
# a park now. Inside `area`, the BGT's sand, yards and the old asphalt road become grass, and the cycle paths (from the
# BGT and OpenStreetMap) the orange of the new cycle path.
GROUND = [
    {
        "name": "Park langs het spoor",
        "area": [
            (175007.5, 448702.5), (175407.5, 448602.5), (175392.5, 448582.0), (175320.0, 448582.0), (175312.0, 448548.0),
            (175236.0, 448548.0), (175234.0, 448598.0), (175050.0, 448605.0), (175002.0, 448632.0),
        ],
        "cycle": (212, 118, 60),
    },
]

# Where children play, in RD meters: the schoolyard west of the old canteen and the terrace of the after-school care
# on its top floor (3D BAG has that roof at 7.3 m).
PLAY = [
    {"name": "Schoolplein", "area": [(174992.5, 448489.0), (175014.0, 448482.5), (175008.0, 448462.0), (174995.0, 448465.0), (174988.0, 448477.0)], "h": 0.0, "kids": 16},
    {"name": "Terras van de BSO", "area": [(175012.5, 448480.6), (175016.4, 448479.6), (175005.0, 448432.4), (175001.4, 448433.3)], "h": 7.3, "kids": 6},
]


def bridge_heights(b):
    """The deck height at every point of a bridge's path: full over the span, sloping evenly to the ground at both ends."""
    p = np.asarray(b["path"], dtype=np.float64)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s0, s1, h = cum[b["span"][0]], cum[b["span"][1]], b["height"]
    return [round(float(v), 2) for v in np.where(cum < s0, h * cum / s0, np.where(cum > s1, h * (cum[-1] - cum) / (cum[-1] - s1), h))]


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
    for b in STYLED:
        pts = [address_xy(a) for a in b["addrs"]]
        for pid in {building_at(rings, *pt)[0] for pt in pts}:
            assert pid, f"no building at {b['addrs']}"
            out["buildings"][pid] = {"name": ", ".join(b["addrs"]), **{k: v for k, v in b.items() if k != "addrs"}}
            print(f"{', '.join(b['addrs'])}: restyled {pid}")
    out["chimneys"] = [{**{k: v for k, v in c.items() if k != "at"}, "x": c["at"][0], "y": c["at"][1]} for c in CHIMNEYS]
    out["bridges"], out["ground"], out["play"] = [{**b, "heights": bridge_heights(b)} for b in BRIDGES], GROUND, PLAY
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
