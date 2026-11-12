# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "shapely"]
# ///
"""Named places of Ede around the Enka-terrein: style overrides and extra geometry for buildings, plus the viewer's
place list.

Sources: OpenStreetMap (names and positions), AHN (heights), BOEi (boei.nl, the foundation that restored the Enka
monuments). Places without a `text` show only their name and kind.

Output: data/build/landmarks.json
"""

import json

import numpy as np

from config import BUILD, RAW, in_era
from geo import wgs_to_rd

# Buildings found by an address or point; `pand` is resolved to the 3D BAG building under that point.
BUILDINGS = [
    # The Enka-terrein: the rayon factory of the Eerste Nederlandsche Kunstzijdefabriek, now a new neighbourhood.
    {
        "name": "Spinnerijschoorsteen",
        "at": (175084, 448248),
        "kind": "Schoorsteen · 1927",
        "text": "De dikke, zoals de arbeiders hem noemden: de schoorsteen van de spinnerij, de hoogste gemetselde schoorsteen van Nederland. Hij voerde de spingassen af, door buizen over de daken en langs de gevels van de spinnerij en de bleekgebouwen; zolang er geen afzuiging was, kregen de spinners er pijnlijke ogen van. In mei 1940 zouden de schoorstenen worden gesloopt, omdat ze voor het Duitse leger een te duidelijk baken waren, maar de snelle inval maakte daar een eind aan. BOEi restaureerde hem in 2020; nu ligt hij midden op het Schoorsteenplein, en er broedt een torenvalk in.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    # The old canteen, a rijksmonument: a school below, after-school care and a nursery on the top floor.
    {
        "name": "Kantinegebouw",
        "at": (175018, 448454),
        "kind": "Rijksmonument · brede school",
        "style": "canteen",
        "wall": (196, 194, 186),
        "roof": (70, 72, 76),
        "frame": (58, 128, 104),
        # The upper floor runs on past the ground floor at both ends, floating on columns over the open space below
        # (AM, Op Enka, p. 33, and photos): in 3D BAG those are the roof parts of about 11 m.
        "stilts": {"lift": (10.5, 11.5), "under": 6.8},
        "text": "J.C. Hartogs, die de fabriek in 1919 bouwde, liet het kantinegebouw na de Tweede Wereldoorlog neerzetten, als een vrolijke plek die zijn medewerkers de oorlog moest doen vergeten: licht, ruim en met art-decodetails, de bovenverdiepingen zwevend op kolommen. BOEi restaureerde het eerst als casco en maakte er daarna een bijna energieneutrale brede school van: in de onderste twee lagen basisschool De Ontdekking, met het schoolplein aan de westkant, en bovenin buitenschoolse opvang en kinderopvang, met een terras. Het project won de Erfgoed Duurzaamheidsprijs 2024.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    {"name": "Gymlokaal Enka", "since": 2021, "at": (174977, 448483), "kind": "Gymzaal"},
    {
        "name": "Westhal",
        "at": (174912, 448571),
        "kind": "Rijksmonument · 1922",
        # One building in 3D BAG with the Poortgebouw: dark brick, red window frames.
        "style": "factory",
        "wall": (118, 66, 52),
        "roof": (92, 94, 98),
        "frame": (170, 34, 32),
        "text": "Een fabriekshal van zo'n 10.000 m², sinds 2014 van BOEi. Lang was het een evenementenlocatie, met autoshows, rommelmarkten en opnames voor televisie, tot dat om veiligheidsredenen stopte. Daarna vestigde Accell er De Fietser, een fietsbelevingscentrum. Die wilde ook op zondag open, en de gemeente paste daarvoor de winkelverordening aan.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
    # The main building of the factory, with the arch and the letters ENKA: in 3D BAG part of the Westhal's building.
    {
        "name": "Poortgebouw",
        "at": (174985, 448604),
        "kind": "Rijksmonument · 1922",
        "text": "Het hoofdgebouw van de fabriek, met de grote boog waar je het terrein op ging en bovenin de letters ENKA. Toen de bouw van de wijk begon, had de Grondbank er haar kantoor, en regelde de hoofduitvoerder van daaruit ook de evenementen in de leegstaande Westhal. Bouwbedrijf Kreeft restaureerde het voor BOEi.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
    {
        "name": "Poort naar de Krimperij",
        "at": (175210, 448526),
        "kind": "Fabrieksmuur · ca. 1922",
        "text": "De boog in de oude noordmuur van de fabriek, nu de doorgang van het park naar de Krimperij. De bijna honderd jaar oude muur is geen monument, maar was nog goed genoeg om te behouden: achter de gevel met de oude raampartijen staan nu nieuwe woningen, de muurwoningen van het Carré.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
    # The new Ketelhuis next to the boiler chimney, after the boiler house that stood there.
    {
        "name": "Het Ketelhuis (Stoer)", "since": 2020,
        "at": (175089.9, 448445.8),
        # Near-black brick with yellow panels beside the windows and grey shutters.
        "style": "stoer",
        "wall": (46, 46, 50),
        "roof": (70, 72, 76),
        "frame": (232, 186, 40),
        "kind": "Woongebouw · 2020",
        "text": "Een nieuw gebouw, geïnspireerd op het ketelhuis dat hier naast de dunne schoorsteen stond, met een groot grasveld eromheen en een speeltuin met schommels en een hindernisbaan. Op de begane grond woont Stoer: ouders van jongvolwassenen met een verstandelijke beperking richtten de stichting op, zodat hun kinderen ieder een eigen appartement hebben, met zorg en gezamenlijke ruimtes. Ze mochten zelf een kruisje op de kaart van de wijk zetten. Boven wonen huurders van Woonstede.",
        "more": "https://www.am.nl/wp-content/uploads/2024/03/OP-Enka-waar-elke-steen-een-verhaal-vertelt.pdf",
    },
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
        "name": "Ketelschoorsteen",
        "at": (175119.4, 448434.5),
        "kind": "Schoorsteen · 1920",
        "text": "De dunne: de schoorsteen van het ketelhuis, zo'n 50 meter hoog, de oudste van de twee. Het ketelhuis zelf is gesloopt; het nieuwe gebouw ernaast is erop geïnspireerd. De fabriek zette haar schoorstenen in haar logo.",
        "more": "https://www.boei.nl/projecten/schoorstenen-enka-ede",
    },
    {"name": "Langzaamverkeerbrug", "since": 2022, "at": (175295, 448655), "kind": "Fiets- en voetgangersbrug", "text": "De brug voor fietsers en voetgangers over het spoor, tussen de Enka-wijk en de Sysselt."},
    {"name": "De padden van het Enka-terrein", "since": 2019, "at": (175126, 448100), "kind": "Informatiebord"},
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
        "Spinnerijschoorsteen", "Ketelschoorsteen", "Het Ketelhuis (Stoer)", "Kantinegebouw", "Gymlokaal Enka", "Poortgebouw", "Poort naar de Krimperij",
        "Westhal", "De padden van het Enka-terrein", "Langzaamverkeerbrug", "De Reehorst",
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

# Station roofs from the BGT "overkapping" that contains `at`: built after the AHN flight, so measured by eye.
# The roof of Ede-Wageningen is folded grey metal, a field of triangles with triangular skylights; the station hall at
# its south-east lobe has a flat green roof, and beside it stands a tall slender tower.
STATIONS = [{"name": "Ede-Wageningen", "since": 2023, "at": (174691, 448810), "height": 9.0, "fold": 1.4, "cell": 6.0,
             "grid": 18.0, "skylights": 0.14, "roof": (112, 116, 122), "glass": (176, 200, 214), "edge": (150, 154, 158),
             "columns": (230, 230, 226), "hall": (174680, 448720, 174770, 448792), "hall_height": 7.0,
             "hall_wall": (70, 74, 80), "hall_roof": (104, 128, 76),
             "tower": {"at": (174681, 448752), "h": 28.0, "w": 4.0}, "tower_colour": (206, 210, 212)}]

# Buildings that only need another colour than the renderer would pick, by address.
STYLED = [{"since": 2020, "addrs": ["Krimperij 3"], "style": "house", "wall": (236, 234, 226), "frame": (60, 62, 66)}]

# Brick chimneys that 3D BAG leaves out, measured in AHN: height above the ground, radius at the foot and the top.
CHIMNEYS = [{"at": (175119.4, 448434.5), "height": 49.6, "r0": 2.6, "r1": 1.9, "wall": (150, 84, 62)}]

# The bridge for cyclists and walkers over the railway, along the Sysseltsepad in OpenStreetMap: from the ground on
# the Sysselt side up to the span over the tracks, and down the hairpin ramp on the Enka side. `span` is the stretch
# (indices into `path`) at full height; the ramps slope from there to the ground at both ends. The render is flat,
# so the deck clears the tracks by its full height.
BRIDGES = [
    {
        "name": "Langzaamverkeerbrug",
        "since": 2022,
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
        "since": 2023,
        "area": [
            (175007.5, 448702.5), (175407.5, 448602.5), (175392.5, 448582.0), (175320.0, 448582.0), (175312.0, 448548.0),
            (175236.0, 448548.0), (175234.0, 448598.0), (175050.0, 448605.0), (175002.0, 448632.0),
        ],
        "cycle": (212, 118, 60),
    },
]

def circle(c, r, n=16):
    return [(round(c[0] + r * np.cos(2 * np.pi * i / n), 2), round(c[1] + r * np.sin(2 * np.pi * i / n), 2)) for i in range(n)]


# Things to play on (scripts/build_buildings.py add_props): a swing set across its circle, and the obstacle course.
PROPS = [
    {"kind": "swings", "since": 2020, "at": (175108.6, 448413.3), "angle": 18.0},
    {"kind": "obstacles", "since": 2020, "at": (175089.7, 448413.9), "r": 7.5},
]

# Where children play, in RD meters: the schoolyard west of the old canteen and the terrace of the after-school care
# on its top floor (3D BAG has that roof at 7.3 m).
PLAY = [
    {"name": "Schoolplein", "since": 2021, "area": [(174992.5, 448489.0), (175014.0, 448482.5), (175008.0, 448462.0), (174995.0, 448465.0), (174988.0, 448477.0)], "h": 0.0, "kids": 16},
    # The playground on the lawn by the Ketelhuis, in two sand circles (OpenStreetMap): swings in the small one, an
    # obstacle course in the big one.
    {"name": "Speeltuin: schommels", "since": 2020, "area": circle((175108.6, 448413.3), 5.1), "h": 0.0, "kids": 3,
     "swing": {"at": (175108.6, 448413.3), "angle": 18.0}},
    {"name": "Speeltuin: hindernisbaan", "since": 2020, "area": circle((175089.7, 448413.9), 8.9), "h": 0.0, "kids": 6},
    {"name": "Terras van de BSO", "since": 2021, "area": [(175012.5, 448480.6), (175016.4, 448479.6), (175005.0, 448432.4), (175001.4, 448433.3)], "h": 7.3, "kids": 6},
]


# Bridges and viaducts from OpenStreetMap (bridge=*, layer >= 1): their deck height, how far the road climbs to it on
# either side, and how wide the deck is, by kind of road.
AUTO_DECK = 6.5
AUTO_RAMP = {"motorway": 130, "motorway_link": 90, "trunk": 110, "primary": 90, "secondary": 80}
AUTO_WIDTH = {"motorway": 13, "motorway_link": 7, "trunk": 11, "primary": 10, "secondary": 9, "tertiary": 9,
              "unclassified": 7, "residential": 7, "service": 5, "cycleway": 4, "footway": 3, "path": 3}


def auto_bridges(manual):
    """A deck with a ramp up and down for every road bridge in OpenStreetMap that the hand-made BRIDGES don't cover:
    the bridge itself at AUTO_DECK meters, the road climbing to it along the ways it continues in."""
    from shapely.geometry import LineString, Point

    path = RAW / "osm_roads.json"
    if not path.exists():
        return []
    ways = [w for w in json.loads(path.read_text())["elements"] if w["type"] == "way" and w.get("geometry")]
    key = lambda g: (round(g["lat"], 7), round(g["lon"], 7))
    ends = {}
    for w in ways:
        for g in (w["geometry"][0], w["geometry"][-1]):
            ends.setdefault(key(g), []).append(w)
    rd = lambda w: [wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]]
    taken = [LineString(b["path"]) for b in manual]

    def ramp(w, end, length):
        """Points along the ways that continue from `end` of w (0 start, -1 end), outward, `length` meters long."""
        pts, here, prev, left = [], w["geometry"][end], w, length
        while left > 0:
            nxt = [v for v in ends.get(key(here), []) if v is not prev and v["tags"].get("highway") == w["tags"]["highway"]]
            if not nxt:
                break
            v = nxt[0]
            g = v["geometry"] if key(v["geometry"][0]) == key(here) else v["geometry"][::-1]
            for a, b in zip(g, g[1:]):
                pa, pb = np.array(wgs_to_rd(a["lat"], a["lon"])), np.array(wgs_to_rd(b["lat"], b["lon"]))
                d = np.linalg.norm(pb - pa)
                if d >= left:
                    pts.append(tuple(pa + (pb - pa) * left / d))
                    left = 0
                    break
                pts.append(tuple(pb))
                left -= d
            here, prev = g[-1], v
        return pts

    out = []
    for w in ways:
        t = w["tags"]
        hw = t.get("highway", "").removesuffix("_link") if t.get("highway") != "motorway_link" else "motorway_link"
        if t.get("bridge") in (None, "no") or int(t.get("layer", "1") or 1) < 1 or hw not in AUTO_WIDTH:
            continue
        span = rd(w)
        if any(LineString(span).distance(m) < 5 for m in taken):
            continue
        length = AUTO_RAMP.get(hw, 60)
        before, after = ramp(w, 0, length)[::-1], ramp(w, -1, length)
        pts = before + span + after
        out.append({"name": t.get("name") or t.get("ref") or f"brug {w['id']}", "path": [(round(x, 2), round(y, 2)) for x, y in pts],
                    "span": (len(before), len(before) + len(span) - 1), "height": AUTO_DECK, "width": AUTO_WIDTH[hw],
                    "deck": (98, 100, 106), "wall": (196, 194, 186), "rail": (150, 154, 158)})
    print(f"{len(out)} bridges from OpenStreetMap")
    return out


def era_skipped(name):
    """Whether a place is left out of the era being built, so its series can do without it."""
    return any(e["name"] == name and not in_era(e) for e in BUILDINGS + POINTS)


def bridge_heights(b):
    """The deck height at every point of a bridge's path: full over the span, sloping evenly to the ground at both ends."""
    p = np.asarray(b["path"], dtype=np.float64)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s0, s1, h = cum[b["span"][0]], cum[b["span"][1]], b["height"]
    up = cum / s0 if s0 > 0 else np.ones_like(cum)  # no ramp on a side: at full height from its end
    down = (cum[-1] - cum) / (cum[-1] - s1) if cum[-1] > s1 else np.ones_like(cum)
    return [round(float(v), 2) for v in h * np.clip(np.minimum(up, down), 0, 1)]


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
    for b in filter(in_era, BUILDINGS):
        if "addrs" in b or "ats" in b:
            pts = b.get("ats") or [address_xy(a) for a in b["addrs"]]
            pids = [building_at(rings, *pt)[0] for pt in pts]
            assert all(pids), f"{b['name']}: no building at {b.get('addrs') or b['ats']}"
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
    for p in filter(in_era, POINTS):
        x, y = p["at"] if "at" in p else address_xy(p["addr"]) if "addr" in p else wgs_to_rd(*p["latlon"])
        out["places"].append({"name": p["name"], "note": p.get("note"), "x": x, "y": y, "h": 2.0})
        out["hotspots"].append({"name": p["name"], "x": x, "y": y, **{k: p[k] for k in INFO if k in p}})
    for b in filter(in_era, STYLED):
        pts = [address_xy(a) for a in b["addrs"]]
        for pid in {building_at(rings, *pt)[0] for pt in pts}:
            assert pid, f"no building at {b['addrs']}"
            out["buildings"][pid] = {"name": ", ".join(b["addrs"]), **{k: v for k, v in b.items() if k != "addrs"}}
            print(f"{', '.join(b['addrs'])}: restyled {pid}")
    out["chimneys"] = [{**{k: v for k, v in c.items() if k != "at"}, "x": c["at"][0], "y": c["at"][1]} for c in filter(in_era, CHIMNEYS)]
    out["stations"] = list(filter(in_era, STATIONS))
    bridges = list(filter(in_era, BRIDGES))
    bridges += auto_bridges(bridges)
    out["bridges"] = [{**b, "heights": bridge_heights(b)} for b in bridges]
    out["ground"], out["play"] = list(filter(in_era, GROUND)), list(filter(in_era, PLAY))
    out["props"] = list(filter(in_era, PROPS))
    for m in MASTS:
        x, y = wgs_to_rd(*m["latlon"])
        out["masts"].append({**{k: v for k, v in m.items() if k != "latlon"}, "x": x, "y": y})
    names = {h["name"] for h in out["hotspots"]}
    for key, label, members in CATEGORIES:
        members = [n for n in members if n in names or not era_skipped(n)]
        missing = [n for n in members if n not in names]
        assert not missing, f"{label}: no hotspot for {missing}"
        out["categories"].append({"key": key, "label": label, "names": members})
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "landmarks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
