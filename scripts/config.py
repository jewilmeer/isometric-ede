"""Shared settings: the area we render, in RD New (EPSG:28992) meters."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# The year the map shows: unset for now, or one of ERAS (ERA=2009 ./build.sh). Each era builds into its own folder.
ERA = os.environ.get("ERA") or None
# A try-out build (PROTO=look uv run build_ground.py): today's map, but only PROTO_AREA, into data/build-<name>, so a
# change to a script shows in a minute instead of after a full build. Render it with render/run.mjs --era <name>.
PROTO = os.environ.get("PROTO") or None
BUILD = ROOT / "data" / (f"build-{PROTO}" if PROTO else "build" if ERA is None else f"build-{ERA}")
NOW = ROOT / "data" / "build"

# The years the map can also show, and what they are made of:
#   ahn2       heights and trees from AHN2 (data/raw/ahn2, 2007-2012) instead of the current AHN
#   built      only 3D BAG buildings from this year or before; what has been demolished since comes from AHN2
#   bare       where the ground was bare then (scripts/build_ground.py paints it sand): RD box (xmin, ymin, xmax, ymax)
#   topo       the buildings demolished since come from the Kadaster's topographic map of this year (data/raw/topo<year>,
#              scripts/topo1987_fetch.py) instead of from AHN2
#   topo_blind where AHN2 measured nothing (the military terrain round the barracks), the buildings demolished since
#              come from the topographic map of this year instead, and the trees from today's AHN
#   osm        "2012": also the buildings in OpenStreetMap of 2012 (data/raw/osm_buildings-2012.json) that nothing else
#              found, such as the old station
#   osm_area   an RD box where those count too although nothing of today's 3D BAG marks it as changed (the station)
#   paving     meters of paving round those buildings in the changed area (default 6; 15 where AHN2 is blind)
ERAS = {
    "2009": {"label": "2009", "ahn2": True, "built": 2009, "bare": (174750, 448100, 175500, 448700), "topo_blind": 2008, "osm": "2012",
             "osm_area": (174300, 448750, 174950, 449000)},
    "1987": {"label": "1987", "ahn2": True, "built": 1987, "bare": None, "topo": 1987, "paving": 15, "osm": "2012"},
}
ERA_CONF = ERAS.get(ERA, {})
# Terrains whose street plan has changed completely since (RD): in an earlier era their ground comes from that year's
# topographic map, none of today's streets on it (scripts/build_ground.py paint_topo). The Enka terrain and the strip
# up to the railway, from the Klinkenbergerweg to the east fence, track and all; and the barracks terrains, where the
# Veluwse Poort and its new streets (the Generaal Hackettlaan through the park among them) came after 2010: the
# Mauritskazerne with the park south of it, and the Elias Beeckman, Arthur Kool, Van Essen and Simon Stevin barracks.
OLD_TERRAIN = [
    [(174600, 448860), (174600, 448190), (175000, 448170), (175460, 448140), (175460, 448580), (175000, 448720), (174750, 448830)],
    [(174560, 449845), (175010, 449845), (175020, 449000), (174780, 449000), (174730, 449490), (174560, 449500)],
    [(175150, 450060), (175570, 450080), (175900, 450130), (175915, 450370), (176270, 450370), (176270, 449840), (176000, 449790),
     (175980, 449740), (175300, 449720), (175150, 449800)],
]
# Today's streets that are new along their whole length (OpenStreetMap names): 25 m either side of them counts as old
# terrain too. The Generaal Hackettlaan, the "Parklaan" across the barracks terrains, opened 2020-2022 on the line of the
# narrow Kazernelaan and Nieuwe Kazernelaan.
NEW_STREETS = ["Generaal Hackettlaan"]


def in_era(entry):
    """Whether a landmark (with optional "since" and "until" years) exists in the era being built."""
    year = ERA_CONF.get("built", 9999)
    return entry.get("since", 0) <= year and year < entry.get("until", 99999)

# Ede around the Enka-terrein: the Enka chimney (175084, 448248) sits in the south-east quarter of the old 4 x 4 km,
# with station Ede-Wageningen in the middle and the old centre around the Oude Kerk at the top. Each era has its own
# area, all on the same 500 m grid (the chunks of the raw data are named after their RD corner, so they are shared).
# Today's map is the largest (6 x 5 km, from Kernhem to Veldhuizen-Oost); the earlier years only need the part the old
# maps and AHN2 cover.
AREAS = {
    None: (170750, 447000, 176750, 452000),
    "2009": (174250, 447500, 176750, 450500),
    "1987": (174250, 447500, 176750, 450500),
}
# The try-out area: 1 x 1 km with the station, the Enka terrain and the streets of terraced houses north of the railway.
PROTO_AREA = (174250, 448000, 175250, 449000)
AREA = PROTO_AREA if PROTO else AREAS[ERA]
XMIN, YMIN, XMAX, YMAX = AREA
NOW_AREA = AREAS[None]  # what the bbox downloads cover
CX, CY = (XMIN + XMAX) / 2, (YMIN + YMAX) / 2

# The BAG woonplaats for addresses, and its postcode areas (the address search pages no further than 10000 results).
PLACE = "Ede"
POSTCODES = ("6711", "6712", "6713", "6714", "6715", "6716", "6717", "6718")

# Downloads and ground textures come in square chunks of this many meters.
CHUNK = 500
NX, NY = int((XMAX - XMIN) / CHUNK), int((YMAX - YMIN) / CHUNK)


def chunk_name(ix, iy):
    """The name of a chunk's raw data: the RD of its lower-left corner, so that every era's area finds the same files."""
    return f"{XMIN + ix * CHUNK}_{YMIN + iy * CHUNK}"

# Ground raster resolution in meters per pixel.
GROUND_RES = 0.25

# The compass directions the map is rendered from (render/run.mjs --look); the first is the one the viewer opens on.
VIEWS = (45, 135, 225, 315)


def write_area():
    """The renderer and the viewer read the area from here."""
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "area.json").write_text(
        json.dumps({"xmin": XMIN, "ymin": YMIN, "xmax": XMAX, "ymax": YMAX, "cx": CX, "cy": CY, "chunk": CHUNK, "nx": NX, "ny": NY})
    )
