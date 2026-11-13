"""What differs in an earlier era (config.ERA): where the map has changed since, and the AHN to use.

The changed area is everything within 20 m of a 3D BAG building built after the era's year, closed over gaps of
30 m: there the streets, gardens and squares of today did not exist yet. scripts/build_ground.py paints it bare sand
inside the era's "bare" box (a building site then) and lawn elsewhere, scripts/build_traffic.py takes its roads out.
"""

import json
from functools import cache

import numpy as np

from config import CX, CY, ERA, ERA_CONF, RAW

AHN = RAW / "ahn2" / "chunks" if ERA_CONF.get("ahn2") else RAW / "ahn"


def built_after(attrs):
    """Whether a 3D BAG building did not stand yet in the era."""
    return ERA is not None and (attrs.get("oorspronkelijkbouwjaar") or 0) > ERA_CONF["built"]


@cache
def changed():
    """The changed area as a shapely geometry in RD meters (empty for now)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if ERA is None:
        return Polygon()
    parts = []
    for line in open(RAW / "3dbag.jsonl"):
        f = json.loads(line)
        co = f["CityObjects"][f["id"]]
        if not built_after(co["attributes"]):
            continue
        t = f["transform"]
        ring = np.array(f["vertices"], dtype=np.float64)[co["geometry"][0]["boundaries"][0][0]] * t["scale"] + t["translate"]
        parts.append(Polygon(ring[:, :2]).buffer(20 + 15))
    return unary_union(parts).buffer(-15)


@cache
def bare():
    """The part of the changed area that was a bare building site, in RD meters."""
    from shapely.geometry import Polygon, box

    b = ERA_CONF.get("bare")
    return changed().intersection(box(*b)) if b else Polygon()


def local(geom):
    """An RD geometry in the render's local meters (x east, z south)."""
    from shapely import affinity

    return affinity.affine_transform(geom, [1, 0, 0, -1, -CX, CY])


@cache
def underpass_ids():
    """OpenStreetMap ways of today tagged bridge=yes that really run under the railway: the road is in a short tunnel and
    the rail is the bridge. They are recognised by what the BGT says: the way lies on a stretch of road at a relative height
    of -1 (wegdeel), no bridge deck (overbruggingsdeel) lies over it, and a railway runs within 40 m. Real road bridges
    over a railway have a BGT deck over them, so they stay."""
    from shapely.geometry import LineString, shape
    from shapely.ops import unary_union

    from geo import wgs_to_rd

    line = lambda w: LineString([wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]])
    way = lambda p: [w for w in json.loads(p.read_text())["elements"] if w["type"] == "way" and w.get("geometry")]
    roads = way(RAW / "osm_roads.json")
    rails = [line(w) for f in ("osm_rail.json", "osm_rail-2012.json") if (RAW / f).exists() for w in way(RAW / f)
             if w["tags"].get("railway") == "rail"]
    rail = unary_union(rails)
    deck = unary_union([shape(f["geometry"]).buffer(0) for f in json.loads((RAW / "bgt_overbruggingsdeel.geojson").read_text())["features"]])
    under = unary_union([shape(f["geometry"]).buffer(0) for f in json.loads((RAW / "bgt_wegdeel.geojson").read_text())["features"]
                         if (f["properties"].get("relatieve_hoogteligging") or 0) < 0])
    out = set()
    for w in roads:
        t = w["tags"]
        if t.get("bridge") in (None, "no") or t.get("highway") in ("motorway", "motorway_link", "trunk"):
            continue
        ln = line(w)
        if ln.length < 200 and ln.intersection(deck).length < 0.3 * ln.length and ln.distance(under) < 2.5 and ln.distance(rail) < 40:
            out.add(w["id"])
    return out


@cache
def under_polygons(with_decks=True):
    """Where a road runs under something (BGT wegdeel at relative height -1) and, with_decks, where a bridge deck lies, as
    one RD geometry: no trees there; what drives under it is hidden (scripts/build_traffic.py, without the decks: trains
    run over some)."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    parts = [shape(f["geometry"]).buffer(0) for f in json.loads((RAW / "bgt_wegdeel.geojson").read_text())["features"]
             if (f["properties"].get("relatieve_hoogteligging") or 0) < 0]
    if with_decks:
        parts += [shape(f["geometry"]).buffer(0) for f in json.loads((RAW / "bgt_overbruggingsdeel.geojson").read_text())["features"]]
    return unary_union(parts)


STATION_OLD = (174565.0, 448867.0)  # the old station Ede-Wageningen


@cache
def rail_zone(meters=12.0):
    """RD geometry within `meters` of any railway track (OpenStreetMap of today and of 2012, BGT spoor), except round the
    old station: trains, platforms and catenary look like buildings in the AHN and the topographic map."""
    from shapely.geometry import LineString, Point, shape
    from shapely.ops import unary_union

    from geo import wgs_to_rd

    lines = []
    for f in ("osm_rail.json", "osm_rail-2012.json"):
        if (RAW / f).exists():
            for w in json.loads((RAW / f).read_text())["elements"]:
                if w["type"] == "way" and w.get("geometry") and w["tags"].get("railway") in ("rail", "light_rail", "tram"):
                    lines.append(LineString([wgs_to_rd(g["lat"], g["lon"]) for g in w["geometry"]]))
    for f in json.loads((RAW / "bgt_spoor.geojson").read_text())["features"]:
        if f["geometry"]:
            lines.append(shape(f["geometry"]))
    zone = unary_union([ln.buffer(meters) for ln in lines])
    return zone.difference(Point(*STATION_OLD).buffer(40))
