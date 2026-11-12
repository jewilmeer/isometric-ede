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
