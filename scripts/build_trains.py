# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "shapely"]
# ///
"""Train routes for the viewer: the tracks each service follows over the map, from OpenStreetMap.

The tracks form a graph on their shared nodes. A route is the shortest way through it that a train can take: never
turning more than a switch allows, on the right-hand track (as in the Netherlands), and for a train that stops, along
a platform. A train that ends at Ede-Wageningen (the Sprinter from Arnhem, the Valleilijn from Barneveld) has a
second route for the way back, which starts where its last carriage stood and so lets it leave over a crossover.

Output: data/build/trains.json
  routes  {key: {"p": [[x, z], ...] local meters, "stops": [{"s": meters along p to the middle of the platform,
          "station": name}], "front": for a return route, how far along p the front of the train stands at the start,
          "h", "t": per view, stretches behind buildings and under trees, as in traffic.json}}
  views   the views those stretches are for
"""

import heapq
import json
import math

import numpy as np
from shapely.geometry import LineString, Point

from build_traffic import DECK_LANES, hidden
from config import ERA, BUILD, CX, CY, RAW, VIEWS, XMAX, XMIN, YMAX, YMIN
from geo import wgs_to_rd

MAX_TURN = math.radians(35)  # sharper than any switch: no reversing through a junction
STATIONS = {"Ede-Wageningen": None, "Ede Centrum": None}
# The platform track each service takes at Ede-Wageningen, by the platform's number. Trains to Arnhem keep to the
# south tracks (5), to Utrecht to the north side of the island (3); the Sprinter from Arnhem crosses over to 4, the
# eastbound side of the island, and so leaves straight back to Arnhem; the Valleilijn turns on 1. (In the
# OpenStreetMap tracks, 5 can only be reached from the south.)
TRACK = {"ic_e": "5", "ic_w": "3", "thru_e": "5", "thru_w": "3", "sprinter": "4", "rr": "1"}
# Before the station moved east of the Klinkerbergerweg (OpenStreetMap of 2012, for the earlier eras): one island
# platform between the two main tracks, its north face "2" and its south face "3", and a platform "1" on the north side
# for the trains that turn. OpenStreetMap had no numbers on them then; load() names them so.
TRACK_THEN = {"ic_e": "3", "ic_w": "2", "thru_e": "3", "thru_w": "2", "sprinter": "1", "rr": "1"}
# Where on a shared platform a turning train stops, in meters east of the platform's middle: the Sprinter from Arnhem
# at the east end of platform 1, the Valleilijn at the west end.
AT_THEN = {"sprinter": 55.0, "rr": -55.0}
# Lengths of the trains that turn back at Ede-Wageningen (render/viewer TRAINS has the same carriages).
TURNING = {"sprinter": 40.5, "rr": 64.0}


def local(lat, lon):
    x, y = wgs_to_rd(lat, lon)
    return x - CX, CY - y


def load():
    els = json.loads((RAW / ("osm_rail.json" if ERA is None else "osm_rail-2012.json")).read_text())["elements"]
    key = lambda p: (round(p["lat"], 7), round(p["lon"], 7))
    pos, adj = {}, {}
    platforms = []
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "node" and t.get("railway") in ("station", "halt") and t.get("name") in STATIONS:
            STATIONS[t["name"]] = np.array(local(e["lat"], e["lon"]))
        if e["type"] != "way" or not e.get("geometry"):
            continue
        if t.get("railway") == "platform":
            platforms.append((t.get("ref"), LineString([local(p["lat"], p["lon"]) for p in e["geometry"]])))
            continue
        if t.get("railway") != "rail" or t.get("service") in ("yard", "siding", "spur") or t.get("usage") == "industrial":
            continue
        ks = [key(p) for p in e["geometry"]]
        for k, p in zip(ks, e["geometry"]):
            pos[k] = np.array(local(p["lat"], p["lon"]))
        for a, b in zip(ks, ks[1:]):
            if a != b:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
    return pos, adj, platforms


def route(pos, adj, starts, goal, penalty=lambda a, b: 1.0):
    """Shortest path over directed track segments [(a, b), ...] from any start segment to one where goal(a, b),
    turning at most MAX_TURN at each node. Returns the node list."""
    unit = lambda a, b: (pos[b] - pos[a]) / (np.linalg.norm(pos[b] - pos[a]) or 1)
    dist, prev, heap = {}, {}, []
    for a, b in starts:
        d = np.linalg.norm(pos[b] - pos[a]) * penalty(a, b)
        dist[(a, b)] = d
        heapq.heappush(heap, (d, a, b))
    while heap:
        d, a, b = heapq.heappop(heap)
        if d > dist.get((a, b), math.inf):
            continue
        if goal(a, b):
            path = [b, a]
            while (a, b) in prev:
                a, b = prev[(a, b)]
                path.append(a)
            return path[::-1]
        u = unit(a, b)
        for c in adj[b]:
            if c == a or np.dot(u, unit(b, c)) < math.cos(MAX_TURN):
                continue
            nd = d + np.linalg.norm(pos[c] - pos[b]) * penalty(b, c)
            if nd < dist.get((b, c), math.inf):
                dist[(b, c)], prev[(b, c)] = nd, (a, b)
                heapq.heappush(heap, (nd, b, c))
    raise RuntimeError("no route")


def ends(pos, adj):
    """Where the tracks come onto the map, as directed segments pointing into it: (a node outside, its neighbour
    inside)."""
    hx, hz = (XMAX - XMIN) / 2, (YMAX - YMIN) / 2
    inside = lambda k: abs(pos[k][0]) <= hx and abs(pos[k][1]) <= hz
    return [(k, n) for k, nb in adj.items() if not inside(k) for n in nb if inside(n)]


def polyline(pos, nodes):
    return np.array([pos[n] for n in nodes])


def cumlen(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def at(p, cum, s):
    return np.array([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


def cut(p, s0, s1):
    """The polyline between s0 and s1 meters along it."""
    cum = cumlen(p)
    inner = p[(cum > s0) & (cum < s1)]
    return np.vstack([at(p, cum, s0), inner, at(p, cum, s1)])


def clip(p):
    """The part of a route on the map, and how many meters were cut from its start."""
    hx, hz = (XMAX - XMIN) / 2 - 1, (YMAX - YMIN) / 2 - 1
    cum = cumlen(p)
    s = np.arange(0, cum[-1], 1.0)
    xy = np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], 1)
    inside = (np.abs(xy[:, 0]) < hx) & (np.abs(xy[:, 1]) < hz)
    idx = np.nonzero(inside)[0]
    s0, s1 = s[idx[0]], s[idx[-1]]
    return cut(p, s0, s1), s0


def platform_stop(p, platforms, station, near=400):
    """Meters along p to the middle of the platform it runs along at a station."""
    line = LineString(p)
    best = None
    for _, pl in platforms:
        c = pl.centroid
        if np.hypot(c.x - station[0], c.y - station[1]) > near:
            continue
        d = line.distance(c)
        if d < 12 and (best is None or d < best[0]):
            best = (d, line.project(c))
    if best is None:
        return line.project(Point(*station))
    return best[1]


def main():
    pos, adj, platforms = load()
    ede, centrum = STATIONS["Ede-Wageningen"], STATIONS["Ede Centrum"]
    entries = ends(pos, adj)
    hx, hz = (XMAX - XMIN) / 2, (YMAX - YMIN) / 2

    def side(e):
        """The edge of the map the track from this end comes in by: walk in from the end to the first point on the
        map and see which edge it is nearest (the far ends can lie anywhere, past a corner)."""
        prev, here = e
        while abs(pos[here][0]) > hx or abs(pos[here][1]) > hz:
            nxt = [n for n in adj[here] if n != prev]
            if not nxt:
                break
            prev, here = here, nxt[0]
        # Where the last stretch from outside crosses the edge.
        (x0, z0), (x1, z1) = pos[prev], pos[here]
        t = min([f for f in ((-hx - x0) / (x1 - x0) if x1 != x0 else 2, (hx - x0) / (x1 - x0) if x1 != x0 else 2,
                             (-hz - z0) / (z1 - z0) if z1 != z0 else 2, (hz - z0) / (z1 - z0) if z1 != z0 else 2)
                  if 0 <= f <= 1 and abs(x0 + f * (x1 - x0)) <= hx + 0.01 and abs(z0 + f * (z1 - z0)) <= hz + 0.01] or [1])
        x, z = x0 + t * (x1 - x0), z0 + t * (z1 - z0)
        return min((hx + x, "west"), (hx - x, "east"), (hz + z, "north"), (hz - z, "south"))[1], z

    sides = {e: side(e) for e in entries}
    west = [e for e in entries if sides[e][0] == "west"]
    east = [e for e in entries if sides[e][0] == "east"]
    north = [e for e in entries if sides[e][0] == "north"]
    if not north and len(west) > 2:
        # On a small map the line to Barneveld leaves by the west edge too: it is the northernmost track there.
        north = [min(west, key=lambda e: sides[e][1])]
        west.remove(north[0])
    assert len(west) >= 2 and len(east) >= 2 and north, (len(west), len(east), len(north))
    exit_of = lambda e: (lambda a, b: b == e[0])

    def edge_z(end, others, x):
        """Where the track from this end crosses x (the tracks swap sides on flyovers further out)."""
        p = polyline(pos, route(pos, adj, [end], lambda a, b: any(b == o[0] for o in others)))
        i = int(np.nonzero(np.diff(np.sign(p[:, 0] - x)))[0][0])
        t = (x - p[i, 0]) / (p[i + 1, 0] - p[i, 0])
        return p[i, 1] + t * (p[i + 1, 1] - p[i, 1])

    hx = (XMAX - XMIN) / 2 - 50
    west.sort(key=lambda e: edge_z(e, east, -hx))
    east.sort(key=lambda e: edge_z(e, west, hx))
    # z runs south: driving east the right-hand track is the southern one (larger z), driving west the northern one.
    w_north, w_south, e_north, e_south = west[0], west[-1], east[0], east[-1]
    # Which platform track a node lies on: the platform at Ede-Wageningen beside it, measured across the line.
    here = [(ref, pl) for ref, pl in platforms if np.hypot(pl.centroid.x - ede[0], pl.centroid.y - ede[1]) < 400]
    xy = np.concatenate([np.asarray(pl.coords if pl.geom_type == "LineString" else pl.exterior.coords) for _, pl in here])
    axis = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, -1]
    across = lambda q: float(axis[0] * (q[1] - ede[1]) - axis[1] * (q[0] - ede[0]))
    along = lambda q: float(np.dot(q - ede, axis))
    sides = [(ref, across(np.array(pl.centroid.coords[0]))) for ref, pl in here]
    if not all(ref for ref, _ in sides):
        # Unnumbered (the old station): the platform farthest out is 1; the other is an island with a face on each
        # side, 2 towards platform 1 and 3 away from it.
        (_, out), (_, mid) = sorted(sides, key=lambda rs: -abs(rs[1]))[:2]
        toward = np.sign(out - mid)
        sides = [("1", out), ("2", mid + toward * 3.0), ("3", mid - toward * 3.0)]
    plan = TRACK if ERA is None else TRACK_THEN
    track = {}
    for a, q in pos.items():
        if abs(along(q)) < 360 and abs(across(q)) < 40:
            track[a] = min(sides, key=lambda rs: abs(rs[1] - across(q)))[0]

    def keep_to(ref):
        """Through the station on the given platform track; elsewhere any track."""
        return lambda a, b: 20.0 if b in track and track[b] != ref else 1.0

    def platform_goal(ref):
        return lambda a, b: track.get(a) == ref and track.get(b) == ref and abs(along(pos[b])) < 120

    routes = {}

    def stops_on(p, stations):
        """Stops along a route; a station off the map (it projects onto an end of the route) is left out."""
        length = cumlen(p)[-1]
        out = [{"s": round(float(platform_stop(p, platforms, STATIONS[st])), 1), "station": st} for st in stations]
        return [st for st in out if 5 < st["s"] < length - 5 or st["station"] == "Ede-Wageningen"]


    def add(key, nodes, stations=()):
        p = clip(polyline(pos, nodes))[0]
        routes[key] = {"p": p, "stops": stops_on(p, stations)}

    add("ic_e", route(pos, adj, [w_south], exit_of(e_south), keep_to(plan["ic_e"])), ["Ede-Wageningen"])
    add("ic_w", route(pos, adj, [e_north], exit_of(w_north), keep_to(plan["ic_w"])), ["Ede-Wageningen"])
    add("thru_e", route(pos, adj, [w_south], exit_of(e_south), keep_to(plan["thru_e"])))
    add("thru_w", route(pos, adj, [e_north], exit_of(w_north), keep_to(plan["thru_w"])))

    def turning(key, entry, back_to, length, via=()):
        """In from an end of the map to a platform at Ede-Wageningen, and back out again from there."""
        ref = plan[key]
        nodes = route(pos, adj, [entry], platform_goal(ref), keep_to(ref))
        try:  # on along the platform track to its far end, so the whole platform lies on the route
            nodes += route(pos, adj, [(nodes[-2], nodes[-1])], lambda a, b: track.get(b) != ref)[2:]
        except RuntimeError:
            pass
        p = polyline(pos, nodes)
        s_mid = platform_stop(p, platforms, ede)
        if ERA is not None and key in AT_THEN:
            # Shift along the route to the right end of the platform, whichever way the train runs there.
            cum = cumlen(p)
            east = along(at(p, cum, s_mid + 1)) - along(at(p, cum, s_mid))
            s_mid += AT_THEN[key] * np.sign(east) * np.sign(axis[0] or 1)
        front, tail = s_mid + length / 2, s_mid - length / 2
        p_in, trim = clip(cut(p, 0, front))
        routes[f"{key}_in"] = {"p": p_in, "stops": [*stops_on(p_in, via), {"s": round(float(s_mid - trim), 1), "station": "Ede-Wageningen"}]}
        # The way back starts at the front of the stopped train and runs back along it; past its last carriage it
        # may take any track to the exit.
        k = int(np.searchsorted(cumlen(p), tail))  # the first node between the tail and the front
        back = route(pos, adj, [(nodes[k], nodes[k - 1])], exit_of(back_to))
        p_out, trim = clip(np.vstack([cut(p, tail, front)[::-1], polyline(pos, back[1:])]))
        print(f"{key}: in on track {ref}")
        routes[f"{key}_out"] = {"p": p_out, "stops": stops_on(p_out, via), "front": round(length - trim, 1)}

    turning("sprinter", e_north, e_south, TURNING["sprinter"])
    turning("rr", north[0], north[0], TURNING["rr"], via=["Ede Centrum"])

    # Under the road bridges, as the cars and walkers are (scripts/build_traffic.py writes them into traffic.json).
    traffic = json.loads((BUILD / "traffic.json").read_text())
    DECK_LANES[:] = [ln for ln in traffic["lanes"] if ln.get("layer", 0) > 0]
    lanes = [{"p": r["p"]} for r in routes.values()]
    hidden(lanes)
    out = {}
    for (k, r), ln in zip(routes.items(), lanes):
        out[k] = {**r, "p": [[round(float(x), 1), round(float(z), 1)] for x, z in r["p"]], "h": ln["h"], "t": ln["t"], "u": ln["u"]}
        print(f"{k}: {cumlen(r['p'])[-1]:.0f} m, stops {[(s['station'], s['s']) for s in r['stops']]}" + (f", front {r['front']}" if "front" in r else ""))
    (BUILD / "trains.json").write_text(json.dumps({"views": list(VIEWS), "routes": out}, separators=(",", ":")))


if __name__ == "__main__":
    main()
