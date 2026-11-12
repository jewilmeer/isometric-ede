"""Ede van toen: a guided tour along the history of the Enka-terrein and its surroundings, as an overlay for the
viewer. The stops are still to be written; with none, the viewer hides the tour.

Photos come from Wikimedia Commons with an open license; they are downloaded once and published with the site, with
credits.

Output: data/build/history.json and data/build/history/*.jpg
"""

import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request

from config import BUILD
from geo import wgs_to_rd

STOPS = []


def commons_thumb(name, width=960):  # Wikimedia only serves a fixed set of thumbnail widths
    h = hashlib.md5(name.encode()).hexdigest()
    q = urllib.parse.quote(name)
    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{q}/{width}px-{q}"


def download(url, tries=6):
    """Wikimedia rate-limits bursts of thumbnail requests; wait politely and retry."""
    req = urllib.request.Request(url, headers={"User-Agent": "isometric-ede/0.1 (personal map)"})
    for attempt in range(tries):
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            time.sleep(3)
            return data
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(15 * (attempt + 1))


def route_order(stops):
    """Visit the stops as a route over the map instead of by date: the first and last stay put, the rest is a
    nearest-neighbour path tidied with 2-opt, so dots that lie together come one after another. Stops on the same
    spot (the old and the new church) keep their chronological order."""
    first, last, middle = stops[0], stops[-1], stops[1:-1]
    dist = lambda a, b: ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
    path, todo = [first], middle[:]
    while todo:
        nxt = min(todo, key=lambda s: (dist(path[-1], s), stops.index(s)))
        path.append(nxt)
        todo.remove(nxt)
    path.append(last)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                before = dist(path[i - 1], path[i]) + dist(path[j], path[j + 1])
                after = dist(path[i - 1], path[j]) + dist(path[i], path[j + 1])
                if after < before - 0.01:
                    path[i : j + 1] = path[i : j + 1][::-1]
                    improved = True
    # Places within 40 m of each other (church, parsonage, Sunday school) in chronological order.
    i = 0
    while i < len(path):
        j = i
        while j + 1 < len(path) - 1 and i > 0 and dist(path[j + 1], path[i]) < 40:
            j += 1
        year = lambda st: int(m.group()) if (m := re.search(r"\d{4}", st["year"])) else 9999
        path[i : j + 1] = sorted(path[i : j + 1], key=lambda st: (year(st), stops.index(st)))
        i = j + 1
    return path


def main():
    out_dir = BUILD / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    stops = []
    for i, s in enumerate(STOPS, start=1):
        x, y = s["at"] if "at" in s else wgs_to_rd(*s["latlon"])
        stop = {"n": i, "year": s["year"], "title": s["title"], "text": s["text"], "x": round(x, 1), "y": round(y, 1),
                "h": s.get("h", 0), "zoom": s.get("zoom", 1.2),
                "source": {"label": s["source"][0], "url": s["source"][1]}, "more": s.get("more")}
        stop["photos"] = []
        for k, (name, credit) in enumerate(s.get("photos", [])):
            dst = out_dir / f"{i:02d}_{k}.jpg"
            if not dst.exists():
                dst.write_bytes(download(commons_thumb(name)))
                subprocess.run(["magick", str(dst), "-resize", "640x640>", "-quality", "82", str(dst)], check=True)
            stop["photos"].append({"src": f"history/{dst.name}", "credit": credit,
                                   "url": f"https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(name)}"})
        stops.append(stop)
    stops = route_order(stops) if stops else []
    for n, stop in enumerate(stops, start=1):
        stop["n"] = n
    # A version in every photo URL, so browsers never show a cached photo that has since been replaced.
    version = hashlib.md5(b"".join(f.read_bytes() for f in sorted(out_dir.glob("*.jpg")))).hexdigest()[:8]
    for stop in stops:
        for ph in stop["photos"]:
            ph["src"] += f"?v={version}"
    (BUILD / "history.json").write_text(json.dumps({"stops": stops}, ensure_ascii=False, indent=1))
    print(f"{len(stops)} stops, {sum(len(s['photos']) for s in stops)} photos on {sum(bool(s['photos']) for s in stops)} stops")


if __name__ == "__main__":
    main()
