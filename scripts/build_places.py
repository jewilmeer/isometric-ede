"""The viewer's search list: named places first, then every address inside the map."""

import json

from config import BUILD, RAW, XMAX, XMIN, YMAX, YMIN


def main():
    lm = json.loads((BUILD / "landmarks.json").read_text())
    places = [{"name": p["name"], "note": p.get("note"), "x": round(p["x"], 1), "y": round(p["y"], 1), "h": p["h"]} for p in lm["places"]]
    for x, y, label in json.loads((RAW / "addresses.json").read_text()):
        if XMIN <= x <= XMAX and YMIN <= y <= YMAX:
            places.append({"name": label.split(", ")[0], "x": round(x, 1), "y": round(y, 1), "h": 4})
    (BUILD / "places.json").write_text(json.dumps(places, ensure_ascii=False))
    print(f"{len(places)} places")


if __name__ == "__main__":
    main()
