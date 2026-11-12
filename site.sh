#!/usr/bin/env bash
# Cut the rendered map into Deep Zoom pyramids and assemble site/, without rendering anything.
#   ./site.sh              from out/full-4.png, and out/full-4-v<look>.png for the other views in scripts/config.py,
#                          and out/full-4-<era>[-v<look>].png for each earlier year in ERAS that has been rendered
# Every run writes the tiles to a fresh folder (tiles-<stamp>/), named in map.json, so browsers never mix cached
# tiles of an older version with a new one.
set -euo pipefail
builtin cd "$(dirname "$0")"
SCALE="${SCALE:-4}"
SITE=site
VIEWS=$(builtin cd scripts && python3 -c "from config import VIEWS; print(*VIEWS)")

stamp=$(date +%Y%m%d-%H%M%S)
rm -rf "$SITE" && mkdir -p "$SITE/tiles-$stamp"

pyramid() {  # image, name under tiles-<stamp>/
  vips dzsave "$1" "$SITE/tiles-$stamp/$2" --tile-size 1024 --overlap 0 --suffix '.webp[Q=90]'
  rm -f "$SITE/tiles-$stamp/$2.dzi" "$SITE/tiles-$stamp/$2_files/vips-properties.xml"
}

ERAS=$(builtin cd scripts && python3 -c "from config import ERAS; print(*ERAS)")
# Today's map, then each earlier year (scripts/config.py ERAS) that has been rendered, under tiles-<stamp>/<era>/.
for era in "" $ERAS; do
  e=$([ -z "$era" ] && echo "" || echo "-$era")
  top=$([ -z "$era" ] && echo "" || echo "$era/")
  for look in $VIEWS; do
    # Looking north-east (45) keeps the names it always had; the other views go into a folder of their own.
    suffix=$([ "$look" = 45 ] && echo "" || echo "-v$look")
    dir="$top$([ -z "$suffix" ] && echo "" || echo "v$look/")"
    full="out/full-$SCALE$e$suffix.png"
    [ -f "$full" ] || { echo "skipping view $look${era:+ of $era}: no $full" >&2; continue; }
    pyramid "$full" "${dir}map"
    # The buildings and trees once more, transparent elsewhere: the viewer lays them over the cars and walkers behind them.
    objects="out/objects-$SCALE$e$suffix.png"
    if [ -f "$objects" ]; then
      vips bandjoin "$full $objects" out/objects.v
      pyramid out/objects.v "${dir}objects"
      rm -f out/objects.v
    fi
  done
  # An earlier year's overlays go next to its tiles' map.json entry, in site/<era>/.
  if [ -n "$era" ] && [ -d "data/build-$era" ] && [ -d "$SITE/tiles-$stamp/$era" ]; then
    mkdir -p "$SITE/$era"
    for f in places hotspots routes history traffic trains; do
      [ -f "data/build-$era/$f.json" ] && cp "data/build-$era/$f.json" "$SITE/$era/"
    done
    [ -d "data/build-$era/history" ] && cp -r "data/build-$era/history" "$SITE/$era/history"
    [ -d "data/build-$era/photos" ] && cp -r "data/build-$era/photos" "$SITE/$era/photos"
  fi
done

python3 - "$SCALE" "$stamp" "$SITE" "$VIEWS" "$ERAS" <<'PY'
import json, os, sys
sys.path.insert(0, "scripts")
from config import ERAS

scale, stamp, site, views, eras = sys.argv[1:]
views, eras = views.split(), eras.split()
m = json.load(open(f"out/tiles-{scale}/layout.json"))


def views_of(era):
    out = []
    for look in views:
        base = f"tiles-{stamp}/" + (f"{era}/" if era else "") + ("" if look == "45" else f"v{look}/")
        if not os.path.isdir(f"{site}/{base}map_files"):
            continue
        view = {"look": int(look), "tiles": f"{base}map_files/"}
        if os.path.isdir(f"{site}/{base}objects_files"):
            view["objects"] = f"{base}objects_files/"
        out.append(view)
    return out


# Today's views stay at the top level too, for viewers that know no eras.
m["views"] = views_of("")
m.update({k: m["views"][0][k] for k in ("tiles", "objects") if k in m["views"][0]})
m["eras"] = [{"key": "now", "label": "Nu", "label_en": "Now", "views": m["views"]}]
for era in eras:
    v = views_of(era)
    if v and os.path.isdir(f"{site}/{era}"):
        label = ERAS[era].get("label", era)
        m["eras"].append({"key": era, "label": label, "label_en": ERAS[era].get("label_en", label), "views": v, "data": f"{era}/"})
json.dump(m, open(f"{site}/map.json", "w"))
PY
cp viewer/index.html viewer/en.json viewer/og.png data/build/places.json data/build/hotspots.json data/build/routes.json data/build/history.json data/build/traffic.json data/build/trains.json "$SITE/"
cp -r data/build/history "$SITE/history"
[ -d data/build/photos ] && cp -r data/build/photos "$SITE/photos"
cp -r viewer/music "$SITE/music"
touch "$SITE/.nojekyll"
echo "$SITE/ is ready: $(find "$SITE" -type f | wc -l) files, $(du -sh "$SITE" | cut -f1)"
