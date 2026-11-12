#!/usr/bin/env bash
# Build the isometric map of Ede from open data, end to end.
#   ./build.sh            fetch, preprocess, render at 4 px/m, cut Deep Zoom tiles into site/
#   SCALE=3 ./build.sh    render at another resolution (update SCALE/WIDTH/HEIGHT in viewer/index.html to match)
#   ERAS= ./build.sh      only today's map, none of the earlier years in scripts/config.py ERAS
set -euo pipefail
builtin cd "$(dirname "$0")"
SCALE="${SCALE:-4}"

prepare() {  # every preprocessing step, into data/build or, with ERA set, data/build-<era>
  (builtin cd scripts && uv run --quiet landmarks.py)
  (builtin cd scripts && uv run --quiet build_buildings.py)
  (builtin cd scripts && uv run --quiet build_trees.py 2>&1 | grep -v GDAL_NODATA)
  (builtin cd scripts && uv run --quiet build_ground.py)
  (builtin cd scripts && python3 build_places.py)
  (builtin cd scripts && uv run --quiet --with numpy python build_hotspots.py)
  (builtin cd scripts && python3 build_routes.py)
  (builtin cd scripts && uv run --quiet build_traffic.py)
  (builtin cd scripts && uv run --quiet build_trains.py)
  (builtin cd scripts && python3 build_history.py)
}

VIEWS=$(builtin cd scripts && python3 -c "from config import VIEWS; print(*VIEWS)")
join() {  # tile folder, output image, bands to keep
  read -r W H C < <(python3 -c "import json; d=json.load(open('$1/layout.json')); print(d['width'], d['height'], d['cols'])")
  vips arrayjoin "$(/bin/ls "$1"/*.png | tr '\n' ' ')" out/joined.v --across "$C"
  vips extract_area out/joined.v out/cropped.v 0 0 "$W" "$H"
  vips extract_band out/cropped.v "$2[compression=1]" 0 --n "$3"
  rm -f out/joined.v out/cropped.v
}
# Every view in scripts/config.py, each with the mask of buildings and trees that the viewer draws again over the
# cars and walkers behind them, joined into out/full-<scale>[-<era>][-v<look>].png and out/objects-<...>.png.
render() {  # an era, or nothing for today
  local era="$1" e="" look suffix
  [ -n "$era" ] && e="-$era"
  for look in $VIEWS; do
    node render/run.mjs all --scale "$SCALE" --tile 2048 --kuwahara 2 --look "$look" ${era:+--era "$era"}
    node render/run.mjs all --scale "$SCALE" --tile 2048 --pass objects --look "$look" ${era:+--era "$era"}
  done
  for look in $VIEWS; do
    suffix=$([ "$look" = 45 ] && echo "" || echo "-v$look")
    join "out/tiles-$SCALE$e$suffix" "out/full-$SCALE$e$suffix.png" 3
    join "out/tiles-$SCALE-objects$e$suffix" "out/objects-$SCALE$e$suffix.png" 1
  done
}

(builtin cd scripts && python3 fetch.py)
prepare
[ -d node_modules ] || bun install
render ""

# The earlier years: the same steps with ERA set, on AHN2 cut into chunks once (scripts/ahn2.py).
ERAS="${ERAS-$(builtin cd scripts && python3 -c "from config import ERAS; print(*ERAS)")}"
if [ -n "$ERAS" ]; then
  [ -d data/raw/ahn2/chunks ] || (builtin cd scripts && uv run --quiet ahn2.py 2>&1 | grep -v GDAL_NODATA)
  # The roads, railway and buildings of OpenStreetMap as they were in 2012, and the Kadaster's old topographic maps.
  (builtin cd scripts && python3 -c "import fetch; fetch.fetch_attic(); fetch.fetch_attic_buildings()")
  for year in $(builtin cd scripts && python3 -c "from config import ERAS; print(*sorted({str(v) for e in ERAS.values() for k, v in e.items() if k in ('topo', 'topo_blind')}))"); do
    [ -f "data/raw/topo$year/mosaic.png" ] || (builtin cd scripts && uv run --quiet topo_fetch.py "$year")
  done
  for era in $ERAS; do
    export ERA="$era"
    prepare
    unset ERA
    render "$era"
  done
fi

./site.sh
