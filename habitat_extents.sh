#!/usr/bin/env bash
#
# fetch_habitat_extents.sh
#
# Downloads global extent layers for three shallow-water benthic habitats,
# to use as an AOI-selection / stratification layer for a marine MISR dataset:
#
#   coral     UNEP-WCMC Global Distribution of Coral Reefs, v4.1 (WCMC-008)
#             polygons, warm-water reefs.               DOI 10.34892/t2wk-5t34
#   seagrass  UNEP-WCMC Global Distribution of Seagrasses, v7.1 (WCMC-013/014)
#             point + polygon occurrence.               DOI 10.34892/x6r3-d211
#   mangrove  Global Mangrove Watch v4.0.19 (annual extent, 1990-2024)
#             polygons, from Zenodo.                    DOI 10.5281/zenodo.12756047
#
# All three are CC BY 4.0. Cite them if you publish.
#
# Requires: curl, unzip, and python3 (used only to parse the Zenodo file list).
# Runs on Pop!_OS / any Linux and under git bash on Windows.
#
# Note: the download endpoints are external (wcmc.io, zenodo.org) so this can't
# be dry-run in a sandbox - run it on your own machine.

# Require bash — pipefail and mapfile are bash-only.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "Error: run this script with bash, not sh:" >&2
  echo "  bash $0  or  ./$0" >&2
  exit 1
fi

set -euo pipefail

# -------- config -------------------------------------------------------------
OUT="${1:-./data/habitat_extents}"  # output dir, override as first arg
GMW_YEAR="2020"                   # which GMW epoch to keep; "" = all epochs
# -----------------------------------------------------------------------------

CORAL_URL="https://wcmc.io/WCMC_008"
SEAGRASS_URL="https://wcmc.io/WCMC_013_014"
ZENODO_RECORD="12756047"

mkdir -p "$OUT"/{coral,seagrass,mangrove}

fetch_zip () {
  # $1 = url, $2 = target dir, $3 = label
  local url="$1" dir="$2" label="$3" zip="$2/${3}.zip"
  echo ">> $label"
  # -L follows redirects (wcmc.io is a shortlink), -f fails on HTTP errors
  curl -fL --retry 3 -o "$zip" "$url"
  unzip -o -q "$zip" -d "$dir"
  rm -f "$zip"
  echo "   unpacked to $dir"
}

# ---- coral ------------------------------------------------------------------
fetch_zip "$CORAL_URL" "$OUT/coral" "coral_reefs_v4_1"

# ---- seagrass ---------------------------------------------------------------
fetch_zip "$SEAGRASS_URL" "$OUT/seagrass" "seagrasses_v7_1"

# ---- mangrove (Zenodo) ------------------------------------------------------
echo ">> mangrove (GMW v4, Zenodo record $ZENODO_RECORD)"
API="https://zenodo.org/api/records/${ZENODO_RECORD}"

# Pull the file list from the Zenodo API and filter.
# If GMW_YEAR is set, keep only files whose name contains that year;
# otherwise keep everything. Falls back to all files if the filter matches none.
mapfile -t URLS < <(
  curl -fsSL "$API" | python3 -c "
import json, sys
year = sys.argv[1]
rec  = json.load(sys.stdin)
files = rec.get('files', [])
def url(f):
    return f.get('links', {}).get('self') or f.get('links', {}).get('download')
sel = [f for f in files if year and year in f.get('key', '')]
if not sel:
    sel = files
for f in sel:
    u = url(f)
    if u:
        print(u)
" "$GMW_YEAR"
)

if [ "${#URLS[@]}" -eq 0 ]; then
  echo "   !! no files returned by Zenodo API - check the record manually:"
  echo "      https://zenodo.org/records/${ZENODO_RECORD}"
else
  for u in "${URLS[@]}"; do
    echo "   fetching $(basename "$u")"
    curl -fL --retry 3 -J -O --output-dir "$OUT/mangrove" "$u"
  done
  # unzip any zipped epochs that came down
  find "$OUT/mangrove" -name '*.zip' -exec unzip -o -q {} -d "$OUT/mangrove" \; -exec rm -f {} \;
fi

echo
echo "Done. Layers under: $OUT"
find "$OUT" -maxdepth 2 -type f \( -name '*.shp' -o -name '*.gpkg' \) | sort