#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ] && [ "$#" -ne 6 ]; then
  echo "Usage: $0 input.osm.pbf output_basename [minlon minlat maxlon maxlat]" >&2
  exit 1
fi

INPUT=$1
OUTPUT=$2
OSM="$OUTPUT.osm"
CSV="$OUTPUT.csv"

# sanity checks (optional)
command -v osmconvert >/dev/null || { echo "osmconvert not found"; exit 1; }
command -v python3     >/dev/null || { echo "python3 not found"; exit 1; }

if [ "$#" -eq 6 ]; then
  MINLON=$3; MINLAT=$4; MAXLON=$5; MAXLAT=$6
  osmconvert "$INPUT" -b=$MINLON,$MINLAT,$MAXLON,$MAXLAT -o="$OSM"
else
  osmconvert "$INPUT" -o="$OSM"
fi

python3 osmtocsv.py "$OSM"
rm -f "$OSM"
echo "✔ Wrote $CSV"
