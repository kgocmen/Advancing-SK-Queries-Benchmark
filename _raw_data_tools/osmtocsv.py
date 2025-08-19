import csv
import xml.etree.ElementTree as ET
import argparse
import os
import json

parser = argparse.ArgumentParser(description="Stream and extract tagged OSM nodes to CSV")
parser.add_argument("input_file", help="Input .osm XML file")
args = parser.parse_args()

# Output directory: ../data relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "data")
os.makedirs(output_dir, exist_ok=True)

# Output filename: <basename>.csv under ../data
base = os.path.splitext(os.path.basename(args.input_file))[0]
output_file = os.path.join(output_dir, base + ".csv")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "lat", "lon", "tags"])
    for event, elem in ET.iterparse(args.input_file, events=("end",)):
        if elem.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in elem.findall("tag")}
            if tags:
                writer.writerow([
                    elem.get("id"),
                    elem.get("lat"),
                    elem.get("lon"),
                    json.dumps(tags, ensure_ascii=False)
                ])
            elem.clear()
