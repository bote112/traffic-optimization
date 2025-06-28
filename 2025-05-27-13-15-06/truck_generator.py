import xml.etree.ElementTree as ET
import random
import subprocess

# === CONFIG ===
INPUT_NET_FILE = "osm.net.xml"
OUTPUT_TRIP_FILE = "trips.truck.xml"
ROUTE_FILE = "truck.rou.xml"
TRIP_DURATION = 1200
NUM_TRUCKS = random.randint(5, 50)

TRUCK_ORIGIN_LANE = [
    "48788485#0_1",
    "48788485#0_2",
    "401741890_0",
    "223942064#0_0",
    "-160010389#2_0",
    "33913538#0_1",
    "33913538#0_2"
]

TRUCK_FINISH_LANE = [
    "48778431#10_2",
    "48778431#10_1",
    "647606778_0",
    "-223942064#0.371_0",
    "160010389#2_0",
    "-33913538#0_2",
    "-33913538#0_1"
]

TRUCK_VTYPE = {
    "id": "truck",
    "vClass": "truck",
    "accel": "1.0",
    "decel": "4.5",
    "length": "12.0",
    "maxSpeed": "13.9"
}

# === Load network and get all truck-legal lane IDs ===
tree = ET.parse(INPUT_NET_FILE)
root = tree.getroot()

truck_legal_lanes = []
for edge in root.findall("edge"):
    if "function" in edge.attrib:
        continue
    for lane in edge.findall("lane"):
        allow = lane.get("allow", "")
        disallow = lane.get("disallow", "")
        if "rail" in allow or "rail" in disallow:
            continue
        if allow == "" or "truck" in allow.split():
            truck_legal_lanes.append(lane.attrib["id"])

# === Generate and sort trips ===
trips_root = ET.Element("routes")
ET.SubElement(trips_root, "vType", TRUCK_VTYPE)

truck_trips = []
for i in range(NUM_TRUCKS):
    origin = random.choice(TRUCK_ORIGIN_LANE)
    if random.random() < 0.8:
        destination = random.choice(TRUCK_FINISH_LANE)
    else:
        destination = random.choice([l for l in truck_legal_lanes if l != origin])

    depart_time = random.randint(0, TRIP_DURATION - 1)
    truck_trips.append({
        "id": f"truck{i}",
        "type": "truck",
        "depart": depart_time,
        "from": origin.rsplit("_", 1)[0],
        "to": destination.rsplit("_", 1)[0]
    })

# Sort by departure time
truck_trips.sort(key=lambda x: x["depart"])

# Add to XML
for trip in truck_trips:
    ET.SubElement(trips_root, "trip", {
        "id": trip["id"],
        "type": trip["type"],
        "depart": str(trip["depart"]),
        "from": trip["from"],
        "to": trip["to"]
    })

# === Save trips file ===
tree_out = ET.ElementTree(trips_root)
tree_out.write(OUTPUT_TRIP_FILE, encoding="UTF-8", xml_declaration=True)

print(f"[INFO] Created {NUM_TRUCKS} sorted truck trips.")
print(f"[INFO] Trip file: {OUTPUT_TRIP_FILE}")

# === Optional: Route them immediately ===
duarouter_cmd = [
    "duarouter",
    "-n", INPUT_NET_FILE,
    "--trip-files", OUTPUT_TRIP_FILE,
    "-a", "vtypes.xml",
    "-o", ROUTE_FILE,
    "--ignore-errors"
]

print(f"[INFO] Routing with duarouter...")
result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
if result.returncode == 0:
    print(f"[INFO] Routing complete. Output: {ROUTE_FILE}")
else:
    print("[ERROR] Routing failed:")
    print(result.stderr)
