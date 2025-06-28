import xml.etree.ElementTree as ET

# === CONFIG ===
INPUT_NET = "osm.net.xml"
OUTPUT_NET = "osm.modified.net.xml"

# Allowed vClasses to assign
new_allow = "emergency passenger delivery motorcycle bicycle"

print("[INFO] Loading network...")
tree = ET.parse(INPUT_NET)
root = tree.getroot()

count_modified = 0

for edge in root.findall("edge"):
    if edge.attrib.get("function") == "internal":
        continue  # skip internal edges

    for lane in edge.findall("lane"):
        allow = lane.attrib.get("allow", "")
        disallow = lane.attrib.get("disallow", "")

        if allow.strip() == "passenger":
            lane.set("allow", new_allow)
            count_modified += 1

print(f"[INFO] Modified {count_modified} lane(s) with allow='passenger'")

tree.write(OUTPUT_NET, encoding="UTF-8")
print(f"[INFO] Saved updated network to: {OUTPUT_NET}")
