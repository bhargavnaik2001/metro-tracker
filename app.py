from flask import Flask, request, jsonify
import json
import time
import math

app = Flask(__name__)

# Load your JSON mapping MAC → zone name
with open("beacon_zones.json") as f:
    BEACON_ZONES = json.load(f)

# Known beacon (x,y) positions, matching the same MACs
BEACON_POSITIONS = {
    "6c:c8:40:34:d4:1e": (100, 600),   # Platform A
    "68:25:dd:33:82:6a": (900, 600),   # Platform B
    "68:25:dd:34:11:82": (500, 300),   # Midpoint
}

TX_POWER = -59   # RSSI at 1 m
ENV_FACTOR = 2.0

def rssi_to_distance(rssi: float) -> float:
    """Log-distance path loss model."""
    if rssi == 0:
        return float('inf')
    return 10 ** ((TX_POWER - rssi) / (10 * ENV_FACTOR))

def trilaterate(pos, dist):
    """Solve 3-beacon trilateration analytically."""
    (x1,y1), (x2,y2), (x3,y3) = pos
    r1, r2, r3 = dist
    A = 2*(x2 - x1);  B = 2*(y2 - y1)
    C = r1*r1 - r2*r2 - x1*x1 + x2*x2 - y1*y1 + y2*y2
    D = 2*(x3 - x2);  E = 2*(y3 - y2)
    F = r2*r2 - r3*r3 - x2*x2 + x3*x3 - y2*y2 + y3*y3
    denom = A*E - B*D
    if denom == 0:
        return 0.0, 0.0
    x = (C*E - B*F) / denom
    y = (A*F - C*D) / denom
    return x, y

@app.route("/bledata", methods=["POST"])
def bledata():
    data = request.get_json()
    if not data or "rssi_map" not in data:
        return jsonify({"error": "Missing rssi_map"}), 400

    rssi_map = data["rssi_map"]
    # keep only known MACs
    visible = [(mac, rssi) for mac,rssi in rssi_map.items() if mac in BEACON_POSITIONS]
    if len(visible) < 3:
        return jsonify({"error": "Need 3 beacons"}), 400

    # pick top 3 by RSSI
    visible.sort(key=lambda x: x[1], reverse=True)
    top3 = visible[:3]

    # positions & distances
    positions = [BEACON_POSITIONS[mac] for mac,_ in top3]
    distances = [rssi_to_distance(rssi) for _,rssi in top3]

    # strongest beacon → zone + accuracy
    strongest_mac, strongest_rssi = top3[0]
    zone = BEACON_ZONES.get(strongest_mac, "Unknown")
    accuracy = round(rssi_to_distance(strongest_rssi), 2)

    # trilateration & clamp
    x, y = trilaterate(positions, distances)
    x = max(0, min(1000, x))
    y = max(0, min(1000, y))

    print(f"[{time.strftime('%H:%M:%S')}] → x={x:.1f}, y={y:.1f}, zone={zone}, acc={accuracy}")
    return jsonify({
        "x": x,
        "y": y,
        "zone": zone,
        "accuracy": accuracy
    }), 200



BASE_DIR = os.path.dirname(__file__)
STATIONS_JSON_PATH = os.path.join(BASE_DIR, "stations", "stations.json")


@app.route("/stations", methods=["GET"])
def list_stations():
    if not os.path.exists(STATIONS_JSON_PATH):
        return jsonify({"error": "stations.json not found"}), 404

    try:
        with open(STATIONS_JSON_PATH, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error reading stations.json: {str(e)}"}), 500



if __name__ == "__main__":
    print("Starting BLE-positioning server on :5000")
    app.run(host="0.0.0.0", port=5000)
