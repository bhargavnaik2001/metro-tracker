from flask import Flask, request, jsonify, send_from_directory, abort
import json
import os
import math
from scipy.optimize import least_squares

app = Flask(__name__)
STATIONS_DIR = "stations"


def load_station_config(station_id):
    path = os.path.join(STATIONS_DIR, station_id, "mapdata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config for station {station_id}")
    with open(path) as f:
        return json.load(f)


@app.route("/mapdata/<station_id>")
def mapdata(station_id):
    try:
        config = load_station_config(station_id)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/mapimage/<station_id>")
def mapimage(station_id):
    image_path = os.path.join(STATIONS_DIR, station_id, "metro_map.png")
    if not os.path.exists(image_path):
        abort(404)
    return send_from_directory(os.path.join(STATIONS_DIR, station_id), "metro_map.png")


@app.route("/stations")
def stations():
    entries = []
    for name in os.listdir(STATIONS_DIR):
        if os.path.isdir(os.path.join(STATIONS_DIR, name)):
            entries.append({
                "id": name,
                "name": name.capitalize().replace("_", " "),
                "baseUrl": request.host_url.rstrip("/")
            })
    return jsonify(entries)


def distance_from_rssi(rssi, tx_power=-59, n=2.0):
    """Estimate distance using RSSI."""
    return 10 ** ((tx_power - rssi) / (10 * n))


def trilaterate(beacons):
    """Perform trilateration based on beacon positions and distances."""
    if len(beacons) < 3:
        raise ValueError("At least 3 beacons required")

    def residuals(p):
        x, y = p
        return [
            math.hypot(x - b['x'], y - b['y']) - b['distance']
            for b in beacons
        ]

    initial_guess = [
        sum(b['x'] for b in beacons) / len(beacons),
        sum(b['y'] for b in beacons) / len(beacons)
    ]

    result = least_squares(residuals, initial_guess, method='lm')
    x, y = result.x
    error = sum(abs(r) for r in result.fun) / len(result.fun)
    return x, y, error


@app.route("/position", methods=["POST"])
def position():
    try:
        data = request.get_json(force=True)
        beacon_inputs = data.get("beacons", [])
        station_id = data.get("stationId", "andheri").lower()

        print(f"📡 Received beacon data for station: {station_id}")
        print(f"📶 Beacon inputs: {beacon_inputs}")

        if len(beacon_inputs) < 3:
            return jsonify({"error": "At least 3 beacons required"}), 400

        config = load_station_config(station_id)
        known_beacons = {
            b['mac'].lower(): b for b in config.get("beacons", [])}

        trilateration_input = []
        for b in beacon_inputs:
            mac = b.get("mac", "").lower()
            rssi = b.get("rssi", -70)
            if mac in known_beacons:
                kb = known_beacons[mac]
                trilateration_input.append({
                    "x": kb["x"],
                    "y": kb["y"],
                    "distance": distance_from_rssi(rssi)
                })

        if len(trilateration_input) < 3:
            return jsonify({"error": "Matching beacons < 3"}), 400

        x, y, accuracy = trilaterate(trilateration_input)

        nearest = min(
            config.get("beacons", []),
            key=lambda b: math.hypot(x - b["x"], y - b["y"])
        )["id"]

        print(
            f"✅ Estimated position: x={x:.2f}, y={y:.2f}, nearest={nearest}, accuracy={accuracy:.2f}")

        return jsonify({
            "x": x,
            "y": y,
            "accuracy": accuracy,
            "nearest": nearest
        })

    except Exception as e:
        print(f"❌ Error in /position: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
