from flask import Flask, request, jsonify, send_from_directory, abort
import json
import os
import math
from scipy.optimize import least_squares

app = Flask(__name__)
STATIONS_DIR = "stations"

# Station data structure to match Android app
STATIONS = [
    {
        "id": "1",
        "name": "Andheri Metro",
        "code": "andheri",
        "ip": "192.168.31.54"
    },
    {
        "id": "2",
        "name": "Dadar Metro",
        "code": "dadar",
        "ip": "192.168.31.55"
    }
]


def load_station_config(station_name):
    """Load station config by name (matches Android's stationName)"""
    station_id = next((s["code"]
                      for s in STATIONS if s["name"] == station_name), None)
    if not station_id:
        raise FileNotFoundError(f"Station {station_name} not found")

    path = os.path.join(STATIONS_DIR, station_id, "mapdata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config for station {station_name}")

    with open(path) as f:
        config = json.load(f)
        # Add station ID to match Android expectations
        config["stationId"] = station_id
        return config


@app.route("/stations")
def get_stations():
    """Endpoint for StationSelectActivity to get available stations"""
    return jsonify({"stations": STATIONS})


@app.route("/mapdata/<station_name>")
def get_map_data(station_name):
    """Endpoint for MainActivity to load station metadata"""
    try:
        config = load_station_config(station_name)
        return jsonify({
            "stationId": config.get("stationId"),
            "stationName": station_name,
            "mapWidth": config["mapWidth"],
            "mapHeight": config["mapHeight"],
            "beacons": config["beacons"],
            "safeZones": config.get("safeZones", [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/mapimage/<station_name>")
def get_map_image(station_name):
    """Endpoint for MainActivity to load map image"""
    station_id = next((s["code"]
                      for s in STATIONS if s["name"] == station_name), None)
    if not station_id:
        abort(404)

    image_path = os.path.join(STATIONS_DIR, station_id, "metro_map.png")
    if not os.path.exists(image_path):
        abort(404)

    return send_from_directory(os.path.join(STATIONS_DIR, station_id), "metro_map.png")


def distance_from_rssi(rssi, tx_power=-59, n=2.0):
    """Estimate distance using RSSI (matches Android's calculation)"""
    return 10 ** ((tx_power - rssi) / (10 * n))


def trilaterate(beacons):
    """Perform trilateration (matches Android's coordinate system)"""
    if len(beacons) < 3:
        raise ValueError("At least 3 beacons required")

    def residuals(p):
        x, y = p
        return [
            math.hypot(x - b['x'], y - b['y']) - b['distance']
            for b in beacons
        ]

    # Use average of beacon positions as initial guess
    initial_guess = [
        sum(b['x'] for b in beacons) / len(beacons),
        sum(b['y'] for b in beacons) / len(beacons)
    ]

    result = least_squares(residuals, initial_guess, method='lm')
    x, y = result.x

    # Calculate accuracy in meters (normalized)
    accuracy = sum(abs(r) for r in result.fun) / len(result.fun)

    return x, y, accuracy


@app.route("/position", methods=["POST"])
def calculate_position():
    """Endpoint for MainActivity to get position updates"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        station_name = data.get("stationName")
        beacon_inputs = data.get("beacons", [])

        print(
            f"📡 Position request for {station_name} with {len(beacon_inputs)} beacons")

        if not station_name:
            return jsonify({"error": "stationName required"}), 400

        if len(beacon_inputs) < 3:
            return jsonify({"error": "At least 3 beacons required"}), 400

        # Load station config
        config = load_station_config(station_name)
        known_beacons = {
            b['mac'].lower(): b for b in config.get("beacons", [])}

        # Prepare trilateration input
        trilateration_input = []
        for b in beacon_inputs:
            mac = b.get("mac", "").lower()
            rssi = b.get("rssi", -70)
            if mac in known_beacons:
                beacon = known_beacons[mac]
                trilateration_input.append({
                    "x": beacon["x"],
                    "y": beacon["y"],
                    "distance": distance_from_rssi(rssi)
                })

        if len(trilateration_input) < 3:
            return jsonify({"error": "Couldn't match enough known beacons"}), 400

        # Calculate position
        x, y, accuracy = trilaterate(trilateration_input)

        # Find nearest zone (matches Android's zoneText)
        nearest_zone = min(
            config.get("safeZones", []),
            key=lambda z: math.hypot(
                x - (z['x'] + z['width']/2),
                y - (z['y'] + z['height']/2)
            )
        )['name'] if config.get("safeZones") else "Unknown"

        print(f"📍 Calculated position: ({x:.2f}, {y:.2f}) near {nearest_zone}")

        return jsonify({
            "x": x,
            "y": y,
            "zone": nearest_zone,
            "accuracy": accuracy
        })

    except Exception as e:
        print(f"❌ Error in /position: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create stations directory if not exists
    if not os.path.exists(STATIONS_DIR):
        os.makedirs(STATIONS_DIR)

    # Create sample station directories
    for station in STATIONS:
        station_dir = os.path.join(STATIONS_DIR, station["code"])
        if not os.path.exists(station_dir):
            os.makedirs(station_dir)

    app.run(host="0.0.0.0", port=5000, debug=True)
