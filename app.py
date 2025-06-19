from flask import Flask, request, jsonify, send_from_directory, abort
import json
import os
import math
import numpy as np
from collections import deque
import threading

app = Flask(__name__)
STATIONS_DIR = "stations"

# Station data structure
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

# Global dictionary to store user position history
user_history = {}
history_lock = threading.Lock()


class SimpleKalmanFilter:
    def __init__(self, process_variance=0.01, measurement_variance=1, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_value
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        kalman_gain = priori_error_estimate / \
            (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + \
            kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        return self.posteri_estimate


def load_station_config(station_name):
    """Load station config by name"""
    station_id = next((s["code"]
                      for s in STATIONS if s["name"] == station_name), None)
    if not station_id:
        raise FileNotFoundError(f"Station {station_name} not found")

    path = os.path.join(STATIONS_DIR, station_id, "mapdata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config for station {station_name}")

    with open(path) as f:
        config = json.load(f)
        config["stationId"] = station_id
        return config


def is_point_in_zone(x, y, zone):
    """Check if point is within a safe zone"""
    return (zone['x'] <= x <= zone['x'] + zone['width'] and
            zone['y'] <= y <= zone['y'] + zone['height'])


def clamp_to_nearest_zone(x, y, zones):
    """Adjust position to the closest point in the nearest safe zone"""
    min_dist = float('inf')
    clamped_point = (x, y)
    nearest_zone = None

    for zone in zones:
        # Calculate closest point in the rectangle
        closest_x = max(zone['x'], min(x, zone['x'] + zone['width']))
        closest_y = max(zone['y'], min(y, zone['y'] + zone['height']))
        dist = math.hypot(x - closest_x, y - closest_y)

        if dist < min_dist:
            min_dist = dist
            clamped_point = (closest_x, closest_y)
            nearest_zone = zone

    return clamped_point[0], clamped_point[1], nearest_zone


def distance_from_rssi(rssi, tx_power=-59, n=2.0, max_distance=None):
    """Estimate distance using RSSI with configurable parameters"""
    # Calculate distance using log-distance path loss model
    distance = 10 ** ((tx_power - rssi) / (10 * n))

    # Apply reasonable constraints
    if max_distance:
        return min(distance, max_distance)
    return distance


def trilaterate(beacons, map_width, map_height):
    """Perform trilateration with boundary constraints using pure Python"""
    if len(beacons) < 3:
        raise ValueError("At least 3 beacons required")

    # Weighted average based on signal strength
    weights = [1 / (b['distance'] + 0.1) for b in beacons]
    total_weight = sum(weights)

    weighted_x = sum(b['x'] * w for b,
                     w in zip(beacons, weights)) / total_weight
    weighted_y = sum(b['y'] * w for b,
                     w in zip(beacons, weights)) / total_weight

    # Clamp to map boundaries
    x = max(0, min(weighted_x, map_width))
    y = max(0, min(weighted_y, map_height))

    # Calculate accuracy
    errors = []
    for b in beacons:
        dist = math.hypot(x - b['x'], y - b['y'])
        errors.append(abs(dist - b['distance']))

    accuracy = sum(errors) / len(errors)
    normalized_accuracy = accuracy / math.hypot(map_width, map_height)

    return x, y, normalized_accuracy


def find_nearest_beacon(x, y, beacons):
    """Find the closest beacon to a position"""
    min_dist = float('inf')
    nearest_beacon = None

    for beacon in beacons:
        dist = math.hypot(x - beacon['x'], y - beacon['y'])
        if dist < min_dist:
            min_dist = dist
            nearest_beacon = beacon

    return nearest_beacon, min_dist


def smooth_position(user_id, new_x, new_y, history_size=5):
    """Apply smoothing to position using historical data"""
    with history_lock:
        # Initialize history for new users
        if user_id not in user_history:
            user_history[user_id] = {
                'x': deque([new_x] * history_size, maxlen=history_size),
                'y': deque([new_y] * history_size, maxlen=history_size),
                'kalman_x': SimpleKalmanFilter(initial_value=new_x),
                'kalman_y': SimpleKalmanFilter(initial_value=new_y)
            }

        # Get user history
        history = user_history[user_id]

        # Add new position to history
        history['x'].append(new_x)
        history['y'].append(new_y)

        # Apply moving average
        avg_x = sum(history['x']) / len(history['x'])
        avg_y = sum(history['y']) / len(history['y'])

        # Apply Kalman filtering
        kalman_x = history['kalman_x'].update(avg_x)
        kalman_y = history['kalman_y'].update(avg_y)

        return kalman_x, kalman_y


@app.route("/")
def hello():
    return jsonify({
        "message": "MetroTracker Indoor Positioning System",
        "status": "operational",
        "version": "1.3"  # Updated version
    })


@app.route("/stations")
def get_stations():
    return jsonify({"stations": STATIONS})


@app.route("/mapdata/<station_name>")
def get_map_data(station_name):
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
    station_id = next((s["code"]
                      for s in STATIONS if s["name"] == station_name), None)
    if not station_id:
        abort(404)

    image_path = os.path.join(STATIONS_DIR, station_id, "metro_map.png")
    if not os.path.exists(image_path):
        abort(404)

    return send_from_directory(os.path.join(STATIONS_DIR, station_id), "metro_map.png")


@app.route("/position", methods=["POST"])
def calculate_position():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        station_name = data.get("stationName")
        beacon_inputs = data.get("beacons", [])
        user_id = data.get("userId")  # Get user ID from request

        if not station_name:
            return jsonify({"error": "stationName required"}), 400
        if len(beacon_inputs) < 3:
            return jsonify({"error": "At least 3 beacons required"}), 400
        if not user_id:
            return jsonify({"error": "userId required"}), 400

        # Load station config
        config = load_station_config(station_name)
        known_beacons = {
            b['mac'].lower(): b for b in config.get("beacons", [])}
        safe_zones = config.get("safeZones", [])
        map_width = config["mapWidth"]
        map_height = config["mapHeight"]
        max_distance = math.hypot(map_width, map_height)

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
                    "distance": distance_from_rssi(rssi, max_distance=max_distance)
                })

        if len(trilateration_input) < 3:
            return jsonify({"error": "Couldn't match enough known beacons"}), 400

        # Calculate initial position
        x, y, accuracy = trilaterate(
            trilateration_input, map_width, map_height)

        # Apply smoothing and stabilization
        smooth_x, smooth_y = smooth_position(user_id, x, y)

        # Use smoothed position for further processing
        x, y = smooth_x, smooth_y

        # Apply safe zone constraints
        in_zone = any(is_point_in_zone(x, y, zone) for zone in safe_zones)
        if not in_zone:
            x, y, nearest_zone = clamp_to_nearest_zone(x, y, safe_zones)
        else:
            # Find which zone we're in
            nearest_zone = next((zone for zone in safe_zones
                                if is_point_in_zone(x, y, zone)),
                                {"name": "Unknown"})

        # Find nearest beacon
        nearest_beacon, beacon_dist = find_nearest_beacon(
            x, y, config["beacons"])
        beacon_name = nearest_beacon.get(
            "id", "Unknown") if nearest_beacon else "None"

        # Convert accuracy to meters (assume map is 1:100 scale)
        accuracy_meters = accuracy * 100

        return jsonify({
            "x": x,
            "y": y,
            "zone": nearest_zone.get("name", "Unknown"),
            "accuracy": accuracy_meters,
            "nearest_beacon": beacon_name,
            "beacon_distance": beacon_dist
        })

    except Exception as e:
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
