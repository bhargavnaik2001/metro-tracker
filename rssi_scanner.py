#!/usr/bin/env python3
import json
import time
import requests
import logging
from bluepy.btle import Scanner, DefaultDelegate, BTLEException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Known beacons with fixed MAC addresses and zone names
BEACONS = {
    "6c:c8:40:34:d4:1e": "Platform A",
    "68:25:dd:33:82:6a": "Platform B",
    "68:25:dd:34:11:82": "Midpoint"
}

# Raspberry Pi Flask server endpoint
SERVER_URL = "http://192.168.31.54:5000/bledata"

# BLE Scan Delegate
class ScanDelegate(DefaultDelegate):
    def __init__(self):
        super().__init__()

def scan_and_send(timeout: float = 2.0):
    scanner = Scanner().withDelegate(ScanDelegate())
    try:
        log.info(f"Scanning for BLE devices for {timeout} seconds...")
        devices = scanner.scan(timeout)
    except BTLEException as e:
        log.error(f"BLE scan failed: {e}")
        return

    found = 0
    for dev in devices:
        mac = dev.addr.lower()
        if mac in BEACONS:
            zone = BEACONS[mac]
            rssi = dev.rssi
            payload = {
                "device": zone,
                "mac": mac,
                "rssi": rssi
            }
            try:
                res = requests.post(SERVER_URL, json=payload, timeout=2)
                if res.status_code == 200:
                    log.info(f"✔ Sent {zone} ({mac}) → RSSI {rssi} dB → Response: {res.text}")
                else:
                    log.warning(f"⚠ Failed to send data for {zone}. Status: {res.status_code}")
            except requests.RequestException as e:
                log.error(f"❌ Network error while sending data for {zone}: {e}")
            found += 1

    if found == 0:
        log.warning("⚠ No known beacons detected.")

def main():
    try:
        while True:
            scan_and_send(timeout=2.0)
            time.sleep(0.5)  # Short delay before next scan
    except KeyboardInterrupt:
        log.info("🛑 Scan stopped by user.")

if __name__ == "__main__":
    main()
