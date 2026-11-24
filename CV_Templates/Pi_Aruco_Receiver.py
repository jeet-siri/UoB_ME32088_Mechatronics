#!/usr/bin/env python3
"""
Tutorial 5 - step 11: RUN THIS ON THE RASPBERRY PI.

Receives the ArUco poses sent over WiFi (UDP) by CV_Python_Aruco_steps.py on the
laptop and uses them to control an LED:

  * LED ON  when marker TARGET_ID is closer than DISTANCE_THRESHOLD mm
  * LED OFF when it is further away, when a different/no marker is seen,
            or when nothing has been received for TIMEOUT seconds (safety)

Message format sent by the laptop (text):
  "NONE"                                  no marker visible
  "3,-45.2,12.8,412.6;7,30.1,-5.0,380.2"  one "id,x,y,z" group (mm) per marker

Based on Network_Templates/UDP_Receive.py. Replace the LED with a motor or
anything else by changing act_on_markers() below.
Stop with Ctrl+C.
"""

import socket
import logging
from gpiozero import LED

logging.basicConfig(level=logging.INFO)

# ---- Settings -------------------------------------------------------------
UDP_IP = "0.0.0.0"          # listen on all network interfaces (WiFi included)
UDP_PORT = 50002            # must match PI_PORT in the laptop script
LED_PIN = 17                # BCM GPIO number your LED is wired to: CHANGE to match your circuit
TARGET_ID = 3               # the marker id that controls the LED
DISTANCE_THRESHOLD = 300    # mm: LED on when the marker is closer than this
TIMEOUT = 1.0               # s: switch the LED off if the laptop stops sending
# ---------------------------------------------------------------------------

led = LED(LED_PIN)


def parse_message(text):
    """Turn "id,x,y,z;id,x,y,z" into {id: (x, y, z)}. "NONE" gives {}."""
    markers = {}
    if text == "NONE":
        return markers
    for group in text.split(";"):
        try:
            marker_id, x, y, z = group.split(",")
            markers[int(marker_id)] = (float(x), float(y), float(z))
        except ValueError:
            logging.warning("Ignoring badly formatted data: %r", group)
    return markers


def act_on_markers(markers):
    """Decide what to do with the received markers. Change this for your own task."""
    if TARGET_ID in markers:
        x, y, z = markers[TARGET_ID]
        if z < DISTANCE_THRESHOLD:
            led.on()
        else:
            led.off()
    else:
        led.off()


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(TIMEOUT)
logging.info("Listening on port %s, LED on GPIO%s, target marker %s, threshold %s mm",
             UDP_PORT, LED_PIN, TARGET_ID, DISTANCE_THRESHOLD)

last_state = None
try:
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            # Nothing received recently: the laptop script stopped or WiFi dropped
            led.off()
            if last_state != "timeout":
                logging.warning("No data for %.1f s - LED off", TIMEOUT)
                last_state = "timeout"
            continue

        markers = parse_message(data.decode("utf-8").strip())
        act_on_markers(markers)

        # Log only when something changes, so the terminal stays readable
        state = ("LED on" if led.is_lit else "LED off", tuple(sorted(markers)))
        if state != last_state:
            logging.info("From %s: markers %s -> %s", addr[0], list(markers) or "none", state[0])
            last_state = state
except KeyboardInterrupt:
    pass
finally:
    led.off()
    sock.close()
    logging.info("Receiver stopped")
