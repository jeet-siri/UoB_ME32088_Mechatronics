"""
Multi-target ArUco shooting WITH the laser servo.
Runs with gantry_home_center_laser_servo.slx on the Pi (homing, 7900-step limit, laser dwell, servo).

Combines:
  - gantry_multi_target_aruco.py (teammate): TargetSelector picks a marker, waits for the Pi to fire,
    marks it shot, sends a short "not visible" gap so the laser turns off, then goes to the next one.
  - gantry_aruco_laser_servo.py: X by the gantry (with the -60 px camera/laser offset), Y by the
    servo from the floor calibration in CV_Templates/servo_calibration.json.

Order: markers in PRIORITY first (in that order), then the rest closest first.
Give every marker a DIFFERENT id - the id is how a shot marker is remembered.

UDP packet = 3 x int16, little-endian: [visible (0/1), error_px, servo angle x 10]
Keys (click the window first): q = quit, r = forget shot markers, a / d = nudge X offset, c = camera
"""

import json
import math
import os
import socket
import struct
import time

import cv2
import numpy as np

# Teammate's target logic and marker detection - stays in Gantry_Test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gantry_Test"))
from gantry_multi_target_aruco import TargetSelector, detect_targets, ON_PX, OFF_PX

PI_IP = "138.38.226.64"
PI_PORT = 50001
SEND_PERIOD = 0.1
CAMERA_INDEX = 0

PRIORITY = []             # e.g. [49, 35, 12]: shoot these first, in this order; [] = closest first

# ---- Laser servo: keep these the same as gantry_aruco_laser_servo.py ----
PIVOT_HEIGHT = 640.0      # mm, floor to the servo shaft
SERVO_ZERO = 90.0         # servo command pointing straight down
SERVO_SIGN = -1
SERVO_DEG_PER_CMD = 0.48  # real beam degrees per commanded degree (measured)
TARGET_U_OFFSET_PX = -60  # where the gantry stops in the image so the laser, not the camera, is over the marker

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cam = np.load(os.path.join(REPO, "CV_Templates", "Calibration.npz"))
CM, DIST = cam["CM"], cam["dist_coef"]
SERVO_CAL_FILE = os.path.join(REPO, "CV_Templates", "servo_calibration.json")


def load_fit():
    """Pixel (v) -> floor Y (mm) fit from servo_calibration.json, or None if not calibrated."""
    if not os.path.exists(SERVO_CAL_FILE):
        return None
    with open(SERVO_CAL_FILE) as f:
        pts = json.load(f)["points"]
    if len(pts) < 3:
        return None
    p = np.array([a for a, _ in pts], float)
    y = np.array([b for _, b in pts], float)
    (a, b, c), *_ = np.linalg.lstsq(np.column_stack([p, np.ones_like(p), -p * y]), y, rcond=None)
    return a, b, c


def servo_for(v_pixel, fit):
    if fit is None:
        return SERVO_ZERO
    a, b, c = fit
    y_mm = (a * v_pixel + b) / (c * v_pixel + 1)
    theta = math.degrees(math.atan2(y_mm, PIVOT_HEIGHT))
    return min(180.0, max(0.0, SERVO_ZERO + SERVO_SIGN * theta / SERVO_DEG_PER_CMD))


def undistort(u, v):
    """Lens-corrected pixel position (kept in pixels), as used for the servo calibration."""
    return cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), CM, DIST, P=CM)[0, 0]


def open_camera(index):
    c = cv2.VideoCapture(index)
    if not c.isOpened():
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Calibration.npz was made at 1280x720
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    got = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (1280, 720):
        print(f"WARNING: camera {index} gives {got[0]}x{got[1]}, not 1280x720 - calibrations will be wrong")
    return c


def draw_text(img, text, pos, scale=0.8, colour=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


def main():
    global CAMERA_INDEX, TARGET_U_OFFSET_PX
    fit = load_fit()
    print("Servo calibrated" if fit else "Servo NOT calibrated - it stays at SERVO_ZERO "
          "(calibrate with gantry_aruco_laser_servo.py, k key)")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(visible, error_px, servo_deg):
        error_px = max(-32768, min(32767, int(error_px)))
        sock.sendto(struct.pack("<hhh", visible, error_px, int(round(servo_deg * 10))), (PI_IP, PI_PORT))

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(f"Could not open camera {CAMERA_INDEX} - try another CAMERA_INDEX")

    selector = TargetSelector(priority=PRIORITY)
    servo = SERVO_ZERO
    last_send = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break
        h, w = frame.shape[:2]
        target_u, cy = w // 2 + TARGET_U_OFFSET_PX, h // 2

        markers = detect_targets(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        raw = {i: c.mean(axis=0) for i, c in markers.items()}
        centres = {i: undistort(*uv) for i, uv in raw.items()}   # lens-corrected, in pixels

        now = time.perf_counter()
        # The selector centres markers on target_u (passed as its "cx") instead of the image centre
        visible, error_px, status = selector.update(centres, target_u, cy, now)
        locked = status["locked"]
        if locked is not None and locked in centres:
            servo = servo_for(float(centres[locked][1]), fit)   # aim Y at the locked marker
        if status["state"] == "SHOT":
            print(f"Shot marker {status['shot']}  (shot so far: {sorted(selector.shot_ids)})")

        if now - last_send >= SEND_PERIOD or status["state"] == "SHOT":
            send(visible, error_px, servo)
            last_send = now

        # ---- drawing ----
        cv2.line(frame, (target_u, 0), (target_u, h), (0, 255, 0), 2)
        for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
            cv2.line(frame, (target_u - d, 0), (target_u - d, h), col, 1)
            cv2.line(frame, (target_u + d, 0), (target_u + d, h), col, 1)
        for i, c in markers.items():
            u, v = (int(x) for x in raw[i])
            if i in selector.shot_ids:
                colour, label = (128, 128, 128), f"{i} SHOT"
            elif i == locked:
                colour, label = (0, 0, 255), f"{i} TARGET"
            else:
                order = f" (P{PRIORITY.index(i) + 1})" if i in PRIORITY else ""
                colour, label = (255, 200, 0), f"{i}{order}"
            cv2.polylines(frame, [c.astype(int)], True, colour, 3)
            cv2.circle(frame, (u, v), 6, colour, -1)
            draw_text(frame, label, (u + 10, v - 10), 0.7, colour)

        state = status["state"]
        if state in ("SETTLING", "LASER ON"):
            line = f"{state} {status['elapsed']:.1f} s  ID {locked}  servo {servo:.1f}"
        elif state == "MOVING":
            line = f"MOVING to ID {locked}  error {error_px:+d} px  servo {servo:.1f}"
        elif state == "DONE":
            line = "DONE - all visible markers shot (r = reset)"
        elif state == "SHOT":
            line = f"SHOT marker {status['shot']}"
        else:
            line = state
        if status["held"]:
            line += "  (blackout, holding)"
        draw_text(frame, line, (10, 35), 0.9, (0, 0, 255) if state == "LASER ON" else (0, 255, 0))
        draw_text(frame, f"shot: {sorted(selector.shot_ids) or '-'}   priority: {PRIORITY or 'closest first'}",
                  (10, 70), 0.7)
        draw_text(frame, f"camera {CAMERA_INDEX} | X offset {TARGET_U_OFFSET_PX:+d} px | "
                         f"servo {'calibrated' if fit else 'NOT calibrated'} | q r a/d c", (10, h - 15), 0.6,
                  (255, 255, 255))
        cv2.imshow("gantry-multi-target-servo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selector.reset_shots()
            print("Shot list cleared")
        if key in (ord("a"), ord("d")):
            TARGET_U_OFFSET_PX += -10 if key == ord("a") else 10
            print(f"TARGET_U_OFFSET_PX = {TARGET_U_OFFSET_PX}")
        if key == ord("c"):
            send(0, 0, servo)
            selector.forget()
            cap.release()
            for step in range(1, 5):
                new_cap = open_camera((CAMERA_INDEX + step) % 4)
                if new_cap is not None:
                    CAMERA_INDEX = (CAMERA_INDEX + step) % 4
                    cap = new_cap
                    break
            print(f"Switched to camera {CAMERA_INDEX}")

    send(0, 0, servo)   # stop the gantry, laser off
    sock.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
