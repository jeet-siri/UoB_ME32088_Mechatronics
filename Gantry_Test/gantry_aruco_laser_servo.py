"""
Aim the laser at the centre of an ArUco marker: the gantry moves along X, the servo tilts along Y.
Runs with gantry_home_center_laser_servo.slx on the Pi.

  X (along the gantry):  the Pi centres the marker horizontally in the image (bang-bang, as before)
  Y (across the gantry): the marker's vertical pixel position -> floor distance Y (mm) -> servo angle
                         servo = SERVO_ZERO + SERVO_SIGN * atan2(Y, PIVOT_HEIGHT)

The pixel -> Y map uses the same fit and file as CV_Templates/Servo_Calibration.py:
  Y = (a*p + b) / (c*p + 1)   saved in CV_Templates/servo_calibration.json
It is exact for a flat floor seen by a fixed, tilted camera, so the camera's height/angle never
need measuring. It needs at least 3 calibration points (5 is better), recorded with the k key:

  1. Find SERVO_ZERO (servo command that points the laser straight down) with
     Servo_Test/servo_laser_test.slx, and put it below. Mark Y = 0 on the floor under the laser dot.
  2. Mark more points on a line ACROSS the gantry, e.g. -200, -100, 100, 200 mm (decide which side
     is + and stick to it).
  3. Put the marker's centre on a mark, let the gantry centre it, press k in the window and type
     that mark's Y (mm) in the terminal. The gantry stops while you type (link timeout) - that's fine.
  4. Repeat for every mark. The fit error per point is printed: a few mm is good.
  5. Check: the laser should now land on the marker. If it tilts the wrong way, flip SERVO_SIGN.

UDP packet = 3 x int16, little-endian: [visible (0/1), error_px, servo angle x 10]
Keys: k = add calibration point, c = switch camera, q = quit
"""

import json
import math
import os
import socket
import struct
import time

import cv2
import cv2.aruco as aruco
import numpy as np

PI_IP = "138.38.226.64"  # Pi's IP (on the Pi: hostname -I)
PI_PORT = 50001
TARGET_ID = None          # e.g. 35 to follow only that marker; None = the first marker seen
SEND_PERIOD = 0.1
ON_PX, OFF_PX = 40, 15    # only for drawing - the real thresholds are in the Simulink model
CAMERA_INDEX = 0          # USB camera (press c to switch)
HOLD_TIME = 0.4           # blackout tolerance (s)
CONFIRM_FRAMES = 3        # a new marker must be seen this many frames in a row

# ---- Laser servo: MEASURE / SET THESE ----
PIVOT_HEIGHT = 640.0      # mm, floor to the servo shaft centre
SERVO_ZERO = 90.0         # servo command (deg) that points the laser straight down
SERVO_SIGN = -1           # +1 or -1: flip if the laser tilts the wrong way (flipped after calibrating)
# Real beam degrees per commanded degree. Measured: commands 60/120 put the dot 165 mm out
# (atan(165/640) = 14.5 deg for 30 commanded) and 105 put it 81 mm out (7.2 deg for 15) -> 0.48.
# The servo block's default 1000-2000 us pulses only swing this servo ~87 deg over 0-180.
# With 0.48 the reachable beam angle is +-43 deg -> Y up to about +-600 mm.
SERVO_DEG_PER_CMD = 0.48
# If the laser lands beside the marker ALONG the gantry (camera and laser not lined up),
# shift where the gantry centres the marker: + moves the stop point right in the image.
TARGET_U_OFFSET_PX = -60  # measured: laser lands ~5 cm along the gantry from the camera's centre

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_FILE = os.path.join(REPO, "CV_Templates", "Calibration.npz")        # camera
SERVO_CAL_FILE = os.path.join(REPO, "CV_Templates", "servo_calibration.json")  # pixel -> Y
Y_PIXEL_AXIS = "v"        # up/down in the image runs across the gantry (same as Servo_Calibration.py)

cam = np.load(CALIBRATION_FILE)
CM, DIST = cam["CM"], cam["dist_coef"]


# ---------------- calibration maths (same as Servo_Calibration.py) ----------------
def fit_pixel_to_y(points):
    p = np.array([pt[0] for pt in points], dtype=float)
    y = np.array([pt[1] for pt in points], dtype=float)
    A = np.column_stack([p, np.ones_like(p), -p * y])
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b, c


def pixel_to_y(p, coeffs):
    a, b, c = coeffs
    return (a * p + b) / (c * p + 1)


def y_to_servo(y_mm):
    theta = math.degrees(math.atan2(y_mm, PIVOT_HEIGHT))   # real beam angle, 0 = straight down
    return min(180.0, max(0.0, SERVO_ZERO + SERVO_SIGN * theta / SERVO_DEG_PER_CMD))


def load_cal():
    if not os.path.exists(SERVO_CAL_FILE):
        return []
    with open(SERVO_CAL_FILE) as f:
        data = json.load(f)
    if data.get("axis") != Y_PIXEL_AXIS:
        print(f"servo_calibration.json uses axis {data.get('axis')} - ignoring it")
        return []
    return [tuple(pt) for pt in data["points"]]


def save_cal(points):
    with open(SERVO_CAL_FILE, "w") as f:
        json.dump({"axis": Y_PIXEL_AXIS, "points": points}, f, indent=2)


def report_fit(points, coeffs):
    print("Calibration fit (pixel -> Y mm):")
    for p, y in points:
        print(f"  pixel {p:7.1f}  measured {y:7.1f}  fitted {pixel_to_y(p, coeffs):7.1f}  "
              f"error {pixel_to_y(p, coeffs) - y:+6.1f} mm")


points = load_cal()
coeffs = fit_pixel_to_y(points) if len(points) >= 3 else None
if coeffs is not None:
    report_fit(points, coeffs)
else:
    print(f"Servo NOT calibrated ({len(points)}/3 points): it will stay at SERVO_ZERO. Press k to add points.")

# ---------------- UDP ----------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(visible, error_px, servo_deg):
    error_px = max(-32768, min(32767, int(error_px)))
    sock.sendto(struct.pack("<hhh", visible, error_px, int(round(servo_deg * 10))), (PI_IP, PI_PORT))


# ---------------- camera ----------------
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                               aruco.DetectorParameters())


def open_camera(index):
    # The camera can do 1080p, but Calibration.npz (and so the servo calibration) was made at
    # 1280x720, and 720p gives a higher frame rate = less lag in the control loop. Keep it at 720p.
    c = cv2.VideoCapture(index)
    if not c.isOpened():
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    got = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (1280, 720):
        print(f"WARNING: camera {index} gives {got[0]}x{got[1]}, not 1280x720 - "
              "Calibration.npz and the servo calibration will be wrong")
    return c


cap = open_camera(CAMERA_INDEX)
if cap is None:
    raise SystemExit(f"Could not open camera {CAMERA_INDEX} - try another CAMERA_INDEX")

last_send = 0.0
last_seen_time = -1e9
last_error_px, last_servo = 0, SERVO_ZERO
streak = 0
v_pixel = None            # undistorted vertical pixel of the current marker (for calibration)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame")
        break
    h, w = frame.shape[:2]
    target_u = w // 2 + TARGET_U_OFFSET_PX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    visible, error_px, servo, marker_id, v_pixel = 0, 0, SERVO_ZERO, None, None
    if ids is not None and len(ids) > 0:
        for ind, i in enumerate(ids.flatten()):
            if TARGET_ID is None or int(i) == TARGET_ID:
                mx, my = corners[ind][0].mean(axis=0)
                # Undistort the centre but keep it in pixels (P=CM), as Servo_Calibration.py does
                u_und, v_und = cv2.undistortPoints(np.array([[[mx, my]]], dtype=np.float32),
                                                   CM, DIST, P=CM)[0, 0]
                v_pixel = float(v_und)
                if coeffs is not None:
                    servo = y_to_servo(pixel_to_y(v_pixel, coeffs))
                visible, error_px, marker_id = 1, int(round(u_und - target_u)), int(i)
                cv2.circle(frame, (int(mx), int(my)), 8, (0, 0, 255), -1)
                break
        aruco.drawDetectedMarkers(frame, corners, ids)

    # False-detection filter + blackout tolerance (same as gantry_center_aruco.py)
    now = time.perf_counter()
    streak = streak + 1 if visible else 0
    tracking = now - last_seen_time < HOLD_TIME
    held = confirming = False
    if visible and (tracking or streak >= CONFIRM_FRAMES):
        last_seen_time, last_error_px, last_servo = now, error_px, servo
    elif visible:
        visible, error_px, servo, confirming = 0, 0, last_servo, True
    elif tracking:
        visible, error_px, servo, held = 1, last_error_px, last_servo, True
    else:
        servo = last_servo        # marker gone: leave the servo where it was

    if now - last_send >= SEND_PERIOD:
        send(visible, error_px, servo)
        last_send = now

    # ---- display ----
    cv2.line(frame, (target_u, 0), (target_u, h), (0, 255, 0), 2)
    for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
        cv2.line(frame, (target_u - d, 0), (target_u - d, h), col, 1)
        cv2.line(frame, (target_u + d, 0), (target_u + d, h), col, 1)
    if confirming:
        text = f"confirming marker {streak}/{CONFIRM_FRAMES}"
    elif held:
        text = f"blackout {now - last_seen_time:.2f} s -> holding"
    elif visible:
        text = f"ID {marker_id}  X error {error_px:+d} px  servo {servo:.1f} deg"
        if coeffs is not None and v_pixel is not None:
            text += f"  (Y {pixel_to_y(v_pixel, coeffs):+.0f} mm)"
    else:
        text = "no marker -> STOP"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cal = "servo calibrated" if coeffs is not None else f"servo NOT calibrated ({len(points)}/3)"
    cv2.putText(frame, f"camera {CAMERA_INDEX} | {cal} | k = add point, x = clear, c = camera, q = quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"X offset {TARGET_U_OFFSET_PX:+d} px (a / d to nudge)", (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("gantry-laser-servo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("k"):
        if v_pixel is None:
            print("No marker visible - put the marker on a floor mark first")
            continue
        p = v_pixel                       # freeze before typing
        send(0, 0, last_servo)            # stop the gantry while you type
        try:
            y_mm = float(input(f"Marker pixel v = {p:.1f}. Type this mark's floor Y in mm and press Enter: "))
        except ValueError:
            print("Not a number - point not added")
            continue
        points.append((p, y_mm))
        save_cal(points)
        print(f"Added point {len(points)}: pixel {p:.1f} -> Y {y_mm:.1f} mm")
        if len(points) >= 3:
            coeffs = fit_pixel_to_y(points)
            report_fit(points, coeffs)
        else:
            print(f"Need {3 - len(points)} more point(s)")
    if key in (ord("a"), ord("d")):
        # Nudge where the gantry stops along X (laser not lined up with the camera). 10 px per press;
        # roughly 1.6 px per mm on the floor, so 5 cm is about 80 px. Put the final value in
        # TARGET_U_OFFSET_PX at the top so it is kept next time.
        TARGET_U_OFFSET_PX += -10 if key == ord("a") else 10
        print(f"TARGET_U_OFFSET_PX = {TARGET_U_OFFSET_PX}")
    if key == ord("x"):
        points, coeffs = [], None
        save_cal(points)
        print("Servo calibration cleared - add points again with k")
    if key == ord("c"):
        send(0, 0, last_servo)
        last_seen_time = -1e9
        cap.release()
        for step in range(1, 5):
            new_cap = open_camera((CAMERA_INDEX + step) % 4)
            if new_cap is not None:
                CAMERA_INDEX = (CAMERA_INDEX + step) % 4
                cap = new_cap
                break
        print(f"Switched to camera {CAMERA_INDEX}")

send(0, 0, last_servo)   # stop the gantry, laser off
sock.close()
cap.release()
cv2.destroyAllWindows()
