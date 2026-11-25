"""
Fire the laser at a YELLOW CIRCLE: the gantry moves along X, the servo tilts along Y.
Runs with gantry_home_center_laser_servo.slx on the Pi (homing, 7900-step limit, laser dwell, servo)
- the Pi does not care whether the target is an ArUco or a circle, it only gets the numbers.

Same as gantry_aruco_laser_servo.py, with the ArUco detector swapped for the team's yellow circle
detection (CV_Templates/CV_Python_Circle_steps.py + CV_Python_MultiColour.py):
  - HSV colour range for "yellow" from CV_Templates/colours.json (tune it in CV_Python_MultiColour.py)
  - locked exposure / gain so the floor is dark and the circle is bright
  - biggest blob that is round enough (area / fitted ellipse area >= MIN_CIRCULARITY)
Kept from the ArUco version: blackout hold, confirm-frames filter, -60 px camera/laser X offset,
servo aiming from CV_Templates/servo_calibration.json (it maps a floor point's pixel row to Y, so it
works for any target on the floor - no need to recalibrate for circles).

UDP packet = 3 x int16, little-endian: [visible (0/1), error_px, servo angle x 10]
Keys (click the camera window first): q = quit, a / d = nudge X offset, c = switch camera
"""

import json
import math
import os
import socket
import struct
import time

import cv2
import numpy as np

PI_IP = "138.38.226.64"
PI_PORT = 50001
SEND_PERIOD = 0.1
ON_PX, OFF_PX = 40, 15    # only for drawing - the real thresholds are in the Simulink model
CAMERA_INDEX = 0          # USB camera. If c is needed to find it, put the number shown here.
HOLD_TIME = 0.4           # s: keep the last target through short detection blackouts
CONFIRM_FRAMES = 3        # a new circle must be seen this many frames in a row

# ---- Laser servo: keep these the same as gantry_aruco_laser_servo.py ----
PIVOT_HEIGHT = 640.0
SERVO_ZERO = 90.0
SERVO_SIGN = -1
SERVO_DEG_PER_CMD = 0.48
TARGET_U_OFFSET_PX = -60

# ---- Circle detection (team's values) ----
COLOUR = "yellow"         # any colour in colours.json
EXPOSURE = -7             # locked exposure used when colours.json was tuned (more negative = darker)
GAIN = 0
AUTO_WHITE_BALANCE = True
MIN_AREA = 300            # ignore blobs smaller than this (pixels)
MIN_CIRCULARITY = 0.75    # blob area / fitted ellipse area (1.0 = perfect ellipse)
# The check above lets squares through (a square fills ~92% of its fitted ellipse), so also count
# corners after simplifying the outline: square/triangle = 3-4, circle (even seen at an angle) = many.
MIN_CORNERS = 7
SMOOTHING = 0.3           # centre smoothing (0 = none). Kept lower than the team's 0.5: the camera
                          # moves with the gantry, and heavy smoothing adds lag -> overshoot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV = os.path.join(REPO, "CV_Templates")
cam = np.load(os.path.join(CV, "Calibration.npz"))
CM, DIST = cam["CM"], cam["dist_coef"]
with open(os.path.join(CV, "colours.json")) as f:
    _colour = json.load(f)[COLOUR]
HSV_LOW, HSV_HIGH = tuple(_colour["low"]), tuple(_colour["high"])
SERVO_CAL_FILE = os.path.join(CV, "servo_calibration.json")


# ---------------- servo aiming (same as the ArUco servo script) ----------------
def load_fit():
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


# ---------------- circle detection (team's code) ----------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def make_mask(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    low, high = np.array(HSV_LOW), np.array(HSV_HIGH)
    if low[0] <= high[0]:
        mask = cv2.inRange(hsv, low, high)
    else:   # hue wraps round 0/179 (e.g. orange/red in colours.json)
        mask = cv2.inRange(hsv, low, np.array([179, *high[1:]])) | \
               cv2.inRange(hsv, np.array([0, *low[1:]]), high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def find_circle(mask):
    """Biggest circle-like blob as an ellipse ((u, v), (w, h), angle), or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_area = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        w, h = ellipse[1]
        ellipse_area = np.pi * w * h / 4
        if ellipse_area <= 0 or area / ellipse_area < MIN_CIRCULARITY:
            continue
        corners = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(corners) < MIN_CORNERS:
            continue
        if area > best_area:
            best, best_area = ellipse, area
    return best


def open_camera(index):
    # DirectShow backend: needed to lock exposure/gain so the colour stays the same
    c = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not c.isOpened():
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Calibration.npz was made at 1280x720
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    c.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    c.set(cv2.CAP_PROP_GAIN, GAIN)
    c.set(cv2.CAP_PROP_AUTO_WB, 1 if AUTO_WHITE_BALANCE else 0)
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
    print(f"Looking for {COLOUR} circles, HSV {HSV_LOW} - {HSV_HIGH}")
    print("Servo calibrated" if fit else "Servo NOT calibrated - it stays at SERVO_ZERO")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(visible, error_px, servo_deg):
        error_px = max(-32768, min(32767, int(error_px)))
        sock.sendto(struct.pack("<hhh", visible, error_px, int(round(servo_deg * 10))), (PI_IP, PI_PORT))

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(f"Could not open camera {CAMERA_INDEX} - try another CAMERA_INDEX")
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", 480, 270)

    last_send, last_seen_time = 0.0, -1e9
    last_error_px, last_servo = 0, SERVO_ZERO
    streak, smooth = 0, None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break
        h, w = frame.shape[:2]
        target_u = w // 2 + TARGET_U_OFFSET_PX

        mask = make_mask(frame)
        ellipse = find_circle(mask)
        visible, error_px, servo = 0, 0, last_servo
        if ellipse is not None:
            centre = np.array(ellipse[0])
            smooth = centre if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * centre
            u_und, v_und = cv2.undistortPoints(np.array([[smooth]], dtype=np.float32), CM, DIST, P=CM)[0, 0]
            visible, error_px, servo = 1, int(round(u_und - target_u)), servo_for(float(v_und), fit)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 3)
            cv2.circle(frame, (int(smooth[0]), int(smooth[1])), 6, (0, 0, 255), -1)
        else:
            smooth = None

        # False-detection filter + blackout tolerance (same as the ArUco scripts)
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

        if now - last_send >= SEND_PERIOD:
            send(visible, error_px, servo)
            last_send = now

        # ---- display ----
        cv2.line(frame, (target_u, 0), (target_u, h), (0, 255, 0), 2)
        for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
            cv2.line(frame, (target_u - d, 0), (target_u - d, h), col, 1)
            cv2.line(frame, (target_u + d, 0), (target_u + d, h), col, 1)
        if confirming:
            text = f"confirming circle {streak}/{CONFIRM_FRAMES}"
        elif held:
            text = f"blackout {now - last_seen_time:.2f} s -> holding"
        elif visible:
            text = f"{COLOUR} circle  X error {error_px:+d} px  servo {servo:.1f} deg"
        else:
            text = f"no {COLOUR} circle -> STOP"
        draw_text(frame, text, (10, 35))
        draw_text(frame, f"camera {CAMERA_INDEX} | X offset {TARGET_U_OFFSET_PX:+d} px | "
                         f"servo {'calibrated' if fit else 'NOT calibrated'} | q a/d c",
                  (10, h - 15), 0.6, (255, 255, 255))
        cv2.imshow("gantry-yellow-circle", frame)
        cv2.imshow("mask", mask)   # the circle should be white here and nothing else

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("a"), ord("d")):
            TARGET_U_OFFSET_PX += -10 if key == ord("a") else 10
            print(f"TARGET_U_OFFSET_PX = {TARGET_U_OFFSET_PX}")
        if key == ord("c"):
            send(0, 0, last_servo)
            last_seen_time, smooth = -1e9, None
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


if __name__ == "__main__":
    main()
