"""
Laser servo calibration (Y axis, at right angles to the gantry).

Separate from CV_Python_Circle_steps.py for now. Uses the same yellow circle
detection, and works out how the circle's pixel position maps to its floor
distance Y (mm) from the point directly below the servo pivot. From Y it
gives the servo angle that points the laser at the target.

Why this works: the target is always on the floor and the camera rides on the
carriage, so each pixel position along the image's Y direction is one floor
distance. The fit  Y = (a*p + b) / (c*p + 1)  is exact for a flat floor seen by
a tilted camera, so the camera height/angle/offset never need measuring.

Servo angle = SERVO_ZERO + SERVO_SIGN * atan2(Y, PIVOT_HEIGHT)
(the beam passes through the servo axis, so the pivot-to-laser-tip length does not matter)

How to calibrate:
  1. Set PIVOT_HEIGHT, SERVO_ZERO and SERVO_SIGN below.
  2. Mark Y = 0 on the floor (directly below the pivot, laser pointing straight
     down) and a few more points on a line at right angles to the gantry,
     e.g. -200, -100, 100, 200 mm.
  3. Put the circle on a mark (gantry centred on it in X), click the
     circle-image window, press c and type the mark's Y in mm in the terminal.
  4. Repeat for 3 or more marks (5 is better). The fit error at each mark is
     printed - it should be a few mm. Saved to servo_calibration.json.
  5. Check: put the circle on a new measured spot, the shown Y should match.

Keys: c = add point, x = clear all points, q = quit
"""

import os
import json
import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Camera (same as CV_Python_Circle_steps.py)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1   # 0 = laptop camera, 1 = USB camera (Arducam)
CALIBRATION_FILE = 'Calibration.npz'
RESOLUTION = (1280, 720)
camera_calibration = np.load(os.path.join(SCRIPT_DIR, CALIBRATION_FILE))
CM = camera_calibration['CM']
dist_coef = camera_calibration['dist_coef']

EXPOSURE = -6
GAIN = 0
WB_TEMPERATURE = 4500
AUTO_WHITE_BALANCE = True

# Yellow target (tuned in the multi-colour script)
HSV_LOW = (18, 40, 150)
HSV_HIGH = (30, 255, 255)
MIN_AREA = 300
MIN_CIRCULARITY = 0.75
SMOOTHING = 0.5

# ---------------------------------------------------------------------------
# Laser servo settings - MEASURE / SET THESE
# ---------------------------------------------------------------------------
PIVOT_HEIGHT = 300.0   # mm, height of the servo shaft centre above the floor
SERVO_ZERO = 90.0      # servo command (deg) that points the laser straight down
SERVO_SIGN = 1         # +1 or -1: flip if the laser moves the wrong way
# Which image direction runs along the floor's Y (at right angles to the gantry):
# "v" = up/down in the image, "u" = left/right (if the camera is mounted sideways)
Y_PIXEL_AXIS = "v"
SERVO_CAL_FILE = os.path.join(SCRIPT_DIR, "servo_calibration.json")


# ---------------------------------------------------------------------------
# Circle detection (same as CV_Python_Circle_steps.py)
# ---------------------------------------------------------------------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def make_mask(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_LOW), np.array(HSV_HIGH))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def find_circle(mask):
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
        if area > best_area:
            best, best_area = ellipse, area
    return best


def draw_text(img, text, pos, scale=0.7, colour=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


# ---------------------------------------------------------------------------
# Calibration maths
# ---------------------------------------------------------------------------
def fit_pixel_to_y(points):
    """Fit Y = (a*p + b) / (c*p + 1) to [(p, Y), ...].
    Linear least squares: a*p + b - c*p*Y = Y."""
    p = np.array([pt[0] for pt in points], dtype=float)
    y = np.array([pt[1] for pt in points], dtype=float)
    A = np.column_stack([p, np.ones_like(p), -p * y])
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b, c


def pixel_to_y(p, coeffs):
    a, b, c = coeffs
    return (a * p + b) / (c * p + 1)


def y_to_servo(y_mm):
    theta = np.degrees(np.arctan2(y_mm, PIVOT_HEIGHT))   # 0 = straight down
    return SERVO_ZERO + SERVO_SIGN * theta


def save_cal(points):
    with open(SERVO_CAL_FILE, "w") as f:
        json.dump({"axis": Y_PIXEL_AXIS, "points": points}, f, indent=2)


def load_cal():
    if not os.path.exists(SERVO_CAL_FILE):
        return []
    with open(SERVO_CAL_FILE) as f:
        data = json.load(f)
    if data.get("axis") != Y_PIXEL_AXIS:
        logging.warning("servo_calibration.json was made with Y_PIXEL_AXIS=%s - ignoring it",
                        data.get("axis"))
        return []
    logging.info("Loaded %d calibration points from %s", len(data["points"]), SERVO_CAL_FILE)
    return [tuple(pt) for pt in data["points"]]


def report_fit(points, coeffs):
    print("Calibration fit (pixel -> Y mm):")
    for p, y in points:
        fitted = pixel_to_y(p, coeffs)
        print(f"  pixel {p:7.1f}  measured Y {y:7.1f}  fitted Y {fitted:7.1f}  error {fitted - y:+6.1f} mm")


points = load_cal()
coeffs = fit_pixel_to_y(points) if len(points) >= 3 else None
if coeffs is not None:
    report_fit(points, coeffs)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {CAMERA_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
cap.set(cv2.CAP_PROP_GAIN, GAIN)
if AUTO_WHITE_BALANCE:
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
else:
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMPERATURE)

cv2.namedWindow("circle-image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("circle-image", RESOLUTION[0] * 3 // 4, RESOLUTION[1] * 3 // 4)

smooth_centre = None
y_pixel = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame")
        break

    ellipse = find_circle(make_mask(frame))
    out = frame.copy()

    if ellipse is None:
        smooth_centre, y_pixel = None, None
        draw_text(out, "No circle detected", (10, 30))
    else:
        (u, v), _, _ = ellipse
        centre = np.array([u, v])
        smooth_centre = centre if smooth_centre is None else \
            SMOOTHING * smooth_centre + (1 - SMOOTHING) * centre
        u, v = smooth_centre

        # Undistort the centre but keep it in pixels (P=CM) so the fit stays exact
        u_und, v_und = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32),
                                           CM, dist_coef, P=CM)[0, 0]
        y_pixel = float(v_und if Y_PIXEL_AXIS == "v" else u_und)

        cv2.ellipse(out, ellipse, (0, 255, 0), 3)
        cv2.circle(out, (int(u), int(v)), 6, (0, 0, 255), -1)

        lines = [f"pixel ({Y_PIXEL_AXIS}) = {y_pixel:.1f}"]
        if coeffs is not None:
            y_floor = pixel_to_y(y_pixel, coeffs)
            lines.append(f"floor Y = {y_floor:+.0f} mm")
            lines.append(f"servo = {y_to_servo(y_floor):.1f} deg")
        else:
            lines.append(f"NOT CALIBRATED ({len(points)}/3 points) - press c")
        for n, line in enumerate(lines):
            draw_text(out, line, (10, 30 + n * 30))

    draw_text(out, f"c = add point ({len(points)} so far)   x = clear   q = quit",
              (10, RESOLUTION[1] - 20), 0.7, (255, 255, 255))
    cv2.imshow("circle-image", out)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if y_pixel is None:
            print("No circle visible - put the circle on a floor mark first")
            continue
        p = y_pixel   # freeze the pixel value before typing
        try:
            y_mm = float(input(f"Circle pixel = {p:.1f}. Type its floor Y in mm and press Enter: "))
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
    elif key == ord('x'):
        points, coeffs = [], None
        save_cal(points)
        print("Calibration cleared")

cap.release()
cv2.destroyAllWindows()
