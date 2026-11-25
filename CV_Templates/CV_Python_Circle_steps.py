"""
Circle target detection - based on CV_Python_Aruco_steps.py.

Detects a BRIGHT COLOURED circle (e.g. fluorescent green) on a dark/black floor,
finds its centre, how far it is from the image centre (pixels and angles),
how much the circle is tilted, and (if its real diameter is known) its x, y, z
position in mm. Sends the result to the Raspberry Pi in the same format as the
ArUco script, so Pi_Aruco_Receiver.py works unchanged.

Sections marked  # CHANGED  differ from the ArUco version; everything else is the same.

Message sent to the Pi:
  CIRCLE_DIAMETER set  -> "0,x,y,z"          (mm, like the ArUco script, id is always 0)
  CIRCLE_DIAMETER None -> "0,err_x,err_y,0"  (pixels from the image centre)
  no circle            -> "NONE"

Use the trackbars in the "mask" window to tune the colour, press q to quit
(the final colour values are printed so you can paste them into HSV_LOW/HSV_HIGH).

The laser servo calibration is in the separate script Servo_Calibration.py.
"""

import os
import logging
import socket   # Step 11: to send data to the Raspberry Pi
import time

import cv2
# CHANGED: cv2.aruco is no longer needed
import numpy as np

logging.basicConfig(level=logging.INFO)

# Folder this script is in, so the calibration file is found wherever you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Step 2: read the camera calibration (unchanged)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1   # 0 = laptop camera, 1 = USB camera (Arducam). NOTE: Calibration.npz is for the USB camera
CALIBRATION_FILE = 'Calibration.npz'
CALIBRATION_RESOLUTION = (1280, 720)
camera_calibration = np.load(os.path.join(SCRIPT_DIR, CALIBRATION_FILE))
CM = camera_calibration['CM']                 # camera matrix
dist_coef = camera_calibration['dist_coef']   # distortion coefficients
logging.info("Loaded camera calibration from %s", CALIBRATION_FILE)

# CHANGED: focal lengths and image centre (pixels) taken from the camera matrix
fx, fy = CM[0, 0], CM[1, 1]
cx, cy = CM[0, 2], CM[1, 2]

# ---------------------------------------------------------------------------
# CHANGED: locked camera settings (found with Camera_Settings_Test.py).
# Lower EXPOSURE (more negative) until the floor is near black but the circle
# is still clearly bright.
# ---------------------------------------------------------------------------
EXPOSURE = -6
GAIN = 0
WB_TEMPERATURE = 4500       # only used when AUTO_WHITE_BALANCE is False
# CHANGED: same as the multi-colour script - auto white balance on (the locked
# 4500 K gave a purple tint), set False to lock it at WB_TEMPERATURE
AUTO_WHITE_BALANCE = True

# ---------------------------------------------------------------------------
# CHANGED (replaces Step 3 ArUco dictionary/detector and Step 7 marker_points):
# circle target settings
# ---------------------------------------------------------------------------
# Real diameter of the circle in mm. Set it to get distance and x, y, z in mm.
# Leave it as None to detect a circle of ANY size (direction/angles only).
CIRCLE_DIAMETER = None      # e.g. 100 for a 10 cm circle

# True  = detect by colour (recommended, ignores white reflections)
# False = detect anything bright (e.g. a white circle)
USE_COLOUR = True

# HSV colour range of the target. OpenCV hue goes 0-179.
# Fluorescent green is roughly H 35-85. Tune with the trackbars.
# (Red/pink wraps around hue 0/179, so for red pick a range on one side only.)
# CHANGED (servo): yellow values tuned in the multi-colour script (colours.json)
HSV_LOW = (18, 40, 150)     # (H, S, V) minimum - V minimum ignores the dark floor
HSV_HIGH = (30, 255, 255)   # (H, S, V) maximum

MIN_AREA = 300              # ignore blobs smaller than this (pixels)
MIN_CIRCULARITY = 0.75      # blob area / fitted ellipse area (1.0 = perfect ellipse)
SMOOTHING = 0.5             # 0 = no smoothing, closer to 1 = smoother but slower

# Print the results at most every PRINT_PERIOD seconds so the terminal stays readable
PRINT_PERIOD = 0.5


# ---------------------------------------------------------------------------
# Step 11: send to the Raspberry Pi over WiFi (UDP) - same as the ArUco script
# ---------------------------------------------------------------------------
SEND_TO_PI = True
PI_IP = "192.168.1.100"     # CHANGE to your Pi's IP address
PI_PORT = 50002             # must match UDP_PORT in Pi_Aruco_Receiver.py
SEND_PERIOD = 0.1           # seconds between messages (10 per second)

pi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
if SEND_TO_PI:
    logging.info("Sending circle data to the Pi at %s:%s", PI_IP, PI_PORT)


def send_to_pi(values):
    """CHANGED: sends one target as "0,a,b,c", or "NONE" when no circle is visible."""
    if values is None:
        message = "NONE"
    else:
        a, b, c = values
        message = f"0,{a:.1f},{b:.1f},{c:.1f}"
    try:
        pi_sock.sendto(message.encode("utf-8"), (PI_IP, PI_PORT))
    except OSError as error:
        # e.g. not connected to WiFi: keep the camera running anyway
        logging.warning("Could not send to the Pi: %s", error)


def draw_text(img, text, pos, scale=0.6):
    """Green text with a black outline, readable on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# CHANGED (replaces Step 4 detector.detectMarkers): find the circle
# ---------------------------------------------------------------------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def make_mask(frame, hsv_low, hsv_high):
    """White where the target is, black everywhere else."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    if USE_COLOUR:
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    else:
        # Bright object on a dark floor: normal (NOT inverted) Otsu threshold
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Remove small speckles, then fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def find_circle(mask):
    """Return the biggest circle-like blob as an ellipse ((u, v), (w, h), angle), or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_area = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or len(contour) < 5:   # fitEllipse needs 5+ points
            continue
        ellipse = cv2.fitEllipse(contour)
        (w, h) = ellipse[1]
        ellipse_area = np.pi * w * h / 4
        if ellipse_area <= 0:
            continue
        # A tilted circle is still a full ellipse, so this check allows for tilt
        # (unlike 4*pi*area/perimeter^2, which fails for tilted circles)
        if area / ellipse_area < MIN_CIRCULARITY:
            continue
        if area > best_area:
            best, best_area = ellipse, area
    return best


# ---------------------------------------------------------------------------
# CHANGED: trackbars to tune the target colour live
# ---------------------------------------------------------------------------
def nothing(_):
    pass


cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
if USE_COLOUR:
    for name, value, maximum in [("H low", HSV_LOW[0], 179), ("H high", HSV_HIGH[0], 179),
                                 ("S low", HSV_LOW[1], 255), ("S high", HSV_HIGH[1], 255),
                                 ("V low", HSV_LOW[2], 255), ("V high", HSV_HIGH[2], 255)]:
        cv2.createTrackbar(name, "mask", value, maximum, nothing)


def read_trackbars():
    if not USE_COLOUR:
        return HSV_LOW, HSV_HIGH
    low = tuple(cv2.getTrackbarPos(n, "mask") for n in ("H low", "S low", "V low"))
    high = tuple(cv2.getTrackbarPos(n, "mask") for n in ("H high", "S high", "V high"))
    return low, high


# ---------------------------------------------------------------------------
# Open the camera
# CHANGED: DirectShow backend + locked exposure, gain and white balance
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    logging.error("Could not open camera %d", CAMERA_INDEX)
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATION_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATION_RESOLUTION[1])
cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
cap.set(cv2.CAP_PROP_GAIN, GAIN)
if AUTO_WHITE_BALANCE:
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
else:
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMPERATURE)
actual_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
logging.info("Camera %d opened at %dx%d", CAMERA_INDEX, *actual_resolution)
if actual_resolution != CALIBRATION_RESOLUTION:
    logging.warning("Camera resolution %s does not match the calibration %s",
                    actual_resolution, CALIBRATION_RESOLUTION)

# CHANGED: windows are now the mask and the circle result
cv2.namedWindow("circle-image", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("mask", 0, 100)
cv2.moveWindow("circle-image", 640, 100)

last_print = 0.0
last_send = 0.0
smooth_centre = None   # CHANGED: smoothed circle centre (u, v)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera %d", CAMERA_INDEX)
        break

    # CHANGED: colour mask + circle search instead of ArUco detection
    hsv_low, hsv_high = read_trackbars()
    mask = make_mask(frame, hsv_low, hsv_high)
    ellipse = find_circle(mask)

    out = frame.copy()
    now = time.perf_counter()
    print_now = (now - last_print) >= PRINT_PERIOD
    send_values = None

    # CHANGED: crosshair at the image centre (where the camera is pointing)
    cv2.drawMarker(out, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)

    if ellipse is None:
        smooth_centre = None
        draw_text(out, "No circle detected", (10, 30), scale=0.7)
        if print_now:
            print("No circle detected")
    else:
        # -------------------------------------------------------------------
        # CHANGED (replaces Steps 7-10 solvePnP pose): measurements from the ellipse
        # -------------------------------------------------------------------
        (u, v), (w, h), angle = ellipse
        major, minor = max(w, h), min(w, h)

        # Smooth the centre so it does not jitter
        if smooth_centre is None:
            smooth_centre = np.array([u, v])
        else:
            smooth_centre = SMOOTHING * smooth_centre + (1 - SMOOTHING) * np.array([u, v])
        u, v = smooth_centre

        # Pixel error from the image centre: +x = right, +y = down
        err_x, err_y = u - cx, v - cy

        # Direction to the target (corrected for lens distortion):
        # normalised coordinates xn, yn -> angles in degrees
        xn, yn = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), CM, dist_coef)[0, 0]
        pan_angle = np.degrees(np.arctan(xn))     # left/right -> gantry
        tilt_angle = np.degrees(np.arctan(yn))    # up/down    -> servo

        # How much the circle itself is tilted relative to the camera
        circle_tilt = np.degrees(np.arccos(np.clip(minor / major, 0, 1)))

        lines = [f"error x = {err_x:+.0f} px   y = {err_y:+.0f} px",
                 f"pan = {pan_angle:+.1f} deg   tilt = {tilt_angle:+.1f} deg",
                 f"circle tilt = {circle_tilt:.0f} deg   size = {major:.0f} px"]

        if CIRCLE_DIAMETER is not None:
            # Known size -> distance. The major axis does not shrink with tilt.
            z = fx * CIRCLE_DIAMETER / major
            x, y = xn * z, yn * z
            lines.append(f"x = {x:.0f}  y = {y:.0f}  z = {z:.0f} mm")
            send_values = (x, y, z)
        else:
            send_values = (err_x, err_y, 0.0)

        # Draw the ellipse, its centre and a line from the crosshair to it
        cv2.ellipse(out, ellipse, (0, 255, 0), 3)
        cv2.circle(out, (int(u), int(v)), 6, (0, 0, 255), -1)
        cv2.line(out, (int(cx), int(cy)), (int(u), int(v)), (0, 255, 255), 2)

        for n, line in enumerate(lines):
            draw_text(out, line, (10, 30 + n * 28), scale=0.7)

        if print_now:
            print(" | ".join(lines))

    if print_now:
        last_print = now

    # Step 11: send to the Pi (at most every SEND_PERIOD seconds)
    if SEND_TO_PI and (now - last_send) >= SEND_PERIOD:
        send_to_pi(send_values)
        last_send = now

    # Step 6: show the results (CHANGED: mask instead of the grey image)
    cv2.imshow("mask", mask)
    cv2.imshow("circle-image", out)

    # Quit with q
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# CHANGED: print the tuned colour range to paste into HSV_LOW / HSV_HIGH
if USE_COLOUR:
    low, high = read_trackbars()
    print(f"\nHSV_LOW = {low}\nHSV_HIGH = {high}")

cap.release()
pi_sock.close()
cv2.destroyAllWindows()
logging.info("Camera capture stopped")
