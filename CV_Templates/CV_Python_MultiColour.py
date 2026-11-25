"""
Multi-colour circle detection - step 1 - based on CV_Python_Circle_steps.py.

Checks that the camera can see and tell apart 5 coloured circles
(yellow, orange, blue, green, pink) on a dark floor. Every colour is searched
for in every frame; each circle found is outlined in its own colour.

Priority selection and sending to the Pi come in step 2.

Sections marked  # CHANGED (multi-colour)  differ from CV_Python_Circle_steps.py.

Windows:
  circle-image  camera image with every detected circle outlined + status panel
  mask          mask of the colour selected with the "colour" trackbar (tune it here)
  all-masks     all the masks side by side - each circle should be white in ONE mask only

Keys (click a window first):
  s  save all colour ranges to colours.json (loaded automatically next time)
  q  quit and print the colour ranges
"""

import os
import json
import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX = 1   # 0 = laptop camera, 1 = USB camera (Arducam)
RESOLUTION = (1280, 720)

# Locked camera settings (found with Camera_Settings_Test.py)
EXPOSURE = -7
GAIN = 0
WB_TEMPERATURE = 4500       # only used when AUTO_WHITE_BALANCE is False
# True  = the camera adjusts white balance itself (colours look natural, may drift a bit)
# False = locked at WB_TEMPERATURE (steady, but wrong colours if the lighting doesn't match)
AUTO_WHITE_BALANCE = True

MIN_AREA = 300              # ignore blobs smaller than this (pixels)
MIN_CIRCULARITY = 0.75      # blob area / fitted ellipse area (1.0 = perfect ellipse)
PRINT_PERIOD = 0.5          # seconds between terminal prints

# ---------------------------------------------------------------------------
# CHANGED (multi-colour): one HSV range per colour instead of HSV_LOW/HSV_HIGH.
# OpenCV hue goes 0-179. Yellow, orange and green are close on the hue scale,
# so make sure their H ranges do NOT overlap.
# If H low > H high the range wraps around 179 -> 0 (needed for red/pink).
# ---------------------------------------------------------------------------
# Ranges centred on the colours measured from the reference circle image
# (measured H, S, V in the comments). The printed circles seen through the
# camera will differ a bit, so fine-tune with the trackbars and save with s.
COLOURS = [  # name,     HSV low,          HSV high,         BGR draw colour
    # Measured through the camera (H, S, V):
    ["yellow", [18, 40, 150],  [30, 255, 255],  (0, 242, 255)],    # camera: 23, 77, 215-236
    ["orange", [176, 50, 100], [11, 255, 255],  (34, 148, 247)],   # camera: 4, 214, 210 and 176, 77, 223 (wraps 179 -> 0)
    ["green",  [58, 50, 90],   [84, 255, 255],  (64, 197, 141)],   # camera: 66-74, 79-119, 124-151
    # Blue covers the old purple range too: under some lighting the camera
    # sees the blue circle at purple hues (~105-125)
    ["blue",   [88, 100, 80],  [128, 255, 255], (227, 178, 4)],
    ["pink",   [145, 60, 100], [175, 255, 255], (147, 54, 238)],   # tuned, works well
]

# CHANGED (multi-colour): saved tuning, loaded at start-up if it exists
COLOURS_FILE = os.path.join(SCRIPT_DIR, "colours.json")


def load_colours():
    if not os.path.exists(COLOURS_FILE):
        return
    with open(COLOURS_FILE) as f:
        saved = json.load(f)
    for colour in COLOURS:
        if colour[0] in saved:
            colour[1] = list(saved[colour[0]]["low"])
            colour[2] = list(saved[colour[0]]["high"])
    logging.info("Loaded colour ranges from %s", COLOURS_FILE)


def save_colours():
    data = {name: {"low": low, "high": high} for name, low, high, _ in COLOURS}
    with open(COLOURS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logging.info("Saved colour ranges to %s", COLOURS_FILE)


def print_colours():
    print("\nColour ranges:")
    for name, low, high, _ in COLOURS:
        print(f"  {name:7s} low = {tuple(low)}  high = {tuple(high)}")


def draw_text(img, text, pos, scale=0.6, colour=(0, 255, 0)):
    """Text with a black outline, readable on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def make_mask(hsv, low, high):
    """CHANGED (multi-colour): takes an already-converted HSV image, and
    handles hue wrap-around (H low > H high) for red/pink."""
    if low[0] <= high[0]:
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
    else:
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(low), np.array([179, high[1], high[2]])),
            cv2.inRange(hsv, np.array([0, low[1], low[2]]), np.array(high)))
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
        if area / ellipse_area < MIN_CIRCULARITY:
            continue
        if area > best_area:
            best, best_area = ellipse, area
    return best


# ---------------------------------------------------------------------------
# CHANGED (multi-colour): "colour" trackbar picks which colour the HSV
# trackbars edit
# ---------------------------------------------------------------------------
load_colours()

HSV_BARS = [("H low", 1, 0, 179), ("H high", 2, 0, 179),
            ("S low", 1, 1, 255), ("S high", 2, 1, 255),
            ("V low", 1, 2, 255), ("V high", 2, 2, 255)]   # (name, low/high index, H/S/V index, max)


def nothing(_):
    pass


cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mask", RESOLUTION[0] // 2, RESOLUTION[1] // 2)
cv2.createTrackbar("colour", "mask", 0, len(COLOURS) - 1, nothing)
for bar, which, channel, maximum in HSV_BARS:
    cv2.createTrackbar(bar, "mask", COLOURS[0][which][channel], maximum, nothing)


def load_trackbars(index):
    """Move the HSV trackbars to the stored values of colour `index`."""
    for bar, which, channel, _ in HSV_BARS:
        cv2.setTrackbarPos(bar, "mask", COLOURS[index][which][channel])


def store_trackbars(index):
    """Copy the HSV trackbar positions into colour `index`."""
    for bar, which, channel, _ in HSV_BARS:
        COLOURS[index][which][channel] = cv2.getTrackbarPos(bar, "mask")


# ---------------------------------------------------------------------------
# Open the camera (same locked settings as the single-colour script)
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    logging.error("Could not open camera %d", CAMERA_INDEX)
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
cap.set(cv2.CAP_PROP_GAIN, GAIN)
# CHANGED (multi-colour): white balance can be auto or locked (see AUTO_WHITE_BALANCE)
if AUTO_WHITE_BALANCE:
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
else:
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMPERATURE)
logging.info("Camera %d opened at %dx%d", CAMERA_INDEX,
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

cv2.namedWindow("circle-image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("circle-image", RESOLUTION[0] // 2, RESOLUTION[1] // 2)
cv2.namedWindow("all-masks", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("mask", 0, 50)
cv2.moveWindow("circle-image", 660, 50)
cv2.moveWindow("all-masks", 0, 500)

# ---------------------------------------------------------------------------
# CHANGED (multi-colour): click on the circle-image window to print the real
# HSV value the camera sees at that point (median of an 11x11 patch).
# Use it to set each colour's range: H about +/-6 around the value,
# S low and V low about 40-60 below the value.
# ---------------------------------------------------------------------------
latest_hsv = None
clicked_point = None


def on_click(event, x, y, flags, param):
    global clicked_point
    if event != cv2.EVENT_LBUTTONDOWN or latest_hsv is None:
        return
    h_img, w_img = latest_hsv.shape[:2]
    x, y = min(max(x, 5), w_img - 6), min(max(y, 5), h_img - 6)
    patch = latest_hsv[y - 5:y + 6, x - 5:x + 6].reshape(-1, 3)
    h, s, v = np.median(patch, axis=0).astype(int)
    clicked_point = (x, y, f"H={h} S={s} V={v}")
    print(f"Clicked ({x}, {y}): H={h}  S={s}  V={v}")


cv2.setMouseCallback("circle-image", on_click)

selected = 0
last_print = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera %d", CAMERA_INDEX)
        break

    # CHANGED (multi-colour): keep the trackbars and the selected colour in sync
    new_selected = cv2.getTrackbarPos("colour", "mask")
    if new_selected != selected:
        selected = new_selected
        load_trackbars(selected)
    else:
        store_trackbars(selected)

    # Blur and convert to HSV once per frame, shared by all colours
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    latest_hsv = hsv   # CHANGED (multi-colour): used by the click-to-measure tool

    out = frame.copy()
    masks, status = [], []

    # CHANGED (multi-colour): search for every colour in every frame
    for name, low, high, draw_colour in COLOURS:
        mask = make_mask(hsv, low, high)
        masks.append(mask)
        ellipse = find_circle(mask)
        if ellipse is None:
            status.append((f"{name}: -", (160, 160, 160)))
            continue
        (u, v), _, _ = ellipse
        cv2.ellipse(out, ellipse, draw_colour, 4)
        cv2.circle(out, (int(u), int(v)), 6, draw_colour, -1)
        draw_text(out, name, (int(u) + 12, int(v) - 12), 0.8, draw_colour)
        status.append((f"{name}: SEEN ({u:.0f}, {v:.0f})", draw_colour))

    # Status panel
    for n, (text, colour) in enumerate(status):
        draw_text(out, text, (10, 35 + n * 32), 0.9, colour)

    # CHANGED (multi-colour): show the last clicked point and its HSV value
    if clicked_point is not None:
        px, py, label = clicked_point
        cv2.drawMarker(out, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 25, 2)
        draw_text(out, label, (px + 15, py + 30), 0.9, (255, 255, 255))

    now = time.perf_counter()
    if now - last_print >= PRINT_PERIOD:
        print(" | ".join(text for text, _ in status))
        last_print = now

    # Selected colour's mask, labelled
    mask_view = cv2.cvtColor(masks[selected], cv2.COLOR_GRAY2BGR)
    draw_text(mask_view, f"editing: {COLOURS[selected][0]}", (10, 50), 1.5, COLOURS[selected][3])

    # All 5 masks side by side, shrunk and labelled
    small = []
    for (name, _, _, draw_colour), mask in zip(COLOURS, masks):
        tile = cv2.cvtColor(cv2.resize(mask, (256, 144)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile, (0, 0), (255, 143), draw_colour, 2)
        draw_text(tile, name, (8, 24), 0.6, draw_colour)
        small.append(tile)

    cv2.imshow("circle-image", out)
    cv2.imshow("mask", mask_view)
    cv2.imshow("all-masks", np.hstack(small))

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_colours()

print_colours()
cap.release()
cv2.destroyAllWindows()
logging.info("Camera capture stopped")
