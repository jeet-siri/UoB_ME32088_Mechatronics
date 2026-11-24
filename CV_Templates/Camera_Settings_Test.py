"""
Camera settings test for the Arducam UC-A53 Rev.E (or any USB/UVC camera).

Checks which settings the camera accepts (exposure, white balance, focus, gain)
and lets you tune them live, so the target detection does not flicker.

Keys (click the image window first):
  [ / ]  exposure down / up (darker / brighter)
  - / =  gain down / up
  a      toggle auto-exposure
  w      toggle auto white balance
  f      toggle autofocus
  s      open the camera driver's settings dialog (Windows DirectShow only)
  p      print the current values
  q      quit and print cap.set(...) lines to paste into the detection script
"""

import cv2
import numpy as np

CAMERA_INDEX = 1                 # 0 = laptop camera, 1 = USB camera
RESOLUTION = (1280, 720)         # same as the calibration

# DirectShow gives the best access to camera settings on Windows
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {CAMERA_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

PROPS = {
    "AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE,
    "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
    "GAIN": cv2.CAP_PROP_GAIN,
    "AUTO_WB": cv2.CAP_PROP_AUTO_WB,
    "WB_TEMPERATURE": cv2.CAP_PROP_WB_TEMPERATURE,
    "AUTOFOCUS": cv2.CAP_PROP_AUTOFOCUS,
    "FOCUS": cv2.CAP_PROP_FOCUS,
    "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
    "CONTRAST": cv2.CAP_PROP_CONTRAST,
    "SATURATION": cv2.CAP_PROP_SATURATION,
}

# DirectShow auto-exposure values: 0.75 = auto, 0.25 = manual
AE_AUTO, AE_MANUAL = 0.75, 0.25


def read_all():
    return {name: cap.get(prop) for name, prop in PROPS.items()}


def print_all(title):
    print(f"\n--- {title} ---")
    for name, value in read_all().items():
        note = "  (not supported?)" if value == -1 else ""
        print(f"{name:15s} = {value}{note}")


def try_set(name, value):
    """Set a property and report whether the camera actually took it."""
    before = cap.get(PROPS[name])
    accepted = cap.set(PROPS[name], value)
    after = cap.get(PROPS[name])
    ok = accepted and abs(after - value) < 1e-3
    print(f"set {name:15s} {before} -> {value}: "
          f"{'OK' if ok else f'IGNORED/CHANGED (reads {after})'}")
    return ok


print(f"Camera {CAMERA_INDEX} opened at "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print_all("Default values")

# Test which settings the camera accepts: switch to manual everything
print("\n--- Trying manual settings ---")
try_set("AUTO_EXPOSURE", AE_MANUAL)
try_set("EXPOSURE", -6)          # log2 seconds: -6 is about 1/64 s
try_set("GAIN", 0)
try_set("AUTO_WB", 0)
try_set("WB_TEMPERATURE", 4500)
try_set("AUTOFOCUS", 0)
print_all("Values after setting")

auto_exposure = False
auto_wb = False
autofocus = False


def draw_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


cv2.namedWindow("camera-settings", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    values = read_all()
    lines = [
        f"mean brightness: {mean_brightness:.0f} / 255",
        f"auto exposure: {'ON' if auto_exposure else 'off'}   exposure: {values['EXPOSURE']:.0f}",
        f"gain: {values['GAIN']:.0f}",
        f"auto WB: {'ON' if auto_wb else 'off'}   WB temp: {values['WB_TEMPERATURE']:.0f}",
        f"autofocus: {'ON' if autofocus else 'off'}   focus: {values['FOCUS']:.0f}",
        "[ ] exposure  -/= gain  a/w/f toggles  s dialog  p print  q quit",
    ]
    for n, line in enumerate(lines):
        draw_text(frame, line, (10, 30 + n * 26))

    cv2.imshow("camera-settings", frame)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('['):
        try_set("EXPOSURE", values["EXPOSURE"] - 1)
    elif key == ord(']'):
        try_set("EXPOSURE", values["EXPOSURE"] + 1)
    elif key == ord('-'):
        try_set("GAIN", max(0, values["GAIN"] - 5))
    elif key == ord('='):
        try_set("GAIN", values["GAIN"] + 5)
    elif key == ord('a'):
        auto_exposure = not auto_exposure
        try_set("AUTO_EXPOSURE", AE_AUTO if auto_exposure else AE_MANUAL)
    elif key == ord('w'):
        auto_wb = not auto_wb
        try_set("AUTO_WB", 1 if auto_wb else 0)
    elif key == ord('f'):
        autofocus = not autofocus
        try_set("AUTOFOCUS", 1 if autofocus else 0)
    elif key == ord('s'):
        cap.set(cv2.CAP_PROP_SETTINGS, 1)   # opens the driver dialog
    elif key == ord('p'):
        print_all("Current values")

# Print the final values as code to paste into the detection script
final = read_all()
print("\n--- Paste these after opening the camera in your detection script ---")
print(f"cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)")
for name in ["AUTO_EXPOSURE", "EXPOSURE", "GAIN", "AUTO_WB", "WB_TEMPERATURE", "AUTOFOCUS", "FOCUS"]:
    if final[name] != -1:
        print(f"cap.set(cv2.CAP_PROP_{name}, {final[name]})")

cap.release()
cv2.destroyAllWindows()
