"""
Centre an ArUco marker with the gantry (camera mounted on the carriage).

Each frame: find the marker, work out how far its centre is from the image centre (in pixels,
horizontal only) and send it to the Raspberry Pi running gantry_center_aruco.slx.
The Pi does the bang-bang control: move while |error| is large, stop when it is small.

UDP packet = 2 x int16, little-endian (the Pi's byte order): [visible (0/1), error_px]
  error_px = marker centre x - image centre x   (+ = marker is right of centre in the raw image)

Press q in the window to quit (sends "not visible" so the gantry stops).
"""

import socket
import struct
import time

import cv2
import cv2.aruco as aruco

PI_IP = "138.38.226.64"  # Pi's IP (on the Pi: hostname -I)
PI_PORT = 50001           # must match the UDP Receive block
TARGET_ID = None          # e.g. 35 to follow only that marker; None = the first marker seen
SEND_PERIOD = 0.1         # s; the Pi reads one packet per 0.05 s step, so don't send faster
ON_PX, OFF_PX = 40, 15    # only for drawing - the real thresholds are in the Simulink model
# Which camera to start with. Windows numbers cameras in the order it finds them, so the USB camera
# can be 0, 1 or 2 depending on the port / plug-in order. Press c in the window to switch to the
# next camera, then put the number shown on screen here.
CAMERA_INDEX = 0
# Blackout tolerance: the detector sometimes misses the marker for a frame or two (motion blur,
# glare). Without this, every miss sends "not visible" -> the gantry stops -> next frame starts
# again = jitter. Instead, for up to HOLD_TIME s after the last detection we keep sending the last
# good error, so the gantry carries on moving through short blackouts. Longer than that = really
# lost -> stop. Too long a hold = it can overshoot the centre while blind; 0.3-0.5 s is a good range.
HOLD_TIME = 0.4
# False-detection filter: with no marker in view, the detector occasionally "finds" a marker in
# random texture for a single frame. On its own that would make the gantry twitch, and HOLD_TIME
# would stretch the twitch to 0.4 s. So a NEW marker must be seen in CONFIRM_FRAMES frames in a row
# before we act on it. Once it is being tracked, single misses are covered by HOLD_TIME as before.
# Setting TARGET_ID above to your marker's id also rejects false detections with other ids.
CONFIRM_FRAMES = 3

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(visible, error_px):
    # Clamp to the int16 range so struct.pack can never fail
    error_px = max(-32768, min(32767, int(error_px)))
    sock.sendto(struct.pack("<hh", visible, error_px), (PI_IP, PI_PORT))


detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                               aruco.DetectorParameters())

def open_camera(index):
    """Open a camera at 1280x720 (the resolution the USB camera's Calibration.npz was made at)."""
    c = cv2.VideoCapture(index)
    if not c.isOpened():
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return c


cap = open_camera(CAMERA_INDEX)
if cap is None:
    raise SystemExit(f"Could not open camera {CAMERA_INDEX} - try another CAMERA_INDEX")
print(f"Using camera {CAMERA_INDEX} (press c in the window to switch)")

last_send = 0.0
last_seen_time = -1e9     # when the marker was last detected
last_error_px = 0         # error at that moment
streak = 0                # frames in a row with a detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame")
        break

    h, w = frame.shape[:2]
    cx = w // 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    visible, error_px, marker_id = 0, 0, None
    if ids is not None and len(ids) > 0:
        for ind, i in enumerate(ids.flatten()):
            if TARGET_ID is None or int(i) == TARGET_ID:
                # corners[ind] has shape (1, 4, 2): the centre is the mean of the 4 corners
                mx, my = corners[ind][0].mean(axis=0)
                visible, error_px, marker_id = 1, int(round(mx - cx)), int(i)
                cv2.circle(frame, (int(mx), int(my)), 8, (0, 0, 255), -1)
                break
        aruco.drawDetectedMarkers(frame, corners, ids)

    # False-detection filter + blackout tolerance (solution 1.1)
    now = time.perf_counter()
    streak = streak + 1 if visible else 0
    tracking = now - last_seen_time < HOLD_TIME   # a confirmed marker was seen very recently
    held = confirming = False
    if visible and (tracking or streak >= CONFIRM_FRAMES):
        last_seen_time, last_error_px = now, error_px   # confirmed detection -> act on it
    elif visible:
        visible, error_px, confirming = 0, 0, True        # new, not yet confirmed -> ignore for now
    elif tracking:
        visible, error_px, held = 1, last_error_px, True  # short blackout -> keep the last error

    if now - last_send >= SEND_PERIOD:
        send(visible, error_px)
        last_send = now

    # Centre line (green) and the bang-bang thresholds: stop zone (yellow), start zone (red)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
    for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
        cv2.line(frame, (cx - d, 0), (cx - d, h), col, 1)
        cv2.line(frame, (cx + d, 0), (cx + d, h), col, 1)
    if confirming:
        text = f"confirming marker {streak}/{CONFIRM_FRAMES} -> not moving yet"
    elif held:
        text = f"blackout {now - last_seen_time:.2f} s -> holding error {error_px:+d} px"
    elif visible:
        text = f"ID {marker_id}  error {error_px:+d} px"
    else:
        text = "no marker -> STOP"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"camera {CAMERA_INDEX}  (c = switch, q = quit)", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Not mirrored: the camera is on the gantry, so left/right on screen should match the real world
    cv2.imshow("gantry-centre", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        # Switch to the next camera that opens (tries up to 4 indices)
        send(0, 0)                      # stop the gantry while switching
        last_seen_time = -1e9           # don't carry an old detection over to the new camera
        cap.release()
        for step in range(1, 5):
            new_cap = open_camera((CAMERA_INDEX + step) % 4)
            if new_cap is not None:
                CAMERA_INDEX = (CAMERA_INDEX + step) % 4
                cap = new_cap
                break
        print(f"Switched to camera {CAMERA_INDEX}")

send(0, 0)  # stop the gantry before we stop sending
sock.close()
cap.release()
cv2.destroyAllWindows()
