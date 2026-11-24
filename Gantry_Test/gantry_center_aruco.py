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
ON_PX, OFF_PX = 40, 15    # only for drawing - the real thresholds are in the Simulink Relay block

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(visible, error_px):
    # Clamp to the int16 range so struct.pack can never fail
    error_px = max(-32768, min(32767, int(error_px)))
    sock.sendto(struct.pack("<hh", visible, error_px), (PI_IP, PI_PORT))


detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                               aruco.DetectorParameters())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera 0")
# A smaller image = higher frame rate = less lag in the control loop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_send = 0.0
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

    if time.perf_counter() - last_send >= SEND_PERIOD:
        send(visible, error_px)
        last_send = time.perf_counter()

    # Centre line (green) and the bang-bang thresholds: stop zone (yellow), start zone (red)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
    for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
        cv2.line(frame, (cx - d, 0), (cx - d, h), col, 1)
        cv2.line(frame, (cx + d, 0), (cx + d, h), col, 1)
    text = f"ID {marker_id}  error {error_px:+d} px" if visible else "no marker -> STOP"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Not mirrored: the camera is on the gantry, so left/right on screen should match the real world
    cv2.imshow("gantry-centre", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

send(0, 0)  # stop the gantry before we stop sending
sock.close()
cap.release()
cv2.destroyAllWindows()
