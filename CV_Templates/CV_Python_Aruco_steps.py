"""
Tutorial 5 - ArUco section, steps 1 to 10.

Built from CV_Python_template: it detects ArUco markers (4x4_50 dictionary),
estimates the pose of every visible marker, draws its axes, and prints its id,
position (tvec, in mm) and orientation.

Works with OpenCV 4.7 or newer, including OpenCV 5 (ArucoDetector + cv2.solvePnP).
Press q in one of the windows to quit.
"""

import os
import logging
import socket   # Step 11: to send data to the Raspberry Pi
import time

import cv2
import cv2.aruco as aruco   # Step 1: import the ArUco library
import numpy as np

logging.basicConfig(level=logging.INFO)

# Folder this script is in, so the calibration file is found wherever you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Step 2: read the camera calibration
# Calibration.npz is our own camera's calibration (made with the notebook and
# copied here from workdir, tutorial step 12). It was made from 1280x720 photos,
# so the camera must also capture at 1280x720 (set below).
# ---------------------------------------------------------------------------
# Which camera to use: 0 = laptop camera, 1 = USB camera.
# A calibration only works for the camera it was made with, so if you switch
# camera, calibrate the new one and point CALIBRATION_FILE at its file.
CAMERA_INDEX = 1
CALIBRATION_FILE = 'Calibration.npz'
CALIBRATION_RESOLUTION = (1280, 720)
camera_calibration = np.load(os.path.join(SCRIPT_DIR, CALIBRATION_FILE))
CM = camera_calibration['CM']                 # camera matrix
dist_coef = camera_calibration['dist_coef']   # distortion coefficients
logging.info("Loaded camera calibration from %s", CALIBRATION_FILE)

# ---------------------------------------------------------------------------
# Step 3: ArUco dictionary, detection parameters and detector
# marker_size is the REAL width of your marker in mm: measure it!
# ---------------------------------------------------------------------------
marker_size = 100   # 10 cm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ---------------------------------------------------------------------------
# Step 7 (part 1): 3D position of the 4 marker corners, in mm, origin at the
# marker centre. Same order as the detector returns them:
# top-left, top-right, bottom-right, bottom-left
# ---------------------------------------------------------------------------
half = marker_size / 2
marker_points = np.array([[-half,  half, 0],
                          [ half,  half, 0],
                          [ half, -half, 0],
                          [-half, -half, 0]], dtype=np.float32)

# Print the poses at most every PRINT_PERIOD seconds so the terminal stays readable
PRINT_PERIOD = 0.5


# ---------------------------------------------------------------------------
# Step 11: send the marker poses to the Raspberry Pi over WiFi (UDP), based on
# Network_Templates/UDP_Send.py. Run Pi_Aruco_Receiver.py on the Pi.
# Find the Pi's IP address by typing  hostname -I  in a terminal on the Pi.
# Set SEND_TO_PI = False to run without a Pi.
# ---------------------------------------------------------------------------
SEND_TO_PI = True
PI_IP = "192.168.1.100"     # CHANGE to your Pi's IP address
PI_PORT = 50002             # must match UDP_PORT in Pi_Aruco_Receiver.py
SEND_PERIOD = 0.1           # seconds between messages (10 per second)

pi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
if SEND_TO_PI:
    logging.info("Sending marker poses to the Pi at %s:%s", PI_IP, PI_PORT)


def send_to_pi(poses):
    """Send "id,x,y,z;id,x,y,z" (mm), or "NONE" when no marker is visible."""
    if not poses:
        message = "NONE"
    else:
        message = ";".join(f"{marker_id},{tvec[0][0]:.1f},{tvec[1][0]:.1f},{tvec[2][0]:.1f}"
                           for marker_id, (tvec, rvec) in sorted(poses.items()))
    try:
        pi_sock.sendto(message.encode("utf-8"), (PI_IP, PI_PORT))
    except OSError as error:
        # e.g. not connected to WiFi: keep the camera running anyway
        logging.warning("Could not send to the Pi: %s", error)


def draw_text(img, text, pos, scale=0.6):
    """Green text with a black outline, readable on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)

# Select the first camera (0), the built-in one on a laptop
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    logging.error("Could not open camera %d", CAMERA_INDEX)
    raise SystemExit(1)
# Capture at the same resolution the calibration was made with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATION_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATION_RESOLUTION[1])
actual_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
logging.info("Camera %d opened at %dx%d", CAMERA_INDEX, *actual_resolution)
if actual_resolution != CALIBRATION_RESOLUTION:
    logging.warning("Camera resolution %s does not match the calibration %s: tvec will be wrong",
                    actual_resolution, CALIBRATION_RESOLUTION)

# Windows: the grey image and the ArUco result (step 6)
cv2.namedWindow("gray-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("aruco-image", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("gray-image", 0, 100)
cv2.moveWindow("aruco-image", 640, 100)

last_print = 0.0
last_send = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera %d", CAMERA_INDEX)
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 4: detect the markers in the grey image
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    out = frame.copy()
    now = time.perf_counter()
    print_now = (now - last_print) >= PRINT_PERIOD

    # poses[id] = (tvec, rvec) for every marker seen in this frame.
    # Use this dictionary for step 11 (sending commands to the Raspberry Pi).
    poses = {}

    # Only continue if at least one marker was found. len(corners) works in both
    # OpenCV 4 (ids is None) and OpenCV 5 (ids is an empty array).
    if len(corners) > 0:
        # Step 5: draw the detected markers on the frame
        out = aruco.drawDetectedMarkers(out, corners, ids)

        # Step 10: loop through every marker; ind is its index in corners,
        # marker_id is its ArUco id
        for ind, marker_id in enumerate(ids.flatten()):
            # Step 7 (part 2): pose of this marker from its 4 corners
            ok, rvec, tvec = cv2.solvePnP(marker_points, corners[ind], CM, dist_coef,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue

            # Step 8: draw the marker axes. They start at the marker centre, so they must be
            # longer than half the marker to be visible: use 1.5x the marker size, 4 px thick.
            # Red = x, green = y, blue = z (pointing out of the marker)
            out = cv2.drawFrameAxes(out, CM, dist_coef, rvec, tvec, marker_size * 1.5, 4)

            poses[int(marker_id)] = (tvec, rvec)

            # Step 9: tvec is a 3x1 array -> x = tvec[0][0], y = tvec[1][0], z = tvec[2][0] (mm)
            x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
            distance = np.linalg.norm(tvec)   # straight-line distance camera -> marker centre

            # Orientation: rotation vector -> rotation matrix -> angles in degrees
            R, _ = cv2.Rodrigues(rvec)
            angles = cv2.RQDecomp3x3(R)[0]    # rotation about x, y, z in degrees

            # Write the id, distance and x, y, z (mm) next to the marker in the image.
            # The label goes below the marker, or above it if there is no room below,
            # and is always kept inside the image.
            marker_corners = corners[ind].reshape(4, 2)
            lines = [f"id {marker_id}: {distance:.0f} mm",
                     f"x = {x:.0f} mm",
                     f"y = {y:.0f} mm",
                     f"z = {z:.0f} mm"]
            img_h, img_w = out.shape[:2]
            label_h = len(lines) * 22
            text_x = int(marker_corners[:, 0].min())
            text_y = int(marker_corners[:, 1].max()) + 20                 # below the marker
            if text_y + label_h > img_h:
                text_y = int(marker_corners[:, 1].min()) - label_h        # above the marker
            text_x = min(max(text_x, 5), img_w - 160)
            text_y = min(max(text_y, 20), img_h - label_h)
            for n, line in enumerate(lines):
                draw_text(out, line, (text_x, text_y + n * 22))

            if print_now:
                print(f"ID {marker_id:2d} | x={x:7.1f}  y={y:7.1f}  z={z:7.1f} mm"
                      f" | distance={distance:7.1f} mm"
                      f" | rot x={angles[0]:6.1f}  y={angles[1]:6.1f}  z={angles[2]:6.1f} deg")

    # Step 11: send this frame's markers to the Pi (at most every SEND_PERIOD seconds)
    if SEND_TO_PI and (now - last_send) >= SEND_PERIOD:
        send_to_pi(poses)
        last_send = now

    # Fixed panel in the top-left corner listing every visible marker, so the values
    # can always be read even when the label next to a marker is hard to see
    if poses:
        for n, (marker_id, (tvec, rvec)) in enumerate(sorted(poses.items())):
            tx, ty, tz = tvec[0][0], tvec[1][0], tvec[2][0]
            draw_text(out, f"ID {marker_id}: x={tx:.0f}  y={ty:.0f}  z={tz:.0f}  dist={np.linalg.norm(tvec):.0f} mm",
                      (10, 30 + n * 28), scale=0.7)
    else:
        draw_text(out, "No markers detected", (10, 30), scale=0.7)

    if print_now:
        if not poses:
            print("No markers detected")
        else:
            print("-" * 90)
        last_print = now

    # Step 6: show the results
    cv2.imshow("gray-image", gray)
    cv2.imshow("aruco-image", out)

    # Quit with q
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
pi_sock.close()
cv2.destroyAllWindows()
logging.info("Camera capture stopped")
