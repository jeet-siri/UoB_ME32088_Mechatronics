"""
This script captures camera frames, displays color and grayscale images, and reports the frame rate.

@author: ioannisgeorgilas
"""

# This is the vision library OpenCV
import cv2
import logging
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities
import time
# ArUco step 1: import the ArUco library
import cv2.aruco as aruco
from pathlib import Path
# ArUco step 11: network library for sending UDP commands to the Raspberry Pi
import socket

logging.basicConfig(level=logging.INFO)

# How to flip the images for DISPLAY only: 1 = mirror (left/right), 0 = upside down, -1 = both.
# Detection runs on the unflipped frame, because a mirrored ArUco marker is not a valid marker.
FLIP_CODE = 1

# ArUco step 11: UDP settings for the Raspberry Pi (Simulink "UDP Receive" block)
PI_IP = "138.38.227.237"  # your Pi's IP (on the Pi: hostname -I)
PI_PORT = 50001           # must match the UDP Receive block's Local IP port
STEPPER_ID = 35           # this marker turns the stepper motor
LASER_ID = 49             # this marker turns the laser on
# Each frame we send 2 bytes: [stepper_on, laser_on], each 0 or 1.
# We send EVERY frame (not only on change) because the UDP Receive block keeps its last
# value: if we stopped sending, the Pi would stay in the last state.
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Send at most every 0.1 s. The Simulink UDP Receive block reads one packet per 0.05 s step;
# sending faster than that would queue packets up and the Pi would lag further and further behind.
SEND_PERIOD = 0.1
last_send = 0.0

# ArUco step 2: load the camera calibration (looked up next to this script)
# (step 12: uses your own Calibration.npz if it is next to this script, otherwise the generic sample)
calibration_path = Path(__file__).resolve().with_name("Calibration.npz")
if not calibration_path.exists():
    calibration_path = calibration_path.with_name("Sample_Calibration.npz")
    logging.warning("Calibration.npz not found - using the generic Sample_Calibration.npz (distances will be wrong)")
camera_calibration = np.load(calibration_path)
CM = camera_calibration['CM']                # camera matrix
dist_coef = camera_calibration['dist_coef']  # distortion coefficients
logging.info("Loaded camera calibration from %s", calibration_path)

# ArUco step 3: dictionary, detection parameters and detector
marker_size = 40  # real size of the printed marker in mm - measure yours!
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ArUco step 7 (part A): 3D corners of the marker in mm, origin at its centre.
# Order matches the detector: top-left, top-right, bottom-right, bottom-left
half = marker_size / 2
marker_points = np.array([[-half, half, 0], [half, half, 0],
                          [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

# Select the first camera (0) that is connected to the machine
# in Laptops should be the built-in camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Could not open camera 0")
    raise SystemExit(1)
logging.info("Camera 0 opened")

# Set the width and height of the camera to 1920x1080
cap.set(3,1920)
cap.set(4,1080)
logging.info(
    "Camera resolution: %dx%d",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

#Create two opencv named windows
cv2.namedWindow("frame-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("gray-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("canny-image", cv2.WINDOW_AUTOSIZE)
#Position the windows next to each other
cv2.moveWindow("frame-image",0,100)
cv2.moveWindow("gray-image",640,100)
cv2.moveWindow("canny-image", 1280, 100)
cv2.createTrackbar("low", "canny-image", 100, 500, lambda x: None)
cv2.createTrackbar("high", "canny-image", 200, 500, lambda x: None)

# Execute this continuously
last_fps_log = time.perf_counter()
while(True):
    
    # Start the performance clock
    start = time.perf_counter()
    
    # Capture current frame from the camera
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera 0")
        break
    
    # Convert the image from the camera to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lo = cv2.getTrackbarPos("low", "canny-image")
    hi = cv2.getTrackbarPos("high", "canny-image")
    canny = cv2.Canny(gray, lo, hi)
    canny_col = cv2.Canny(frame, lo, hi)
    canny_blur = cv2.Canny(blur, 100, 200)

    # ArUco step 4: detect markers in the gray image
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # ArUco step 5: draw the detected markers on a copy of the frame
    out = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    # ArUco steps 7 (part B), 8, 9 and 10: pose, axes and printout for EVERY detected marker.
    # Only run if at least one marker was found, otherwise corners[...] would crash.
    # poses maps marker id -> tvec (mm), so later code (e.g. step 11) can look markers up by id.
    poses = {}
    if len(corners) > 0:
        # corners and ids have one entry per marker in matching order, so corners[ind] belongs to id.
        # ids.flatten() works for both OpenCV 4 (column array) and 5 (flat array).
        for ind, id in enumerate(ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(marker_points, corners[ind], CM, dist_coef,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                out = cv2.drawFrameAxes(out, CM, dist_coef, rvec, tvec, 10)
                poses[int(id)] = tvec

                # tvec is 3x1, so x/y/z are tvec[0][0], tvec[1][0], tvec[2][0].
                print(f"ID {id}: "
                      f"x={tvec[0][0]:.1f} y={tvec[1][0]:.1f} z={tvec[2][0]:.1f} mm")

    # ArUco step 11: decide the commands from the visible ids and send them to the Pi.
    # No marker visible -> both 0, so the stepper and the laser turn off.
    visible_ids = set(int(i) for i in ids.flatten()) if len(corners) > 0 else set()
    stepper_on = 1 if STEPPER_ID in visible_ids else 0
    laser_on = 1 if LASER_ID in visible_ids else 0
    if time.perf_counter() - last_send >= SEND_PERIOD:
        sock.sendto(bytes([stepper_on, laser_on]), (PI_IP, PI_PORT))
        last_send = time.perf_counter()

    # ArUco step 6: show the result (status text added after the flip so it isn't mirrored)
    shown = cv2.flip(out, FLIP_CODE)
    cv2.putText(shown, f"STEPPER {'ON' if stepper_on else 'OFF'}  LASER {'ON' if laser_on else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('aruco-image', shown)

    # Display the original frame in a window
    cv2.imshow('frame-image', cv2.flip(frame, FLIP_CODE))

    
    # Display the grey image in another window
    cv2.imshow('gray-image', cv2.flip(gray, FLIP_CODE))
    cv2.imshow('canny-colour', cv2.flip(canny_col, FLIP_CODE))
    cv2.imshow('canny-image', cv2.flip(canny, FLIP_CODE))
    # Stop the performance counter
    end = time.perf_counter()
    
    # Report the frame rate once per second rather than on every frame
    if end - last_fps_log >= 1.0:
        logging.info("Camera FPS: %.2f", 1 / (end - start))
        last_fps_log = end

    # If the button q is pressed in one of the windows 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # Exit the While loop
        break
    

# Safety: tell the Pi to switch everything off before we stop sending
sock.sendto(bytes([0, 0]), (PI_IP, PI_PORT))
sock.close()

# When everything done, release the capture
cap.release()
# close all windows
cv2.destroyAllWindows()
logging.info("Camera capture stopped")
# exit the kernel
exit(0)