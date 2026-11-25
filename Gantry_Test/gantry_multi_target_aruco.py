"""
Multi-target ArUco shooting (camera mounted on the gantry carriage).

With several markers in view: pick the one closest to the laser, drive the gantry to it, let the
Pi fire the laser, then ignore that marker for good and go to the next closest. Stops when every
visible marker has been shot.

Pi side: gantry_home_center_laser.slx (recommended) or gantry_center_laser.slx - no changes needed.
Same UDP packet as gantry_center_aruco.py = 2 x int16, little-endian: [visible (0/1), error_px]
  error_px = marker centre x - image centre x   (+ = marker is right of centre in the raw image)

How the Pi fires (from the model): it moves while |error| > ON_PX, stops below OFF_PX, and turns
the laser on after it has been centred and stopped for 0.5 s. It keeps the laser on for as long as
the marker stays centred - there is no time limit and no feedback to the laptop. So this script
copies that timing on the laptop: once the laser should have been on for FIRE_TIME, it marks the
marker as shot and sends "not visible" for GAP_TIME, which makes the Pi switch the laser off,
then it sends the next target's error.

"Closest" = smallest horizontal pixel distance to the image centre, because the gantry only moves
in X (the laser servo is not used yet). Give every marker a DIFFERENT id - the id is how a shot
marker is remembered.

Keys (click the window first): q = quit (gantry stops, laser off), r = forget shot markers,
c = switch camera.
"""

import socket
import struct
import time

import cv2
import cv2.aruco as aruco

PI_IP = "138.38.226.64"  # Pi's IP (on the Pi: hostname -I)
PI_PORT = 50001           # must match the UDP Receive block
SEND_PERIOD = 0.1         # s; the Pi reads one packet per 0.05 s step, so don't send faster
CAMERA_INDEX = 0          # press c in the window to switch camera, then put the number shown here

# Must match the Simulink model (Bang-bang relay / Stateflow thresholds)
ON_PX, OFF_PX = 40, 15    # Pi starts moving above ON_PX, stops below OFF_PX
DWELL_TIME = 0.5          # s the Pi waits centred and stopped before the laser comes on (Dwell >= 10)

# Laptop-side shooting settings
FIRE_TIME = 1.0           # s the laser stays on per marker
LATENCY_MARGIN = 0.15     # s extra, for WiFi delay; raise it if a marker is marked shot too early
GAP_TIME = 0.3            # s of "not visible" after a shot so the Pi turns the laser off (6 Pi steps)

# Same detection filters as gantry_center_aruco.py, but per marker
HOLD_TIME = 0.4           # s: keep using the last error through short detection blackouts
CONFIRM_FRAMES = 3        # a NEW marker must be seen this many frames in a row before it counts


class TargetSelector:
    """Decides which marker to aim at and when it has been shot. No camera or network in here,
    so it can be tested with made-up detections.

    update({id: (u, v)}, cx, cy, now) -> (visible, error_px, status)
    """

    def __init__(self):
        self.shot_ids = set()
        self.forget()

    def forget(self):
        """Forget what is in view (e.g. after switching camera) but remember the shot markers."""
        self.streaks = {}          # id -> frames in a row it has been detected
        self.last_seen = {}        # id -> time of its last confirmed detection
        self.locked_id = None      # the marker we are going for
        self.locked_error = 0      # its last error (re-sent during short blackouts)
        self.centred_since = None  # when the Pi (we think) stopped on the locked marker
        self.pi_moving = False     # our copy of the Pi's bang-bang relay state
        self.gap_until = 0.0

    def reset_shots(self):
        self.shot_ids.clear()
        self.locked_id = None
        self.centred_since = None

    def _confirm(self, detections, now):
        """Return the detections that pass the false-detection filter."""
        self.streaks = {i: self.streaks.get(i, 0) + 1 for i in detections}
        confirmed = {}
        for i, uv in detections.items():
            tracked = now - self.last_seen.get(i, -1e9) < HOLD_TIME
            if tracked or self.streaks[i] >= CONFIRM_FRAMES:
                confirmed[i] = uv
                self.last_seen[i] = now
        return confirmed

    def _stop(self, state, **extra):
        # Sending "not visible": the Pi stops (Idling) and the laser goes off
        self.pi_moving = False
        self.centred_since = None
        return 0, 0, dict(state=state, locked=None, error=0, held=False, **extra)

    def update(self, detections, cx, cy, now):
        confirmed = self._confirm(detections, now)

        if now < self.gap_until:
            return self._stop("GAP")

        # Keep going for the locked marker; drop it if it has been gone longer than HOLD_TIME
        held = False
        if self.locked_id is not None:
            if self.locked_id in confirmed:
                self.locked_error = int(round(confirmed[self.locked_id][0] - cx))
            elif now - self.last_seen.get(self.locked_id, -1e9) < HOLD_TIME:
                held = True
            else:
                self.locked_id = None
                self.centred_since = None

        # Nothing locked: pick the closest marker that has not been shot
        if self.locked_id is None:
            candidates = [i for i in confirmed if i not in self.shot_ids]
            if not candidates:
                return self._stop("DONE" if confirmed else "NO TARGET")
            self.locked_id = min(candidates, key=lambda i: (abs(confirmed[i][0] - cx),
                                                            abs(confirmed[i][1] - cy)))
            self.locked_error = int(round(confirmed[self.locked_id][0] - cx))
            self.centred_since = None

        err = self.locked_error

        # Copy the Pi's relay: moving above ON_PX, stopped below OFF_PX, unchanged in between
        if abs(err) > ON_PX:
            self.pi_moving = True
        elif abs(err) < OFF_PX:
            self.pi_moving = False

        if self.pi_moving:
            self.centred_since = None
            state, elapsed = "MOVING", 0.0
        else:
            if self.centred_since is None:
                self.centred_since = now
            elapsed = now - self.centred_since
            laser_on_at = DWELL_TIME + LATENCY_MARGIN
            if elapsed >= laser_on_at + FIRE_TIME:
                shot = self.locked_id
                self.shot_ids.add(shot)
                self.locked_id = None
                self.gap_until = now + GAP_TIME
                return self._stop("SHOT", shot=shot)
            state = "LASER ON" if elapsed >= laser_on_at else "SETTLING"

        return 1, err, dict(state=state, locked=self.locked_id, error=err, held=held,
                            elapsed=elapsed)


detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                               aruco.DetectorParameters())


def detect_targets(gray):
    """Return {marker_id: corners (4x2)}. The only part to change for colour circles later."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:   # None in OpenCV 4, empty array in OpenCV 5
        return {}
    return {int(i): c[0] for i, c in zip(ids.flatten(), corners)}


def open_camera(index):
    """Open a camera at 1280x720 (the resolution the USB camera's Calibration.npz was made at)."""
    c = cv2.VideoCapture(index)
    if not c.isOpened():
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return c


def draw_text(img, text, pos, scale=0.8, colour=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


def main():
    global CAMERA_INDEX
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(visible, error_px):
        # Clamp to the int16 range so struct.pack can never fail
        error_px = max(-32768, min(32767, int(error_px)))
        sock.sendto(struct.pack("<hh", visible, error_px), (PI_IP, PI_PORT))

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(f"Could not open camera {CAMERA_INDEX} - try another CAMERA_INDEX")
    print(f"Using camera {CAMERA_INDEX} (press c in the window to switch)")

    selector = TargetSelector()
    last_send = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        markers = detect_targets(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        centres = {i: c.mean(axis=0) for i, c in markers.items()}

        now = time.perf_counter()
        visible, error_px, status = selector.update(centres, cx, cy, now)
        if status["state"] == "SHOT":
            print(f"Shot marker {status['shot']}  (shot so far: {sorted(selector.shot_ids)})")

        # Send every SEND_PERIOD, and straight away when we stop so the laser goes off promptly
        if now - last_send >= SEND_PERIOD or status["state"] == "SHOT":
            send(visible, error_px)
            last_send = now

        # ---- Drawing ----
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
        for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
            cv2.line(frame, (cx - d, 0), (cx - d, h), col, 1)
            cv2.line(frame, (cx + d, 0), (cx + d, h), col, 1)
        for i, c in markers.items():
            u, v = (int(x) for x in centres[i])
            if i in selector.shot_ids:
                colour, label = (128, 128, 128), f"{i} SHOT"
            elif i == status["locked"]:
                colour, label = (0, 0, 255), f"{i} TARGET"
            else:
                colour, label = (255, 200, 0), f"{i}"
            cv2.polylines(frame, [c.astype(int)], True, colour, 3)
            cv2.circle(frame, (u, v), 6, colour, -1)
            draw_text(frame, label, (u + 10, v - 10), 0.7, colour)
            if i == status["locked"]:
                cv2.line(frame, (cx, v), (u, v), colour, 2)

        state = status["state"]
        if state in ("SETTLING", "LASER ON"):
            line = f"{state} {status['elapsed']:.1f} s  ID {status['locked']}  error {error_px:+d} px"
        elif state == "MOVING":
            line = f"MOVING to ID {status['locked']}  error {error_px:+d} px"
        elif state == "DONE":
            line = "DONE - all visible markers shot (r = reset)"
        elif state == "SHOT":
            line = f"SHOT marker {status['shot']}"
        else:
            line = state
        if status["held"]:
            line += "  (blackout, holding)"
        draw_text(frame, line, (10, 35), 0.9, (0, 0, 255) if state == "LASER ON" else (0, 255, 0))
        draw_text(frame, f"shot: {sorted(selector.shot_ids) or '-'}", (10, 70), 0.7)
        draw_text(frame, f"camera {CAMERA_INDEX}  (q = quit, r = reset shots, c = switch camera)",
                  (10, h - 15), 0.7, (255, 255, 255))

        # Not mirrored: the camera is on the gantry, so left/right on screen matches the real world
        cv2.imshow("gantry-multi-target", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selector.reset_shots()
            print("Shot list cleared")
        if key == ord("c"):
            send(0, 0)                  # stop the gantry while switching
            selector.forget()           # don't carry detections over to the new camera
            cap.release()
            for step in range(1, 5):
                new_cap = open_camera((CAMERA_INDEX + step) % 4)
                if new_cap is not None:
                    CAMERA_INDEX = (CAMERA_INDEX + step) % 4
                    cap = new_cap
                    break
            print(f"Switched to camera {CAMERA_INDEX}")

    send(0, 0)  # stop the gantry and turn the laser off before we stop sending
    sock.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
