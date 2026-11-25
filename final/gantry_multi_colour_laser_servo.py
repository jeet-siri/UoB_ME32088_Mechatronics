"""
Multi-colour circle targeting with live mask tuning: find the coloured shapes (4: yellow, orange,
green, blue), log where they are,
then shoot them in PRIORITY order and stop.
Runs with gantry_home_center_laser_servo.slx on the Pi (homing, 7900-step limit, laser dwell, servo).

Combines:
  - CV_Templates/CV_Python_MultiColour.py (teammate): 5 colours, HSV trackbars, click-to-measure,
    all-masks view, colours.json
  - gantry_multi_target_aruco.py (teammate): TargetSelector - drives to one target, waits for the Pi
    to fire, then a short "not visible" gap so the laser turns off
  - gantry_yellow_circle_laser_servo.py: circle filter (incl. the corner check that rejects squares),
    locked camera settings, servo aiming from servo_calibration.json, -60 px X offset

How a target is chosen (RUN mode):
  1. LOG: a circle only becomes a target once it has been seen in LOG_HIT_RATIO of the frames for
     LOG_TIME s. Random flickers / reflections never get there. Its FLOOR position is then remembered
     (the tracker measures how far the picture shifts = how far the camera moved), so it can be found
     again even when it is out of view. ONE_PER_COLOUR: the first real circle of each colour is
     logged, any other circle of that colour is ignored.
  2. SHOOT: the logged targets in PRIORITY colour order. If the next one is out of view, the gantry
     drives towards its remembered position until it comes back into view.
  3. RECHECK: after every shot the gantry stops for RECHECK_TIME s before the next target.
  4. DONE: after TOTAL_TARGETS shots it stops for good (r starts again).

Starts in TUNE mode: the gantry holds still and the laser stays off. Tune the masks, then g = RUN.
Windows:
  gantry-multi-colour  camera view: candidate circles (thin, with % logged), logged targets
                       (thick, P1..P4), shot (grey). CLICK to print the HSV the camera sees there.
  mask                 the colour picked with the "colour" trackbar + its HSV trackbars
  all-masks            every colour's mask side by side - each circle white in ONE mask only
Keys (click a window first):
  g = TUNE <-> RUN   s = save colour ranges   r = reset (forget logged + shot targets)
  a / d = nudge X offset   c = switch camera   q = quit (gantry stops, laser off)
UDP packet = 3 x int16, little-endian: [visible (0/1), error_px, servo angle x 10]
"""

import json
import os
import socket
import struct
import time

import cv2
import numpy as np

# The shared code (servo aiming / circle filter, teammate's TargetSelector) stays in Gantry_Test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gantry_Test"))

import gantry_yellow_circle_laser_servo as single           # servo aiming, circle filter, camera
from gantry_multi_target_aruco import TargetSelector, ON_PX, OFF_PX

PI_IP, PI_PORT, SEND_PERIOD = single.PI_IP, single.PI_PORT, single.SEND_PERIOD
CAMERA_INDEX = single.CAMERA_INDEX

# Shoot in this colour order. ONLY these colours are searched for (pink removed: too hard to see
# reliably - its range stays in colours.json, add "pink" back here to use it again).
PRIORITY = ["yellow", "orange", "blue", "green"]   # green last: it takes longest to be found/logged
TOTAL_TARGETS = 4         # one shape per colour above: stop after this many shots
ONE_PER_COLOUR = True     # one shape per colour. False if two shapes can share a colour
LOG_TIME = 2.0            # s a circle must be tracked before it is logged as a target
LOG_HIT_RATIO = 0.8       # ... and seen in at least this fraction of the frames in that time
RECHECK_TIME = 1.0        # s the gantry stops after each shot before going to the next target
SEARCH_TIMEOUT = 15.0     # s to look for an out-of-view target before skipping it

# name -> BGR colour used for drawing (the HSV ranges come from colours.json)
DRAW = {"yellow": (0, 242, 255), "orange": (34, 148, 247), "green": (64, 197, 141),
        "blue": (227, 178, 4), "pink": (147, 54, 238)}
COLOURS_FILE = os.path.join(single.CV, "colours.json")
with open(COLOURS_FILE) as f:
    RANGES = {name: [list(v["low"]), list(v["high"])] for name, v in json.load(f).items()}
NAMES = [n for n in PRIORITY if n in RANGES]   # colours searched for, tuned and shown


def save_colours():
    with open(COLOURS_FILE, "w") as f:
        # save EVERY colour (incl. unused pink) so the team's other scripts keep their ranges
        json.dump({n: {"low": RANGES[n][0], "high": RANGES[n][1]} for n in RANGES}, f, indent=2)
    print(f"Saved colour ranges to {COLOURS_FILE}")


def make_mask(hsv, low, high):
    """Teammate's mask: handles hue wrap-around (H low > H high) for orange/red."""
    if low[0] <= high[0]:
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
    else:
        mask = cv2.inRange(hsv, np.array(low), np.array([179, high[1], high[2]])) | \
               cv2.inRange(hsv, np.array([0, low[1], low[2]]), np.array(high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, single.KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, single.KERNEL)


def find_circles(mask):
    """EVERY circle-like blob (same filters as the single-circle script, squares rejected)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < single.MIN_AREA or len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        w, h = ellipse[1]
        ellipse_area = np.pi * w * h / 4
        if ellipse_area <= 0 or area / ellipse_area < single.MIN_CIRCULARITY:
            continue
        if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) < single.MIN_CORNERS:
            continue
        found.append(ellipse)
    return found


class CircleTracker:
    """Labels circles ("yellow#3") and keeps the labels from frame to frame.

    update([(colour, u, v), ...], now) -> {label: (u, v)} of circles seen THIS frame (lens-corrected px)

    Camera motion: when the gantry moves, every circle shifts sideways in the image by the same
    amount. The median shift of the matched circles goes into cam_u, so floor_u = u + cam_u stays the
    same for a circle that is not moving. (If NO circle is visible while the gantry moves, that motion
    is missed - normally something is in view.)
    """
    MATCH_PX = 80          # max movement of one circle between frames to count as the same circle
    TRACK_TIMEOUT = 0.5    # s a label is kept while unseen (covers detection blackouts)

    def __init__(self):
        self.tracks = {}       # label -> dict(colour, u, v, seen, born, hits, frames)
        self.counts = {}
        self.cam_u = 0.0

    def update(self, detections, now):
        unmatched = list(detections)
        shifts, current = [], {}
        for label, t in sorted(self.tracks.items(), key=lambda kv: -kv[1]["seen"]):
            t["frames"] += 1
            best, best_d = None, self.MATCH_PX
            for d in unmatched:
                if d[0] == t["colour"]:
                    dist = np.hypot(d[1] - t["u"], d[2] - t["v"])
                    if dist < best_d:
                        best, best_d = d, dist
            if best is not None:
                unmatched.remove(best)
                shifts.append(best[1] - t["u"])
                t.update(u=best[1], v=best[2], seen=now, hits=t["hits"] + 1)
                current[label] = (best[1], best[2])
        if shifts:   # picture moved by s -> camera moved by -s
            self.cam_u -= float(np.median(shifts))
        for colour, u, v in unmatched:
            self.counts[colour] = self.counts.get(colour, 0) + 1
            label = f"{colour}#{self.counts[colour]}"
            self.tracks[label] = dict(colour=colour, u=u, v=v, seen=now, born=now, hits=1, frames=1)
            current[label] = (u, v)
        self.tracks = {k: t for k, t in self.tracks.items() if now - t["seen"] < self.TRACK_TIMEOUT}
        return current

    def floor_u(self, u):
        return u + self.cam_u

    def log_progress(self, label, now):
        """0..1: how close a circle is to being logged (needs LOG_TIME s at LOG_HIT_RATIO)."""
        t = self.tracks[label]
        if t["hits"] < LOG_HIT_RATIO * t["frames"]:
            return 0.0
        return min(1.0, (now - t["born"]) / LOG_TIME)


class Mission:
    """Logs the targets, then shoots them one by one in priority order, then stops.

    update(circles, target_u, cy, now) -> (visible, error_px, v_for_servo or None, status)
    circles = {label: (u, v)} from CircleTracker.update.
    """
    SAME_PLACE_PX = 60     # a circle this close to a logged target's floor position (same colour) is it

    def __init__(self, tracker):
        self.tr = tracker
        self.reset()

    def reset(self):
        self.logged = []              # dict(key, colour, fu, v, shot, skipped)
        self.shots = 0
        self.pause_until = 0.0
        self.current, self.current_since = None, 0.0
        self.selector = TargetSelector()

    def _rank(self, colour):
        return PRIORITY.index(colour) if colour in PRIORITY else len(PRIORITY)

    def _find(self, colour, fu, v):
        for L in self.logged:
            if L["colour"] == colour and np.hypot(L["fu"] - fu, L["v"] - v) < self.SAME_PLACE_PX:
                return L
        return None

    def live_position(self, L, circles):
        """Image position of logged target L: from the camera if it is in view, else predicted."""
        for lb, (u, v) in circles.items():
            if self.tr.tracks[lb]["colour"] == L["colour"] and \
                    np.hypot(self.tr.floor_u(u) - L["fu"], v - L["v"]) < self.SAME_PLACE_PX:
                return (u, v), True
        return (L["fu"] - self.tr.cam_u, L["v"]), False

    def log(self, circles, now):
        for lb, (u, v) in circles.items():
            colour = self.tr.tracks[lb]["colour"]
            if self.tr.log_progress(lb, now) < 1.0:
                continue
            fu = self.tr.floor_u(u)
            L = self._find(colour, fu, v)
            if L is not None:
                L["fu"], L["v"] = fu, v          # refine the remembered position
            elif not (ONE_PER_COLOUR and any(x["colour"] == colour for x in self.logged)):
                self.logged.append(dict(key=f"{colour}-{len(self.logged) + 1}", colour=colour,
                                        fu=fu, v=v, shot=False, skipped=False))
                print(f"Logged {colour} target ({len(self.logged)} logged)")

    def _stop(self, state, **extra):
        return 0, 0, None, dict(state=state, locked=None, held=False, **extra)

    def update(self, circles, target_u, cy, now):
        self.log(circles, now)
        if self.shots >= TOTAL_TARGETS:
            return self._stop("DONE")
        if now < self.pause_until:
            return self._stop("RECHECK", left=self.pause_until - now)
        todo = [L for L in self.logged if not L["shot"] and not L["skipped"]]
        if not todo:
            return self._stop("SEARCHING")
        cur = min(todo, key=lambda L: (self._rank(L["colour"]), abs(L["fu"] - self.tr.cam_u - target_u)))
        if cur is not self.current:
            self.current, self.current_since = cur, now
            self.selector.forget()
        (u, v), in_view = self.live_position(cur, circles)
        if in_view:
            self.current_since = now                 # the search timeout only runs while out of view
        elif now - self.current_since > SEARCH_TIMEOUT:
            cur["skipped"] = True
            print(f"Could not find {cur['key']} again - skipped")
            return self._stop("SEARCHING")
        visible, err, st = self.selector.update({cur["key"]: (u, v)}, target_u, cy, now)
        st["target"], st["in_view"] = cur["key"], in_view
        if st["state"] == "SHOT":
            cur["shot"] = True
            self.shots += 1
            self.current = None
            self.pause_until = now + RECHECK_TIME
            print(f"Shot {cur['key']}  ({self.shots}/{TOTAL_TARGETS})")
        return visible, err, v, st


# ---------------- tuning window (teammate's trackbars) ----------------
HSV_BARS = [("H low", 0, 0, 179), ("H high", 1, 0, 179), ("S low", 0, 1, 255),
            ("S high", 1, 1, 255), ("V low", 0, 2, 255), ("V high", 1, 2, 255)]


def load_bars(name):
    for bar, which, ch, _ in HSV_BARS:
        cv2.setTrackbarPos(bar, "mask", RANGES[name][which][ch])


def store_bars(name):
    for bar, which, ch, _ in HSV_BARS:
        RANGES[name][which][ch] = cv2.getTrackbarPos(bar, "mask")


latest_hsv, clicked = None, None


def on_click(event, x, y, flags, param):
    """Print the median HSV of an 11x11 patch where you click."""
    global clicked
    if event != cv2.EVENT_LBUTTONDOWN or latest_hsv is None:
        return
    hh, ww = latest_hsv.shape[:2]
    x, y = min(max(x, 5), ww - 6), min(max(y, 5), hh - 6)
    h, s, v = np.median(latest_hsv[y - 5:y + 6, x - 5:x + 6].reshape(-1, 3), axis=0).astype(int)
    clicked = (x, y, f"H={h} S={s} V={v}")
    print(f"Clicked ({x}, {y}): H={h}  S={s}  V={v}")


def main():
    global CAMERA_INDEX, latest_hsv
    fit = single.load_fit()
    print("Servo calibrated" if fit else "Servo NOT calibrated - it stays at SERVO_ZERO")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(visible, error_px, servo_deg):
        error_px = max(-32768, min(32767, int(error_px)))
        sock.sendto(struct.pack("<hhh", visible, error_px, int(round(servo_deg * 10))), (PI_IP, PI_PORT))

    cap = single.open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(f"Could not open camera {CAMERA_INDEX} - try another CAMERA_INDEX")

    cv2.namedWindow("gantry-multi-colour", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gantry-multi-colour", 960, 540)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", 640, 360)
    cv2.namedWindow("all-masks", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("gantry-multi-colour", 0, 0)
    cv2.moveWindow("mask", 970, 0)
    cv2.moveWindow("all-masks", 0, 600)
    cv2.createTrackbar("colour", "mask", 0, len(NAMES) - 1, lambda _: None)
    for bar, which, ch, mx in HSV_BARS:
        cv2.createTrackbar(bar, "mask", RANGES[NAMES[0]][which][ch], mx, lambda _: None)
    cv2.setMouseCallback("gantry-multi-colour", on_click)

    tracker = CircleTracker()
    mission = Mission(tracker)
    running, selected = False, 0
    servo, last_send = single.SERVO_ZERO, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break
        h, w = frame.shape[:2]
        offset = single.TARGET_U_OFFSET_PX
        target_u, cy = w // 2 + offset, h // 2

        # Trackbars <-> selected colour (switching colour loads its values into the bars)
        sel = cv2.getTrackbarPos("colour", "mask")
        if sel != selected:
            selected = sel
            load_bars(NAMES[selected])
        else:
            store_bars(NAMES[selected])

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        latest_hsv = hsv
        masks, detections, ellipses = {}, [], {}
        for name in NAMES:
            masks[name] = make_mask(hsv, *RANGES[name])
            for e in find_circles(masks[name]):      # every round blob, squares rejected
                u, v = cv2.undistortPoints(np.array([[e[0]]], dtype=np.float32), single.CM, single.DIST,
                                           P=single.CM)[0, 0]
                detections.append((name, float(u), float(v)))
                ellipses[(name, float(u), float(v))] = e

        now = time.perf_counter()
        circles = tracker.update(detections, now)                  # label -> (u, v)
        if running:
            visible, error_px, aim_v, status = mission.update(circles, target_u, cy, now)
            if aim_v is not None:
                servo = single.servo_for(float(aim_v), fit)
        else:
            visible, error_px, status = 0, 0, {"state": "TUNE", "held": False}
        if now - last_send >= SEND_PERIOD or status["state"] == "SHOT":
            send(visible, error_px, servo)
            last_send = now

        # ---- main window ----
        out = frame.copy()
        cv2.line(out, (target_u, 0), (target_u, h), (0, 255, 0), 2)
        for d, col in ((OFF_PX, (0, 255, 255)), (ON_PX, (0, 0, 255))):
            cv2.line(out, (target_u - d, 0), (target_u - d, h), col, 1)
            cv2.line(out, (target_u + d, 0), (target_u + d, h), col, 1)
        # every circle seen this frame: thin outline + how far it is to being logged
        for lb, uv in circles.items():
            colour = lb.split("#")[0]
            e = ellipses[(colour, *uv)]
            cv2.ellipse(out, e, DRAW.get(colour, (255, 255, 255)), 1)
            p = tracker.log_progress(lb, now)
            if p < 1.0 and running:
                single.draw_text(out, f"? {int(p * 100)}%", (int(e[0][0]) + 10, int(e[0][1]) + 25), 0.5,
                                 DRAW.get(colour, (255, 255, 255)))
        # logged targets: thick ring at their (live or predicted) position
        for L in mission.logged:
            (u, v), in_view = mission.live_position(L, circles)
            colour = (128, 128, 128) if L["shot"] or L["skipped"] else DRAW.get(L["colour"], (255, 255, 255))
            tag = "SHOT" if L["shot"] else "SKIPPED" if L["skipped"] else \
                  f"P{mission._rank(L['colour']) + 1}" + ("" if in_view else " (out of view)")
            if L is mission.current:
                tag = "TARGET" + ("" if in_view else " (driving to it)")
            pu = int(min(max(u, 10), w - 10))      # keep out-of-view targets on the screen edge
            cv2.circle(out, (pu, int(v)), 30, colour, 6 if L is mission.current else 3)
            single.draw_text(out, f"{L['colour']} {tag}", (pu + 35, int(v) - 10), 0.8, colour)

        state = status["state"]
        if state == "TUNE":
            line = "TUNE - gantry held, laser off. Tune masks, then g = RUN"
        elif state in ("SETTLING", "LASER ON"):
            line = f"{state} {status['elapsed']:.1f} s  {status['target']}  servo {servo:.1f}"
        elif state == "MOVING":
            line = f"MOVING to {status['target']}  error {error_px:+d} px  servo {servo:.1f}"
        elif state == "RECHECK":
            line = f"RECHECK - stopped {status['left']:.1f} s"
        elif state == "SEARCHING":
            line = "SEARCHING - waiting for targets to be logged"
        elif state == "DONE":
            line = f"DONE - {TOTAL_TARGETS} targets shot, stopped (r = start again)"
        else:
            line = state
        if status.get("held"):
            line += "  (blackout, holding)"
        single.draw_text(out, line, (10, 40), 1.0, (0, 0, 255) if state == "LASER ON" else (0, 255, 0))
        single.draw_text(out, f"logged {len(mission.logged)}   shot {mission.shots}/{TOTAL_TARGETS}   "
                              f"priority: {' > '.join(PRIORITY)}", (10, 80), 0.8)
        if clicked:
            cv2.drawMarker(out, clicked[:2], (255, 255, 255), cv2.MARKER_CROSS, 25, 2)
            single.draw_text(out, clicked[2], (clicked[0] + 15, clicked[1] + 30), 0.9, (255, 255, 255))
        single.draw_text(out, f"camera {CAMERA_INDEX} | X offset {offset:+d} px | "
                              f"g run/tune  s save  r reset  a/d offset  c camera  q quit",
                         (10, h - 15), 0.7, (255, 255, 255))
        cv2.imshow("gantry-multi-colour", out)

        # ---- mask windows ----
        name = NAMES[selected]
        mask_view = cv2.cvtColor(masks[name], cv2.COLOR_GRAY2BGR)
        single.draw_text(mask_view, f"editing: {name}  {tuple(RANGES[name][0])} - {tuple(RANGES[name][1])}",
                         (10, 50), 1.2, DRAW.get(name, (255, 255, 255)))
        cv2.imshow("mask", mask_view)
        tiles = []
        for n in NAMES:
            tile = cv2.cvtColor(cv2.resize(masks[n], (256, 144)), cv2.COLOR_GRAY2BGR)
            col = DRAW.get(n, (255, 255, 255))
            cv2.rectangle(tile, (0, 0), (255, 143), col, 2)
            count = sum(1 for lb in circles if lb.split("#")[0] == n)
            single.draw_text(tile, f"{n}{f' x{count}' if count else ''}", (8, 24), 0.6, col)
            tiles.append(tile)
        cv2.imshow("all-masks", np.hstack(tiles))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            running = not running
            mission.selector.forget()
            print("RUN - logging and shooting targets" if running else "TUNE - gantry held, laser off")
        if key == ord("s"):
            save_colours()
        if key == ord("r"):
            mission.reset()
            print("Reset - logged and shot targets forgotten")
        if key in (ord("a"), ord("d")):
            single.TARGET_U_OFFSET_PX += -10 if key == ord("a") else 10
            print(f"TARGET_U_OFFSET_PX = {single.TARGET_U_OFFSET_PX}")
        if key == ord("c"):
            send(0, 0, servo)
            mission.selector.forget()
            tracker.tracks = {}         # different camera = different picture
            cap.release()
            for step in range(1, 5):
                new_cap = single.open_camera((CAMERA_INDEX + step) % 4)
                if new_cap is not None:
                    CAMERA_INDEX = (CAMERA_INDEX + step) % 4
                    cap = new_cap
                    break
            print(f"Switched to camera {CAMERA_INDEX}")

    send(0, 0, servo)   # stop the gantry, laser off
    sock.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nColour ranges now (s saves them):")
    for n in NAMES:
        print(f"  {n:7s} low = {tuple(RANGES[n][0])}  high = {tuple(RANGES[n][1])}")


if __name__ == "__main__":
    main()
