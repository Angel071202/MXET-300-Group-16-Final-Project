# Project_Ball_Chase.py
#
# SEARCH: forward in short bouts -> settle -> rotate in short bouts -> settle -> repeat.
# ON PINK: stop -> settle -> CENTER (deadband + gain) -> settle -> CHASE until lost -> settle -> search.
#
# This robot stalls below wheel command ~3, so there is NO ramping.
# Smoothness comes from: short bouts + brief pauses between + longer pauses
# at major state transitions. Big single movements are replaced with multiple
# small ones so each start/stop feels less jarring.

import cv2
import numpy as np
import L2_speed_control as sc
import L2_inverse_kinematics as ik
import L2_kinematics as kin
import netifaces as ni
from time import sleep, time

# -----------------------------
# CAMERA SETUP
# -----------------------------
def getIp():
    for interface in ni.interfaces()[1:]:
        try:
            ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
            return ip
        except KeyError:
            continue
    return 0

stream_ip = getIp()
camera_input = 'http://' + stream_ip + ':8090/?action=stream' if stream_ip else 0

size_w = 240
size_h = 160

# -----------------------------
# HSV VALUES — EDIT THESE TO TUNE PINK
# -----------------------------
PINK_H_MIN, PINK_S_MIN, PINK_V_MIN =   0, 190, 155
PINK_H_MAX, PINK_S_MAX, PINK_V_MAX = 255, 255, 255

# -----------------------------
# DETECTION — same as L3_image_filter
# -----------------------------
MIN_SIZE = 6

# -----------------------------
# SPEED SETTINGS (all at or above MIN_SPEED; this robot won't move below ~3)
# -----------------------------
MIN_SPEED        = 3.0
SEARCH_FWD_SPEED = 4.0
CHASE_FWD_SPEED  = 4.0
ROTATE_SPEED     = 3.5

# -----------------------------
# TIMING
# -----------------------------
# Forward is broken into short bouts — same idea as rotation — so the start/stop
# cycle feels smaller and the camera has clean still moments for detection.
FWD_BOUTS        = 2     # number of forward bouts per "forward leg"
FWD_BOUT_SEC     = 0.7   # each bout duration
FWD_LOOK_SEC     = 0.3   # brief pause between forward bouts

# Rotation bouts. 0.2s is enough for the motors to actually rotate a bit.
ROTATE_BOUTS     = 4
ROTATE_BOUT_SEC  = 0.2
ROTATE_LOOK_SEC  = 0.6   # longer look pause — need clean frames for detection
ROTATE_DIR       = +1    # +1 left, -1 right

# Two settle durations:
#   SETTLE_SEC     = big pause at MAJOR state changes (search/center/chase)
#   MINI_SETTLE_SEC= tiny pause between sub-bouts & minor internal transitions
SETTLE_SEC       = 1.5
MINI_SETTLE_SEC  = 0.3

# -----------------------------
# CENTERING (ported from SideMotion, slightly softer tuning)
# -----------------------------
CENTER_DEADBAND  = 0.20   # narrower than before — ball needs to be closer to center
CENTER_TURN_GAIN = 10.0   # lower gain — small errors don't slam to max speed
CENTER_MAX_SPEED = 6.0    # cap lowered so fast turns don't overshoot as hard
CENTER_SMOOTHING = 0.75   # more smoothing — detection noise damped harder
CENTER_HOLD_SEC  = 1.0    # must stay inside deadband this long before centering is done

# -----------------------------
# BALL DETECTION (matches L3_image_filter.colorTracking)
# -----------------------------
def find_pink(image):
    image = cv2.resize(image, (size_w, size_h))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(
        hsv,
        (PINK_H_MIN, PINK_S_MIN, PINK_V_MIN),
        (PINK_H_MAX, PINK_S_MAX, PINK_V_MAX),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if 0.5 * w > MIN_SIZE:
            return (x, y, w, h)
    return None

# -----------------------------
# DRIVE HELPERS — no ramping, direct commands.
# -----------------------------
def stop():
    sc.driveOpenLoop(np.array([0.0, 0.0]))

def drive_forward(speed):
    sc.driveOpenLoop(np.array([speed, speed]))

def rotate_in_place(speed, direction):
    if direction > 0:
        sc.driveOpenLoop(np.array([-speed, +speed]))
    else:
        sc.driveOpenLoop(np.array([+speed, -speed]))

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_input)
    if not camera.isOpened():
        print("Failed to open camera.")
        return

    camera.set(3, size_w)
    camera.set(4, size_h)

    print("Running. Ctrl+C to stop.")
    print(f"PINK HSV: ({PINK_H_MIN},{PINK_S_MIN},{PINK_V_MIN}) "
          f"to ({PINK_H_MAX},{PINK_S_MAX},{PINK_V_MAX})")

    # State scratchpad
    state             = 'SEARCH_FWD'
    phase_end         = time() + FWD_BOUT_SEC
    settle_until      = 0.0
    next_state        = None
    current_settle    = SETTLE_SEC
    smoothed_error    = 0.0
    center_hold_start = None
    fwd_bouts_done    = 0
    rot_bouts_done    = 0

    def go_to_via_settle(new_state, mini=False):
        """Every transition goes through a SETTLE pause.
           mini=True uses the short pause for sub-bouts."""
        nonlocal state, settle_until, next_state, smoothed_error
        nonlocal center_hold_start, current_settle
        current_settle = MINI_SETTLE_SEC if mini else SETTLE_SEC
        state = 'SETTLE'
        settle_until = time() + current_settle
        next_state = new_state
        smoothed_error = 0.0
        center_hold_start = None
        tag = 'MINI-SETTLE' if mini else 'SETTLE'
        print(f"-> {tag} {current_settle:.2f}s, then {new_state}")

    try:
        while True:
            sleep(0.05)

            ret, image = camera.read()
            if not ret:
                print("Failed to retrieve image!")
                break

            ball = find_pink(image)

            # ----- Global pink-detection interrupt (all search/look phases) -----
            if ball is not None and state in ('SEARCH_FWD', 'FWD_LOOK',
                                              'SEARCH_ROT', 'ROTATE_LOOK'):
                stop()
                x, y, w, h = ball
                print(f"!! PINK DETECTED (w={w}px) -> stop, SETTLE, then CENTER")
                go_to_via_settle('CENTER')
                continue

            # ============= SETTLE =============
            if state == 'SETTLE':
                stop()
                if time() >= settle_until:
                    # Initialize the target state
                    if next_state == 'SEARCH_FWD':
                        phase_end = time() + FWD_BOUT_SEC
                        fwd_bouts_done = 0
                    elif next_state == 'SEARCH_ROT':
                        phase_end = time() + ROTATE_BOUT_SEC
                        rot_bouts_done = 0
                    elif next_state == 'CENTER':
                        smoothed_error = 0.0
                        center_hold_start = None
                    print(f"SETTLE done -> {next_state}")
                    state = next_state
                    next_state = None
                else:
                    remaining = settle_until - time()
                    print(f"SETTLE | {remaining:.2f}s left (-> {next_state})")
                continue

            # ============= SEARCH_FWD (bout) =============
            # Drive forward for one bout. When bout ends, either do a mini-pause
            # (FWD_LOOK) for the next bout, or finish the forward leg and go to rotation.
            if state == 'SEARCH_FWD':
                if time() >= phase_end:
                    stop()
                    fwd_bouts_done += 1
                    print(f"SEARCH_FWD bout {fwd_bouts_done}/{FWD_BOUTS} done")
                    if fwd_bouts_done >= FWD_BOUTS:
                        print("SEARCH_FWD: all bouts done")
                        go_to_via_settle('SEARCH_ROT')
                    else:
                        state = 'FWD_LOOK'
                        phase_end = time() + FWD_LOOK_SEC
                        print(f"-> FWD_LOOK {FWD_LOOK_SEC:.2f}s")
                else:
                    drive_forward(SEARCH_FWD_SPEED)
                    remaining = phase_end - time()
                    print(f"SEARCH_FWD bout {fwd_bouts_done+1}/{FWD_BOUTS} | "
                          f"{remaining:.2f}s left @ {SEARCH_FWD_SPEED}")
                continue

            # ============= FWD_LOOK =============
            # Brief pause between forward bouts. Camera gets a still moment.
            if state == 'FWD_LOOK':
                stop()
                if time() >= phase_end:
                    state = 'SEARCH_FWD'
                    phase_end = time() + FWD_BOUT_SEC
                    print(f"FWD_LOOK done -> next forward bout")
                else:
                    remaining = phase_end - time()
                    print(f"FWD_LOOK | {remaining:.2f}s left (looking for pink)")
                continue

            # ============= SEARCH_ROT (bout) =============
            if state == 'SEARCH_ROT':
                if time() >= phase_end:
                    stop()
                    rot_bouts_done += 1
                    print(f"SEARCH_ROT bout {rot_bouts_done}/{ROTATE_BOUTS} done")
                    if rot_bouts_done >= ROTATE_BOUTS:
                        print("SEARCH_ROT: all bouts done")
                        go_to_via_settle('SEARCH_FWD')
                    else:
                        state = 'ROTATE_LOOK'
                        phase_end = time() + ROTATE_LOOK_SEC
                        print(f"-> ROTATE_LOOK {ROTATE_LOOK_SEC:.2f}s")
                else:
                    rotate_in_place(ROTATE_SPEED, ROTATE_DIR)
                    remaining = phase_end - time()
                    print(f"SEARCH_ROT bout {rot_bouts_done+1}/{ROTATE_BOUTS} | "
                          f"{remaining:.2f}s left (dir={'L' if ROTATE_DIR>0 else 'R'})")
                continue

            # ============= ROTATE_LOOK =============
            if state == 'ROTATE_LOOK':
                stop()
                if time() >= phase_end:
                    state = 'SEARCH_ROT'
                    phase_end = time() + ROTATE_BOUT_SEC
                    print(f"ROTATE_LOOK done -> next rotation bout")
                else:
                    remaining = phase_end - time()
                    print(f"ROTATE_LOOK | {remaining:.2f}s left (looking for pink)")
                continue

            # ============= CENTER =============
            if state == 'CENTER':
                if ball is None:
                    stop()
                    smoothed_error = 0.0
                    center_hold_start = None
                    print("CENTER | lost ball")
                    go_to_via_settle('SEARCH_FWD')
                    continue

                x, y, w, h = ball
                center_x = x + w / 2.0
                raw_error = (center_x - size_w / 2.0) / (size_w / 2.0)
                smoothed_error = (CENTER_SMOOTHING * smoothed_error
                                  + (1.0 - CENTER_SMOOTHING) * raw_error)

                if abs(smoothed_error) < CENTER_DEADBAND:
                    stop()
                    if center_hold_start is None:
                        center_hold_start = time()
                    held = time() - center_hold_start
                    print(f"CENTER | centered err={smoothed_error:+.2f}  "
                          f"held {held:.2f}/{CENTER_HOLD_SEC}s")
                    if held >= CENTER_HOLD_SEC:
                        print("CENTER done")
                        go_to_via_settle('CHASE')
                    continue

                center_hold_start = None
                sign = 1.0 if smoothed_error > 0 else -1.0
                effective_error = smoothed_error - sign * CENTER_DEADBAND
                turn = CENTER_TURN_GAIN * effective_error

                if abs(turn) < MIN_SPEED:
                    turn = MIN_SPEED * sign
                turn = max(-CENTER_MAX_SPEED, min(CENTER_MAX_SPEED, turn))

                left_speed  = +turn
                right_speed = -turn
                sc.driveOpenLoop(np.array([left_speed, right_speed]))
                print(f"CENTER | err={smoothed_error:+.2f}  "
                      f"L={left_speed:+.2f} R={right_speed:+.2f}")
                continue

            # ============= CHASE =============
            if state == 'CHASE':
                if ball is None:
                    stop()
                    print("CHASE | ball lost")
                    go_to_via_settle('SEARCH_FWD')
                else:
                    x, y, w, h = ball
                    drive_forward(CHASE_FWD_SPEED)
                    print(f"CHASE | width={w}px @ {CHASE_FWD_SPEED}")
                continue

    except KeyboardInterrupt:
        pass

    finally:
        stop()
        camera.release()
        print("Exiting.")

if __name__ == '__main__':
    main()