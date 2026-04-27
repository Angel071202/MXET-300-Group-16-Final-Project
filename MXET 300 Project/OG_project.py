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
# HSV VALUES — EDIT THESE TO TUNE COLORS
# -----------------------------
# PINK (target #1 — robot chases this and lands it in the LEFT scoop compartment)
PINK_H_MIN, PINK_S_MIN, PINK_V_MIN =   0, 50, 185
PINK_H_MAX, PINK_S_MAX, PINK_V_MAX = 255, 240, 255

# GREEN (the "decoy" — robot approaches but doesn't pick up; rotates away)
# Tuned with HSV slider tool — clean mask on actual green box.
GREEN_H_MIN, GREEN_S_MIN, GREEN_V_MIN =  10,  0,  180
GREEN_H_MAX, GREEN_S_MAX, GREEN_V_MAX = 40, 255, 255

# BLUE (target #2 — robot chases this and lands it in the RIGHT scoop compartment)
# Tuned with HSV slider tool — clean mask on actual blue object.
# NOTE: BLUE and GREEN ranges overlap on H and S, only V differs:
# blue requires V >= 200, green caps at V <= 205. The 200-205 sliver is
# ambiguous. Watch detection logs if either color gets misclassified.
BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN =  0, 195, 110
BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX = 250, 255, 255

# -----------------------------
# DETECTION — same as L3_image_filter
# -----------------------------
MIN_SIZE = 6

# Bounding-box width thresholds for "close enough — stop":
# Tune these by watching the CHASE/APPROACH log lines and noting the width
# at which the robot is at the desired distance from the object.
PINK_STOP_WIDTH  = 90    # CHASE pink stops when ball width >= this
GREEN_STOP_WIDTH = 90    # APPROACH green stops when box width >= this

# -----------------------------
# SPEED SETTINGS (all at or above MIN_SPEED; this robot won't move below ~3)
# -----------------------------
MIN_SPEED        = 3.0
CHASE_FWD_SPEED  = 3.0
ROTATE_SPEED     = 3 # raised from 2.5 — at 2.5 the wheels would barely
                        # turn the robot, especially after stopping in front
                        # of a box (static friction). 3.5 matches the
                        # forward-drive regime that we know moves reliably.

# LEFT_SWEEP rotates at the same speed as RIGHT_SWEEP (= ROTATE_SPEED) so
# the post-kick rotation feels constant in both directions. We tried slowing
# LEFT_SWEEP to 2.0 to give detection more dwell time per frame, but the
# kick at 4.0 dropping to 2.0 mid-sweep made the rotation feel like a
# "burst then crawl". Now both sweeps kick from 4.0 down to ROTATE_SPEED.
LEFT_SWEEP_ROTATE_SPEED = 3

# Startup kick: at the start of each LEFT_SWEEP and RIGHT_SWEEP, run
# wheels at this higher speed for KICK_SEC to break static friction.
# Without this, the first second of rotation barely moves because the
# motors are starting from a complete stop and need to overcome friction.
ROTATE_KICK_SPEED = 4
ROTATE_KICK_SEC   = 0.4

# -----------------------------
# TIMING
# -----------------------------

# All search/rotate behavior is the LEFT_SWEEP/RIGHT_SWEEP pattern defined
# below in POST-ARRIVAL ROTATION PATTERN. The robot does this sweep at
# startup and after every box arrival. If nothing is detected during either
# sweep, the robot enters STOPPED and stays there.

# -----------------------------
# ROTATION DIRECTION — flip this single value to change ALL rotation behavior
# -----------------------------
# This affects POST_ARRIVAL_ROT (the look-left-look-right sweep at startup
# and after visiting a box). +1 means LEFT_SWEEP rotates left first.
#
#   +1 = robot rotates LEFT  (counter-clockwise from above, camera pans left)
#   -1 = robot rotates RIGHT (clockwise from above, camera pans right)
#
# Pick whichever direction tends to face more of your boxes first based on
# where the robot starts. If unsure, leave at +1.
ROTATE_DIR       = +1

# Two settle durations:
#   SETTLE_SEC     = big pause at MAJOR state changes (search/center/chase)
#   MINI_SETTLE_SEC= tiny pause between sub-bouts & minor internal transitions
SETTLE_SEC       = 0.5
MINI_SETTLE_SEC  = 0.3

# Pause after either reaching/losing pink OR arriving at green, before rotating to search again
LOST_WAIT_SEC = 2.0

# CHASE FINISH: after CHASE ends (either ball width is large enough or pink is
# lost from view), drive STRAIGHT forward for this many seconds to push the
# ball into the cardboard scoop. Without this, the robot stops just short of
# physically catching the ball.
# Tune up if the ball isn't fully in the compartment; tune down if the robot
# overshoots and bumps the ball away.
CHASE_FINISH_SEC = 1.5

# POST-ARRIVAL ROTATION PATTERN (after visiting a box or losing pink)
# Two sweeps: LEFT then RIGHT. If nothing found in either, robot stops.
# The just-visited box is hidden from detection by two lockouts:
#   - At the very start of LEFT_SWEEP (rotate off the box)
#   - At the start of RIGHT_SWEEP (passing back over the box on the way to
#     the right side; lockout duration = LEFT_SWEEP_SEC + extra padding,
#     because returning to start heading still leaves the box in the camera's
#     wide FOV; we need extra time to fully sweep it out of view)
#
# Tune in seconds based on actual robot behavior. Don't trust degree
# estimates — they vary with floor friction, battery, and start-up inertia.
LEFT_SWEEP_SEC      = 2.0   # how long to sweep left looking for new boxes
RIGHT_SWEEP_SEC     = 3.0   # how long to sweep right total (first ~LEFT_SWEEP_SEC
                            # of this is the "return" past the box, with
                            # detection blocked; the rest is the actual right scan)
START_LOCKOUT_SEC   = 1.0   # detection blocked at start of LEFT_SWEEP so the
                            # robot can rotate the just-visited box out of frame
                            # before allowing any new detection
RIGHT_SWEEP_LOCKOUT_PADDING_SEC = 1.5  # extra lockout time on top of LEFT_SWEEP_SEC
                            # at start of RIGHT_SWEEP. Without this, the robot
                            # returns to original heading but the box is still
                            # in view (camera FOV is wide). Increase if the
                            # just-visited box is being re-detected during
                            # RIGHT_SWEEP. Total RIGHT_SWEEP lockout =
                            # LEFT_SWEEP_SEC + this padding.

# -----------------------------
# CENTERING
# -----------------------------
# This robot can rotate at lower commands than it can drive forward (both
# wheels working together overcome stall friction more easily). So CENTER
# uses its own min-turn floor that's *below* MIN_SPEED. This makes
# proportional control actually proportional instead of bang-bang.
CENTER_DEADBAND   = 0.30   # |error| below this = "close enough", stop turning.
                           # Loosened from 0.15 — CENTER doesn't need to be
                           # precise. APPROACH does closed-loop steering which
                           # corrects any residual offset while driving.
CENTER_TURN_GAIN  = 2.0    # how hard to turn per unit error.
                           # Lowered from 4.0 since smoothing is also lower now;
                           # the robot reacts faster, so it doesn't need as much
                           # raw gain to correct.
CENTER_MIN_TURN   = 1.9    # smallest turn command that overcomes friction
                           # when both wheels work together. Tried 1.3 to
                           # reduce overshoot but the wheels stalled at that
                           # speed — back to 1.9 which physically rotates.
                           # Overshoot is now mitigated by lower MAX_SPEED
                           # and higher SMOOTHING.
CENTER_MAX_SPEED  = 1.9    # clamp on wheel command magnitude.
                           # Must be >= CENTER_MIN_TURN or the floor gets
                           # immediately clamped down and turn is constant.
CENTER_SMOOTHING  = 0.5    # 0 = react every frame, 0.9 = very smooth/laggy.
                           # Raised from 0.3 — more smoothing damps fast
                           # error swings that otherwise feed oscillation.
CENTER_HOLD_SEC   = 0.3    # must stay inside deadband this long before centering is done.
                           # Shortened from 1.0 — once the target is roughly
                           # in front, hand off to APPROACH instead of dwelling.

# -----------------------------
# APPROACH STEERING (closed-loop correction during CHASE and APPROACH_GREEN)
# -----------------------------
# When driving forward toward a target, even small angular errors compound
# over distance — the robot drifts past the target. These constants govern
# small course corrections that keep the robot aimed at the target while
# also driving forward.
APPROACH_DEADBAND  = 0.15   # |error| below this = no steering correction
APPROACH_TURN_GAIN = 2.0    # how strongly to steer per unit error.
                            # With max |err|≈1.0, this gives turn ≈ 2.0
                            # at frame edge, capped by APPROACH_MAX_TURN.
APPROACH_MAX_TURN  = 1.0    # max steering correction added/subtracted
                            # to wheel commands. Keep modest so the robot
                            # primarily moves forward (3.0) with small
                            # steering deltas (±1.0 → wheels in [2.0, 4.0]).
APPROACH_SMOOTHING = 0.3    # smoothing factor for the error signal,
                            # same role as CENTER_SMOOTHING

# CHASE TARGET OFFSET — used only in CHASE (pink) so the ball lands in
# one of the front cardboard scoop's side compartments instead of being
# blocked by the center divider.
# Positive = ball ends up on the robot's LEFT side compartment
# Negative = ball ends up on the robot's RIGHT side compartment
# 0.0 = ball goes dead-center (will hit the divider)
# 0.3 = ball appears 30% to the left of camera-frame center as robot drives
# Tune higher if the ball misses the compartment to the right; tune lower
# if the ball misses to the left.
CHASE_TARGET_OFFSET = 0.3

# CHASE_BLUE_TARGET_OFFSET — same idea as CHASE_TARGET_OFFSET but for the
# blue ball, which lands in the RIGHT scoop compartment. Negative because
# the blue ball needs to appear on the camera-RIGHT during chase, which
# means the robot is angled with the ball off to its right side.
# Mirror of pink's offset: -0.3 vs +0.3.
CHASE_BLUE_TARGET_OFFSET = -0.5

# -----------------------------
# COLOR DETECTION (matches L3_image_filter.colorTracking)
# -----------------------------
def find_color(image, hsv_lo, hsv_hi):
    """Return (x, y, w, h) of the largest blob in the given HSV range,
    or None if no blob meets the size threshold."""
    image = cv2.resize(image, (size_w, size_h))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_lo, hsv_hi)
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

def find_target(image):
    """Look for pink, blue, and green. Order = priority: first-detected wins.
    Pink and blue are both target colors (chase + scoop), green is decoy.
    Returns (color, bbox) where color is 'PINK', 'BLUE', or 'GREEN',
    or (None, None) if nothing found."""
    pink_box = find_color(
        image,
        (PINK_H_MIN, PINK_S_MIN, PINK_V_MIN),
        (PINK_H_MAX, PINK_S_MAX, PINK_V_MAX),
    )
    if pink_box is not None:
        return ('PINK', pink_box)
    blue_box = find_color(
        image,
        (BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN),
        (BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX),
    )
    if blue_box is not None:
        return ('BLUE', blue_box)
    green_box = find_color(
        image,
        (GREEN_H_MIN, GREEN_S_MIN, GREEN_V_MIN),
        (GREEN_H_MAX, GREEN_S_MAX, GREEN_V_MAX),
    )
    if green_box is not None:
        return ('GREEN', green_box)
    return (None, None)

# -----------------------------
# DRIVE HELPERS — no ramping, direct commands.
# -----------------------------
def stop():
    sc.driveOpenLoop(np.array([0.0, 0.0]))

def drive_forward(speed):
    sc.driveOpenLoop(np.array([speed, speed]))

def drive_forward_steered(speed, turn):
    """Drive forward at `speed` while applying a steering correction `turn`.
    Positive turn = steer right (camera pans right) — for ball on the right.
    Negative turn = steer left (camera pans left) — for ball on the left.
    Sign convention matches CENTER on this robot: L=+turn, R=-turn for the
    rotational component, added to forward [speed, speed].
    Result: L = speed + turn, R = speed - turn."""
    sc.driveOpenLoop(np.array([speed + turn, speed - turn]))

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
    print(f"PINK HSV:  ({PINK_H_MIN},{PINK_S_MIN},{PINK_V_MIN}) "
          f"to ({PINK_H_MAX},{PINK_S_MAX},{PINK_V_MAX})")
    print(f"BLUE HSV:  ({BLUE_H_MIN},{BLUE_S_MIN},{BLUE_V_MIN}) "
          f"to ({BLUE_H_MAX},{BLUE_S_MAX},{BLUE_V_MAX})")
    print(f"GREEN HSV: ({GREEN_H_MIN},{GREEN_S_MIN},{GREEN_V_MIN}) "
          f"to ({GREEN_H_MAX},{GREEN_S_MAX},{GREEN_V_MAX})")

    # State scratchpad
    # Start with POST_ARRIVAL_ROT so the robot does its look-left-look-right
    # sweep at startup. If nothing is found, it transitions to STOPPED.
    # All search behavior in this program is the POST_ARRIVAL_ROT sweep —
    # there is no continuous 360° rotation.
    state             = 'POST_ARRIVAL_ROT'
    rot_leg           = 1
    phase_start       = time()  # when current rotation phase started, for kick window
    phase_end         = time() + LEFT_SWEEP_SEC
    detect_lockout_until = time() + START_LOCKOUT_SEC
    settle_until      = 0.0
    next_state        = None
    current_settle    = SETTLE_SEC
    smoothed_error    = 0.0
    approach_error    = 0.0    # smoothed error during CHASE/APPROACH_GREEN
                               # (separate from CENTER's smoothed_error so
                               # the two states don't interfere)
    center_hold_start = None
    target_color      = None   # 'PINK', 'BLUE', or 'GREEN' — what we're currently locked on
    # (rot_leg and detect_lockout_until were initialized above with the
    # startup POST_ARRIVAL_ROT state; see comments at top of state scratchpad.)

    def go_to_via_settle(new_state, mini=False):
        """Every transition goes through a SETTLE pause.
           mini=True uses the short pause for sub-bouts."""
        nonlocal state, settle_until, next_state, smoothed_error, approach_error
        nonlocal center_hold_start, current_settle
        current_settle = MINI_SETTLE_SEC if mini else SETTLE_SEC
        state = 'SETTLE'
        settle_until = time() + current_settle
        next_state = new_state
        # Don't wipe smoothed_error when heading into CENTER — preserving it
        # avoids a jerk on the first centering frame. Only reset when we're
        # leaving the centering flow entirely (back to search).
        if new_state in ('LOST_WAIT', 'POST_ARRIVAL_ROT'):
            smoothed_error = 0.0
        # approach_error is reset whenever we ENTER an approach state fresh
        # (i.e. after CENTER finishes and we head into CHASE / APPROACH_GREEN).
        if new_state in ('CHASE', 'APPROACH_GREEN'):
            approach_error = 0.0
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

            color, ball = find_target(image)

            # ----- Global color-detection interrupt -----
            # Watches POST_ARRIVAL_ROT for any colored target.
            # Pink is preferred over green (handled inside find_target).
            # IMPORTANT: LOST_WAIT is NOT in this list. The wait after
            # reaching/losing a target must run to completion so the robot
            # transitions into POST_ARRIVAL_ROT and rotates away from the
            # just-visited target. Otherwise the robot would re-lock onto
            # the same green box it just stopped in front of.
            # During POST_ARRIVAL_ROT, detection is also blocked for the
            # configured lockout windows: at the start of LEFT_SWEEP (clearing
            # the box from view) and during the start of RIGHT_SWEEP (when
            # passing back over the box's original heading).
            if ball is not None and state == 'POST_ARRIVAL_ROT':
                if time() < detect_lockout_until:
                    pass  # lockout active — fall through to keep rotating
                else:
                    stop()
                    x, y, w, h = ball
                    target_color = color
                    print(f"!! {color} DETECTED (w={w}px) -> stop, SETTLE, then CENTER")
                    go_to_via_settle('CENTER')
                    continue

            # ============= SETTLE =============
            if state == 'SETTLE':
                stop()
                if time() >= settle_until:
                    # Initialize the target state
                    if next_state == 'CENTER':
                        # keep smoothed_error as-is to avoid a jerk on entry
                        center_hold_start = None
                    elif next_state == 'LOST_WAIT':
                        phase_end = time() + LOST_WAIT_SEC
                    elif next_state == 'CHASE_FINISH':
                        # Drive forward for CHASE_FINISH_SEC to push the ball
                        # into the scoop after CHASE detection ends.
                        phase_end = time() + CHASE_FINISH_SEC
                    elif next_state == 'POST_ARRIVAL_ROT':
                        # Start fresh on LEFT_SWEEP.
                        rot_leg = 1
                        phase_start = time()
                        phase_end = time() + LEFT_SWEEP_SEC
                        # Suppress detection for the first chunk of LEFT_SWEEP
                        # so we rotate physically off the just-visited target.
                        detect_lockout_until = time() + START_LOCKOUT_SEC
                    print(f"SETTLE done -> {next_state}")
                    state = next_state
                    next_state = None
                else:
                    remaining = settle_until - time()
                    print(f"SETTLE | {remaining:.2f}s left (-> {next_state})")
                continue

            # ============= CENTER =============
            if state == 'CENTER':
                if ball is None:
                    stop()
                    smoothed_error = 0.0
                    center_hold_start = None
                    print(f"CENTER ({target_color}) | lost target")
                    target_color = None
                    # Lost the target during centering — fall back to a fresh
                    # look-left-look-right sweep. If nothing is found, robot stops.
                    go_to_via_settle('POST_ARRIVAL_ROT')
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
                    print(f"CENTER ({target_color}) | centered err={smoothed_error:+.2f}  "
                          f"held {held:.2f}/{CENTER_HOLD_SEC}s")
                    if held >= CENTER_HOLD_SEC:
                        print(f"CENTER ({target_color}) done")
                        if target_color == 'PINK':
                            go_to_via_settle('CHASE')
                        elif target_color == 'BLUE':
                            go_to_via_settle('CHASE_BLUE')
                        else:  # GREEN
                            go_to_via_settle('APPROACH_GREEN')
                    continue

                center_hold_start = None
                sign = 1.0 if smoothed_error > 0 else -1.0
                effective_error = smoothed_error - sign * CENTER_DEADBAND
                turn = CENTER_TURN_GAIN * effective_error

                # Use CENTER_MIN_TURN (not MIN_SPEED) as the floor — rotation
                # needs less command than forward driving since both wheels help.
                if abs(turn) < CENTER_MIN_TURN:
                    turn = CENTER_MIN_TURN * sign
                turn = max(-CENTER_MAX_SPEED, min(CENTER_MAX_SPEED, turn))

                # Direction: on this robot, L=+, R=- rotates counter-clockwise
                # (camera pans left). When ball is left of center (err<0), we
                # want camera to pan left → L=+ when turn<0... wait, that's
                # wrong. Let's be concrete:
                #   err<0  → ball on left  → want camera to pan left  → CCW rotation → L=-, R=+
                #   err>0  → ball on right → want camera to pan right → CW rotation  → L=+, R=-
                # turn has same sign as err (turn = gain * err, both negative
                # when err is negative). So we want L = +turn, R = -turn:
                #   err=-0.5 → turn=-X → L=-X, R=+X ✓ (CCW, camera pans left)
                #   err=+0.5 → turn=+X → L=+X, R=-X ✓ (CW, camera pans right)
                # Confirmed empirically from log data on this specific robot.
                left_speed  = +turn
                right_speed = -turn
                sc.driveOpenLoop(np.array([left_speed, right_speed]))
                print(f"CENTER ({target_color}) | err={smoothed_error:+.2f}  "
                      f"L={left_speed:+.2f} R={right_speed:+.2f}")
                continue

            # ============= CHASE (pink only) =============
            # Drive forward (with steering correction) until pink is no longer
            # seen OR until ball width hits PINK_STOP_WIDTH. Either way,
            # transition through LOST_WAIT (2s pause) → POST_ARRIVAL_ROT.
            if state == 'CHASE':
                if ball is None or color != 'PINK':
                    stop()
                    print("CHASE | pink no longer seen — pushing ball into scoop")
                    target_color = None
                    go_to_via_settle('CHASE_FINISH')
                else:
                    x, y, w, h = ball
                    if w >= PINK_STOP_WIDTH:
                        stop()
                        print(f"CHASE | reached pink (width={w}px >= {PINK_STOP_WIDTH}) — pushing into scoop")
                        target_color = None
                        go_to_via_settle('CHASE_FINISH')
                    else:
                        # Compute steering correction from horizontal error.
                        # The CHASE_TARGET_OFFSET shifts the "zero error" point
                        # from frame center to a side, so the ball lands in
                        # the chosen side compartment of the front scoop.
                        # Positive offset → ball ends up on robot's LEFT side.
                        center_x = x + w / 2.0
                        raw_error = ((center_x - size_w / 2.0) / (size_w / 2.0)
                                     + CHASE_TARGET_OFFSET)
                        approach_error = (APPROACH_SMOOTHING * approach_error
                                          + (1.0 - APPROACH_SMOOTHING) * raw_error)
                        if abs(approach_error) < APPROACH_DEADBAND:
                            turn = 0.0
                        else:
                            turn = APPROACH_TURN_GAIN * approach_error
                            turn = max(-APPROACH_MAX_TURN, min(APPROACH_MAX_TURN, turn))
                        drive_forward_steered(CHASE_FWD_SPEED, turn)
                        print(f"CHASE | width={w}px err={approach_error:+.2f} "
                              f"turn={turn:+.2f} @ {CHASE_FWD_SPEED} "
                              f"(target offset {CHASE_TARGET_OFFSET:+.2f})")
                continue

            # ============= CHASE_BLUE (blue only) =============
            # Mirror of CHASE but with the opposite (negative) offset so the
            # blue ball lands in the RIGHT scoop compartment instead of the
            # left. Otherwise identical: drive forward with steering until
            # blue is no longer seen OR width hits PINK_STOP_WIDTH, then
            # CHASE_FINISH pushes it the rest of the way in.
            if state == 'CHASE_BLUE':
                if ball is None or color != 'BLUE':
                    stop()
                    print("CHASE_BLUE | blue no longer seen — pushing ball into scoop")
                    target_color = None
                    go_to_via_settle('CHASE_FINISH')
                else:
                    x, y, w, h = ball
                    if w >= PINK_STOP_WIDTH:
                        stop()
                        print(f"CHASE_BLUE | reached blue (width={w}px >= {PINK_STOP_WIDTH}) — pushing into scoop")
                        target_color = None
                        go_to_via_settle('CHASE_FINISH')
                    else:
                        # Same geometry as CHASE, but with the offset flipped
                        # to negative so the ball drifts to the camera-RIGHT
                        # side of frame and ends up in the right scoop slot.
                        center_x = x + w / 2.0
                        raw_error = ((center_x - size_w / 2.0) / (size_w / 2.0)
                                     + CHASE_BLUE_TARGET_OFFSET)
                        approach_error = (APPROACH_SMOOTHING * approach_error
                                          + (1.0 - APPROACH_SMOOTHING) * raw_error)
                        if abs(approach_error) < APPROACH_DEADBAND:
                            turn = 0.0
                        else:
                            turn = APPROACH_TURN_GAIN * approach_error
                            turn = max(-APPROACH_MAX_TURN, min(APPROACH_MAX_TURN, turn))
                        drive_forward_steered(CHASE_FWD_SPEED, turn)
                        print(f"CHASE_BLUE | width={w}px err={approach_error:+.2f} "
                              f"turn={turn:+.2f} @ {CHASE_FWD_SPEED} "
                              f"(target offset {CHASE_BLUE_TARGET_OFFSET:+.2f})")
                continue

            # ============= CHASE_FINISH (pink only) =============
            # After CHASE ends, drive STRAIGHT forward (no steering) for
            # CHASE_FINISH_SEC to physically push the ball into the scoop.
            # Then go through LOST_WAIT → POST_ARRIVAL_ROT to look for next color.
            if state == 'CHASE_FINISH':
                if time() >= phase_end:
                    stop()
                    print("CHASE_FINISH done -> ball should be in scoop")
                    go_to_via_settle('LOST_WAIT')
                else:
                    drive_forward_steered(CHASE_FWD_SPEED, 0.0)
                    remaining = phase_end - time()
                    print(f"CHASE_FINISH | {remaining:.2f}s left "
                          f"(pushing ball at {CHASE_FWD_SPEED})")
                continue

            # ============= APPROACH_GREEN =============
            # Drive forward (with steering correction) until close
            # (GREEN_STOP_WIDTH). Don't crash. Then go through LOST_WAIT →
            # POST_ARRIVAL_ROT to look for the next color.
            if state == 'APPROACH_GREEN':
                if ball is None or color != 'GREEN':
                    stop()
                    print("APPROACH_GREEN | green lost — waiting then rotating")
                    target_color = None
                    go_to_via_settle('LOST_WAIT')
                else:
                    x, y, w, h = ball
                    if w >= GREEN_STOP_WIDTH:
                        stop()
                        print(f"APPROACH_GREEN | reached green (width={w}px >= {GREEN_STOP_WIDTH}) — stopping in front")
                        target_color = None
                        go_to_via_settle('LOST_WAIT')
                    else:
                        # Same closed-loop steering as CHASE
                        center_x = x + w / 2.0
                        raw_error = (center_x - size_w / 2.0) / (size_w / 2.0)
                        approach_error = (APPROACH_SMOOTHING * approach_error
                                          + (1.0 - APPROACH_SMOOTHING) * raw_error)
                        if abs(approach_error) < APPROACH_DEADBAND:
                            turn = 0.0
                        else:
                            turn = APPROACH_TURN_GAIN * approach_error
                            turn = max(-APPROACH_MAX_TURN, min(APPROACH_MAX_TURN, turn))
                        drive_forward_steered(CHASE_FWD_SPEED, turn)
                        print(f"APPROACH_GREEN | width={w}px err={approach_error:+.2f} "
                              f"turn={turn:+.2f} @ {CHASE_FWD_SPEED}")
                continue

            # ============= LOST_WAIT =============
            # 2-second pause after losing/reaching any target. Detection
            # interrupt is active during this state, so if a color reappears
            # (subject to lockout for green), we'll lock back on immediately.
            if state == 'LOST_WAIT':
                stop()
                if time() >= phase_end:
                    print("LOST_WAIT done -> rotating to search")
                    go_to_via_settle('POST_ARRIVAL_ROT')
                else:
                    remaining = phase_end - time()
                    print(f"LOST_WAIT | {remaining:.2f}s left")
                continue

            # ============= POST_ARRIVAL_ROT =============
            # Two-phase sweep after LOST_WAIT (follows green-arrived OR pink-ended):
            #   Phase 1 (LEFT_SWEEP):  rotate ROTATE_DIR for LEFT_SWEEP_SEC
            #                          (detection blocked for first START_LOCKOUT_SEC
            #                           to clear just-visited box from view)
            #   Phase 2 (RIGHT_SWEEP): rotate -ROTATE_DIR for RIGHT_SWEEP_SEC
            #                          (detection blocked for first LEFT_SWEEP_SEC
            #                           — the "return" portion passing back over
            #                           the box's original heading)
            # If both phases complete with no detection, robot enters STOPPED.
            if state == 'POST_ARRIVAL_ROT':
                if time() >= phase_end:
                    stop()
                    if rot_leg == 1:
                        # LEFT_SWEEP done — start RIGHT_SWEEP. Block detection
                        # for LEFT_SWEEP_SEC + extra padding: the LEFT_SWEEP_SEC
                        # covers the "return" portion that passes back over
                        # the box's original heading, and the padding gives
                        # extra time to swing the box fully out of the camera's
                        # wide FOV before detection re-enables.
                        rot_leg = 2
                        phase_start = time()
                        phase_end = time() + RIGHT_SWEEP_SEC
                        detect_lockout_until = (time() + LEFT_SWEEP_SEC
                                                + RIGHT_SWEEP_LOCKOUT_PADDING_SEC)
                        print("POST_ARRIVAL_ROT LEFT_SWEEP done — RIGHT_SWEEP "
                              "(detection blocked while passing back over box)")
                    else:
                        # Both phases done — give up and stay put.
                        print("POST_ARRIVAL_ROT both sweeps done — no color found, stopping")
                        go_to_via_settle('STOPPED')
                else:
                    # LEFT_SWEEP uses ROTATE_DIR; RIGHT_SWEEP uses the opposite.
                    leg_dir = ROTATE_DIR if rot_leg == 1 else -ROTATE_DIR
                    # Apply kick speed for the first ROTATE_KICK_SEC of each
                    # phase to break static friction; this fixes the slow-start
                    # issue where the wheels were barely moving the robot
                    # despite seconds of rotation commands.
                    # After the kick: LEFT_SWEEP rotates slower than RIGHT_SWEEP
                    # so detection has more dwell time on each box per frame.
                    in_kick_window = (time() - phase_start) < ROTATE_KICK_SEC
                    if in_kick_window:
                        speed = ROTATE_KICK_SPEED
                    elif rot_leg == 1:
                        speed = LEFT_SWEEP_ROTATE_SPEED
                    else:
                        speed = ROTATE_SPEED
                    rotate_in_place(speed, leg_dir)
                    remaining = phase_end - time()
                    lockout_left = max(0.0, detect_lockout_until - time())
                    phase_name = 'LEFT_SWEEP' if rot_leg == 1 else 'RIGHT_SWEEP'
                    kick_tag = ' KICK' if in_kick_window else ''
                    print(f"POST_ARRIVAL_ROT {phase_name}{kick_tag} | "
                          f"{remaining:.2f}s left "
                          f"(dir={'L' if leg_dir>0 else 'R'} @ {speed:.1f}, "
                          f"lockout {lockout_left:.2f}s)")
                continue

            # ============= STOPPED =============
            # Terminal state — no color found after a full two-leg sweep.
            # Stay put until Ctrl+C. The detection interrupt list does NOT
            # include STOPPED, so the robot will remain here even if a color
            # later wanders into view.
            if state == 'STOPPED':
                stop()
                # Print sparingly so the terminal isn't spammed.
                # (Could add a one-time-only print, but a quiet loop is fine.)
                continue

    except KeyboardInterrupt:
        pass

    finally:
        stop()
        camera.release()
        print("Exiting.")

if __name__ == '__main__':
    main()