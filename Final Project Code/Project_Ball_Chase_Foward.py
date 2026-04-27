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
from L1_lidar import Lidar
from L2_vector import getNearest

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
PINK_H_MIN, PINK_S_MIN, PINK_V_MIN =   0,  115, 165
PINK_H_MAX, PINK_S_MAX, PINK_V_MAX = 255, 250, 255

# BLUE (target #2 — chases this ball into the RIGHT scoop compartment)
# Obstacle detection used to be "big blue thing in camera" but is now done
# by LIDAR (see OBSTACLE_DIST_M). Blue is purely a ball color now.
BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN =   15, 185, 65
BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX = 255, 255, 255

# -----------------------------
# DETECTION — same as L3_image_filter
# -----------------------------
MIN_SIZE = 6

# Bounding-box width threshold for "close enough — stop":
# Tune by watching the CHASE log lines and noting the width at which the
# robot is at the desired distance from the ball.
PINK_STOP_WIDTH  = 90    # CHASE pink stops when ball width >= this (also used for blue)

# -----------------------------
# SPEED SETTINGS (all at or above MIN_SPEED; this robot won't move below ~3)
# -----------------------------
MIN_SPEED        = 3.0
CHASE_FWD_SPEED  = 3.0

# -----------------------------
# TIMING
# -----------------------------

# Two settle durations:
#   SETTLE_SEC     = big pause at MAJOR state changes (search/center/chase)
#   MINI_SETTLE_SEC= tiny pause between sub-bouts & minor internal transitions
SETTLE_SEC       = 0.2
MINI_SETTLE_SEC  = 0.2

# Pause after reaching/losing a ball before going back to FORWARD_DRIVE.
# Short because the just-caught color is added to caught_colors so it won't
# be re-detected this run, removing the main reason we needed a long pause.
LOST_WAIT_SEC = 0.5

# CHASE FINISH: after CHASE ends (either ball width is large enough or the
# ball is lost from view), drive STRAIGHT forward for this many seconds to
# push the ball into the cardboard scoop. Without this, the robot stops just
# short of physically catching the ball.
# Tune up if the ball isn't fully in the compartment; tune down if the robot
# overshoots and bumps the ball away.
CHASE_FINISH_SEC = 3.0

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
# APPROACH STEERING (closed-loop correction during CHASE and CHASE_BLUE)
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

# ============================================================
# FORWARD_DRIVE flow constants (initial straight-drive behavior)
# ============================================================
# The robot starts by driving forward indefinitely. While driving, it watches
# the camera for any pink/blue ball (chase it) AND polls the LIDAR for
# obstacles in a forward cone (stop and wait if too close).

# Forward-drive cruise speed.
FORWARD_DRIVE_SPEED = 3.0

# ============================================================
# LIDAR OBSTACLE DETECTION
# ============================================================
# The robot uses a SICK TiM561 LIDAR (via L1_lidar.py) for obstacle
# detection during FORWARD_DRIVE and OBSTACLE_WAIT. Every loop iteration
# polls the LIDAR for the nearest obstacle in a forward cone. If the
# nearest point in that cone is closer than OBSTACLE_DIST_M, the robot
# stops and waits.
#
# Camera-based OBSTACLE color detection is REMOVED — LIDAR replaces it.
#
# IMPORTANT: LIDAR is NOT polled during CHASE / CHASE_BLUE. The ball IS
# the target during those states — we want to drive INTO it, not stop
# short of it.

# Distance threshold in meters. Anything within OBSTACLE_DIST_M directly
# in front of the robot stops it.
OBSTACLE_DIST_M = 0.5

# Half-angle of the forward cone we care about, in degrees. The LIDAR
# scans 270° but for forward driving we only care about ±FRONT_CONE_DEG
# from straight ahead. 30° = 60° total cone.
FRONT_CONE_DEG = 30

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

def find_target(image, ignore_colors=None):
    """Look for pink and blue balls. Order = priority: PINK > BLUE.
    Returns (color, bbox) where color is 'PINK' or 'BLUE',
    or (None, None) if nothing found.

    ignore_colors: optional set/iterable of color names to skip entirely.
    Used to prevent re-detection of a ball that's already been caught
    this run.

    Note: obstacle detection is NOT done here anymore — that's the LIDAR's
    job (see lidar_obstacle_in_front)."""
    if ignore_colors is None:
        ignore_colors = set()
    if 'PINK' not in ignore_colors:
        pink_box = find_color(
            image,
            (PINK_H_MIN, PINK_S_MIN, PINK_V_MIN),
            (PINK_H_MAX, PINK_S_MAX, PINK_V_MAX),
        )
        if pink_box is not None:
            return ('PINK', pink_box)
    if 'BLUE' not in ignore_colors:
        blue_box = find_color(
            image,
            (BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN),
            (BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX),
        )
        if blue_box is not None:
            return ('BLUE', blue_box)
    return (None, None)

# -----------------------------
# LIDAR OBSTACLE HELPER
# -----------------------------
def lidar_obstacle_in_front(lidar):
    """Returns (is_blocked, distance_m, angle_deg) describing the nearest
    LIDAR point inside the forward cone of ±FRONT_CONE_DEG.
    is_blocked is True iff a valid point exists within OBSTACLE_DIST_M.
    distance_m / angle_deg are None if the scan isn't ready yet OR if no
    valid point is in the cone."""
    scan = lidar.get()
    if scan is None:
        return (False, None, None)
    try:
        # Filter to the forward cone (angles from -FRONT_CONE_DEG to +FRONT_CONE_DEG)
        # scan is shape (N, 2): column 0 = distance, column 1 = angle (deg)
        in_cone = scan[np.abs(scan[:, 1]) <= FRONT_CONE_DEG]
        if in_cone.shape[0] == 0:
            return (False, None, None)
        # getNearest expects an Nx2 array; filter to valid distances first
        # (distance > 0.016 per L2_vector.getValid).
        valids = in_cone[in_cone[:, 0] > 0.016]
        if valids.shape[0] == 0:
            return (False, None, None)
        idx = np.argmin(valids[:, 0])
        dist = float(valids[idx, 0])
        ang = float(valids[idx, 1])
        return (dist < OBSTACLE_DIST_M, dist, ang)
    except (ValueError, IndexError):
        return (False, None, None)

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

    # Connect to LIDAR. The LIDAR runs on a background thread (see L1_lidar)
    # and lidar.get() returns the most recent scan. If LIDAR fails to connect,
    # the helper lidar_obstacle_in_front() will just return "not blocked"
    # (since lidar.get() returns None) — robot still drives but won't stop
    # for obstacles. We print a warning if that happens.
    print("Connecting to LIDAR...")
    lidar = Lidar()
    lidar.connect()
    lidar_thread = lidar.run()
    sleep(1.0)  # let the thread spin up and grab a first scan
    if lidar.get() is None:
        print("WARNING: LIDAR not returning data. Obstacle detection disabled.")
    else:
        print("LIDAR ready.")

    print("Running. Ctrl+C to stop.")
    print(f"PINK HSV:  ({PINK_H_MIN},{PINK_S_MIN},{PINK_V_MIN}) "
          f"to ({PINK_H_MAX},{PINK_S_MAX},{PINK_V_MAX})")
    print(f"BLUE HSV:  ({BLUE_H_MIN},{BLUE_S_MIN},{BLUE_V_MIN}) "
          f"to ({BLUE_H_MAX},{BLUE_S_MAX},{BLUE_V_MAX})")
    print(f"LIDAR obstacle threshold: < {OBSTACLE_DIST_M}m within ±{FRONT_CONE_DEG}°")

    # State scratchpad
    # Start in FORWARD_DRIVE: the robot drives straight ahead indefinitely
    # while watching the camera for pink/blue balls and the LIDAR for
    # obstacles in a forward cone. Ball wins over obstacle.
    # After catching a ball OR after an obstacle clears, the robot returns
    # to FORWARD_DRIVE — there is NO look-left/look-right sweep anymore.
    state             = 'FORWARD_DRIVE'
    phase_end         = time()
    settle_until      = 0.0
    next_state        = None
    current_settle    = SETTLE_SEC
    smoothed_error    = 0.0
    approach_error    = 0.0    # smoothed error during CHASE / CHASE_BLUE
                               # (separate from CENTER's smoothed_error so
                               # the two states don't interfere)
    center_hold_start = None
    target_color      = None   # 'PINK' or 'BLUE' — what we're currently locked on
    caught_colors     = set()  # colors already caught this run; never re-detect.
                               # Populated when CHASE_FINISH completes.

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
        # leaving the centering flow entirely.
        if new_state == 'LOST_WAIT':
            smoothed_error = 0.0
        # approach_error is reset whenever we ENTER a chase state fresh
        # (i.e. after CENTER finishes and we head into CHASE / CHASE_BLUE).
        if new_state in ('CHASE', 'CHASE_BLUE'):
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

            color, ball = find_target(image, ignore_colors=caught_colors)

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
                    elif next_state == 'FORWARD_DRIVE':
                        # No special init — FORWARD_DRIVE just drives forward
                        # and watches for pink/blue every frame.
                        pass
                    elif next_state == 'OBSTACLE_WAIT':
                        # No timer — we stay in OBSTACLE_WAIT until LIDAR
                        # shows the path is clear (or a ball appears).
                        pass
                    print(f"SETTLE done -> {next_state}")
                    state = next_state
                    next_state = None
                else:
                    remaining = settle_until - time()
                    print(f"SETTLE | {remaining:.2f}s left (-> {next_state})")
                continue

            # ============= FORWARD_DRIVE (initial state) =============
            # Drive forward indefinitely. Watch for:
            #   - Pink/blue ball in camera -> CENTER -> CHASE (balls always win)
            #   - LIDAR obstacle in front  -> stop, OBSTACLE_WAIT
            # Ball detection is checked FIRST so the robot doesn't get stuck
            # in OBSTACLE_WAIT just because a ball happens to be close.
            if state == 'FORWARD_DRIVE':
                if ball is not None:
                    # Ball wins — start the chase regardless of LIDAR
                    x, y, w, h = ball
                    stop()
                    target_color = color
                    print(f"!! {color} BALL DETECTED while driving "
                          f"(w={w}px) -> SETTLE, then CENTER")
                    go_to_via_settle('CENTER')
                    continue
                # No ball — check LIDAR for obstacles
                blocked, dist, ang = lidar_obstacle_in_front(lidar)
                if blocked:
                    stop()
                    print(f"!! LIDAR OBSTACLE (dist={dist:.2f}m @ {ang:+.1f}°) "
                          f"-> stop, wait for it to clear")
                    go_to_via_settle('OBSTACLE_WAIT')
                    continue
                # Path clear, no ball — keep driving forward
                drive_forward(FORWARD_DRIVE_SPEED)
                if dist is not None:
                    print(f"FORWARD_DRIVE | searching... nearest={dist:.2f}m "
                          f"@ {ang:+.1f}° (clear)")
                else:
                    print("FORWARD_DRIVE | searching... (no LIDAR data)")
                continue

            # ============= OBSTACLE_WAIT =============
            # An obstacle was detected by LIDAR in front. Stay stopped until
            # LIDAR shows the path is clear OR a ball appears in the camera.
            if state == 'OBSTACLE_WAIT':
                stop()
                if ball is not None:
                    # A ball appeared while we were waiting — chase it
                    x, y, w, h = ball
                    target_color = color
                    print(f"OBSTACLE_WAIT | {color} ball appeared "
                          f"(w={w}px) -> CENTER")
                    go_to_via_settle('CENTER')
                    continue
                blocked, dist, ang = lidar_obstacle_in_front(lidar)
                if blocked:
                    print(f"OBSTACLE_WAIT | still blocked (dist={dist:.2f}m "
                          f"@ {ang:+.1f}°)")
                    continue
                # LIDAR is clear — go straight back to FORWARD_DRIVE
                if dist is not None:
                    print(f"OBSTACLE_WAIT | path clear (nearest={dist:.2f}m) "
                          f"-> FORWARD_DRIVE")
                else:
                    print("OBSTACLE_WAIT | path clear (no LIDAR data) "
                          "-> FORWARD_DRIVE")
                go_to_via_settle('FORWARD_DRIVE')
                continue

            # ============= CENTER =============
            if state == 'CENTER':
                if ball is None:
                    stop()
                    smoothed_error = 0.0
                    center_hold_start = None
                    print(f"CENTER ({target_color}) | lost target")
                    target_color = None
                    # Lost the target during centering — pause briefly in
                    # LOST_WAIT so the ball/situation can settle, then go
                    # back to driving forward to look for it again.
                    go_to_via_settle('LOST_WAIT')
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
                        else:  # BLUE
                            go_to_via_settle('CHASE_BLUE')
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
            # transition through CHASE_FINISH (push into scoop) → LOST_WAIT
            # → FORWARD_DRIVE.
            if state == 'CHASE':
                if ball is None or color != 'PINK':
                    stop()
                    print("CHASE | pink no longer seen — pushing ball into scoop")
                    caught_colors.add('PINK')
                    print(f"CHASE | caught PINK; will ignore PINK from now on. caught={sorted(caught_colors)}")
                    target_color = None
                    go_to_via_settle('CHASE_FINISH')
                else:
                    x, y, w, h = ball
                    if w >= PINK_STOP_WIDTH:
                        stop()
                        print(f"CHASE | reached pink (width={w}px >= {PINK_STOP_WIDTH}) — pushing into scoop")
                        caught_colors.add('PINK')
                        print(f"CHASE | caught PINK; will ignore PINK from now on. caught={sorted(caught_colors)}")
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
                    caught_colors.add('BLUE')
                    print(f"CHASE_BLUE | caught BLUE; will ignore BLUE from now on. caught={sorted(caught_colors)}")
                    target_color = None
                    go_to_via_settle('CHASE_FINISH')
                else:
                    x, y, w, h = ball
                    if w >= PINK_STOP_WIDTH:
                        stop()
                        print(f"CHASE_BLUE | reached blue (width={w}px >= {PINK_STOP_WIDTH}) — pushing into scoop")
                        caught_colors.add('BLUE')
                        print(f"CHASE_BLUE | caught BLUE; will ignore BLUE from now on. caught={sorted(caught_colors)}")
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

            # ============= CHASE_FINISH (pink and blue) =============
            # After CHASE / CHASE_BLUE ends, drive STRAIGHT forward (no steering)
            # for CHASE_FINISH_SEC to physically push the ball into the scoop.
            # Then go through LOST_WAIT → FORWARD_DRIVE to keep searching.
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

            # ============= LOST_WAIT =============
            # 2-second pause after losing/reaching a ball, before going back
            # to FORWARD_DRIVE.
            if state == 'LOST_WAIT':
                stop()
                if time() >= phase_end:
                    print("LOST_WAIT done -> back to FORWARD_DRIVE")
                    go_to_via_settle('FORWARD_DRIVE')
                else:
                    remaining = phase_end - time()
                    print(f"LOST_WAIT | {remaining:.2f}s left")
                continue

    except KeyboardInterrupt:
        pass

    finally:
        stop()
        camera.release()
        try:
            lidar.kill(lidar_thread)
        except Exception as e:
            print(f"LIDAR shutdown error (ignored): {e}")
        print("Exiting.")

if __name__ == '__main__':
    main()