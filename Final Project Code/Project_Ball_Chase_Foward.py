# Project_Ball_Chase.py
# main loop for the ball-sorting scuttle. drives forward, looks for pink/blue
# ping pong balls with the camera, chases whichever it sees first, scoops it
# into the matching side compartment. lidar handles obstacle stops while we're
# searching (not while chasing).
#
# the scuttle won't actually move at wheel commands below ~3, so all forward
# speeds are pinned at 3 and we don't bother trying to ramp.

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
# camera setup
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
# HSV ranges
# tuned with the node-red sliders on the actual balls under lab lighting.
# re-tune if you move to a room with different lights — the pink especially
# starts grabbing wall reflections if the lighting changes.
# -----------------------------
# pink ball -> goes in LEFT compartment
PINK_H_MIN, PINK_S_MIN, PINK_V_MIN =   0,  115, 165
PINK_H_MAX, PINK_S_MAX, PINK_V_MAX = 255, 250, 255

# blue ball -> goes in RIGHT compartment
BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN =   15, 185, 65
BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX = 255, 255, 255

# -----------------------------
# detection thresholds
# -----------------------------
MIN_SIZE = 6  # smaller blob than this and we treat it as noise

# how big the ball has to look (in pixels wide) before we say "close enough,
# go push it in." used for both pink and blue.
PINK_STOP_WIDTH  = 90

# -----------------------------
# speeds. scuttle stalls below ~3, don't go lower.
# -----------------------------
MIN_SPEED        = 3.0
CHASE_FWD_SPEED  = 3.0

# -----------------------------
# timing
# -----------------------------
SETTLE_SEC       = 0.2   # short pause between state changes so the wheels
MINI_SETTLE_SEC  = 0.2   # actually stop before the next state runs

# pause after we catch a ball before going back to driving forward.
# used to be 2.0 but the caught_colors set means we won't re-detect the
# same ball anyway, so we can move on quicker.
LOST_WAIT_SEC = 0.5

# after the chase ends (ball goes under the camera blind spot), drive
# straight forward for this long to physically push the ball into the scoop.
# without this the robot stops right when the ball disappears and the ball
# never makes it into the compartment. tune up if it's barely making it in,
# down if we're overshooting.
CHASE_FINISH_SEC = 3.0

# -----------------------------
# centering
# the scuttle can rotate at lower commands than it can drive forward (both
# wheels working against each other beats friction easier than both pushing
# the same way). so CENTER has its own min-speed below MIN_SPEED.
# -----------------------------
CENTER_DEADBAND   = 0.30   # if the error is smaller than this, we call it centered.
                           # had it at 0.15 first but it kept oscillating across
                           # zero — wider deadband fixed it. CHASE corrects the
                           # leftover offset anyway.
CENTER_TURN_GAIN  = 2.0    # P gain. was 4.0 originally, dropped after the
                           # smoothing change.
CENTER_MIN_TURN   = 1.9    # below this the wheels just don't actually turn.
                           # tried 1.3 to be gentler but they stalled.
CENTER_MAX_SPEED  = 1.9    # cap. has to be >= MIN_TURN otherwise the floor
                           # gets clamped right back down and we never move.
CENTER_SMOOTHING  = 0.5    # exponential smoothing on the error.
                           # 0 = react every frame, higher = laggier.
CENTER_HOLD_SEC   = 0.3    # need to stay inside the deadband this long
                           # before we say centering is done

# -----------------------------
# CHASE / CHASE_BLUE steering
# closed-loop heading correction while we drive forward. CENTER gets us
# pointed roughly at the ball, this keeps us pointed at it as we close in.
# -----------------------------
APPROACH_DEADBAND  = 0.15
APPROACH_TURN_GAIN = 2.0
APPROACH_MAX_TURN  = 1.0    # keep <= 1.0 so wheels stay in [2.0, 4.0] at
                            # speed 3.0 — never below the stall floor.
APPROACH_SMOOTHING = 0.3

# offsets that bias the ball into the correct side of the scoop.
# the scoop has a divider down the middle — if the ball goes dead-center
# it just hits the divider and bounces off. so we shift the "centered"
# point off to one side so the ball slides into a compartment.
#  positive = ball ends up on robot's LEFT side
#  negative = ball ends up on robot's RIGHT side
CHASE_TARGET_OFFSET      =  0.3   # pink -> left
CHASE_BLUE_TARGET_OFFSET = -0.5   # blue -> right (had to push this one
                                  # higher than pink, the right compartment
                                  # geometry was a bit different)

# -----------------------------
# forward drive + lidar
# -----------------------------
FORWARD_DRIVE_SPEED = 3.0

# lidar is only checked while we're searching (FORWARD_DRIVE / OBSTACLE_WAIT).
# during a chase we WANT to drive into the ball — if we polled lidar there,
# the robot would stop just before contact thinking the ball is an obstacle.
OBSTACLE_DIST_M = 0.5    # anything closer than this in front = stop
FRONT_CONE_DEG  = 30     # half-angle of the "in front of me" cone we care
                         # about. lidar sees 270° total, this picks the
                         # ~60° slice straight ahead.

# -----------------------------
# color detection helper
# -----------------------------
def find_color(image, hsv_lo, hsv_hi):
    """largest blob in the given hsv range, or None."""
    image = cv2.resize(image, (size_w, size_h))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_lo, hsv_hi)
    # open then close to clean up the mask — kills speckle, fills small holes
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
    """check pink first, then blue. first hit wins.
    ignore_colors lets us skip a color we've already caught this run."""
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
# lidar obstacle check
# -----------------------------
def lidar_obstacle_in_front(lidar):
    """returns (blocked, dist_m, angle_deg). blocked=True if the nearest
    point inside the forward cone is within OBSTACLE_DIST_M.
    if no scan yet OR no points in cone, returns (False, None, None)."""
    scan = lidar.get()
    if scan is None:
        return (False, None, None)
    try:
        # scan is (N,2): col 0 = distance, col 1 = angle in degrees
        in_cone = scan[np.abs(scan[:, 1]) <= FRONT_CONE_DEG]
        if in_cone.shape[0] == 0:
            return (False, None, None)
        # filter out the bogus near-zero readings (per L2_vector.getValid)
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
# drive helpers — direct commands, no ramping
# -----------------------------
def stop():
    sc.driveOpenLoop(np.array([0.0, 0.0]))

def drive_forward(speed):
    sc.driveOpenLoop(np.array([speed, speed]))

def drive_forward_steered(speed, turn):
    """forward at `speed` plus a steering bias `turn`.
    L = speed + turn, R = speed - turn
    so positive turn slows the right wheel -> robot curves right."""
    sc.driveOpenLoop(np.array([speed + turn, speed - turn]))

# -----------------------------
# main loop
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

    # spin up the lidar. it runs on a background thread (see L1_lidar.py),
    # we just call .get() to grab the latest scan.
    # if it fails to connect, lidar_obstacle_in_front() will just always
    # return "not blocked" and the robot drives without obstacle stops.
    print("Connecting to LIDAR...")
    lidar = Lidar()
    lidar.connect()
    lidar_thread = lidar.run()
    sleep(1.0)   # give the thread a sec to actually grab a scan
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

    # ----- state vars -----
    # we start in FORWARD_DRIVE and stay in it whenever we're searching.
    # there used to be a left/right sweep state but we ripped that out.
    state             = 'FORWARD_DRIVE'
    phase_end         = time()
    settle_until      = 0.0
    next_state        = None
    current_settle    = SETTLE_SEC
    smoothed_error    = 0.0
    approach_error    = 0.0    # separate from smoothed_error so CENTER and
                               # CHASE don't stomp on each other's smoothing
    center_hold_start = None
    target_color      = None   # 'PINK' or 'BLUE' while we're locked on
    caught_colors     = set()  # already-caught balls. find_target skips these.

    def go_to_via_settle(new_state, mini=False):
        """all transitions go through SETTLE first.
        mini=True is the shorter pause (used between sub-bouts)."""
        nonlocal state, settle_until, next_state, smoothed_error, approach_error
        nonlocal center_hold_start, current_settle
        current_settle = MINI_SETTLE_SEC if mini else SETTLE_SEC
        state = 'SETTLE'
        settle_until = time() + current_settle
        next_state = new_state
        # don't reset smoothed_error if we're going INTO center — keeping it
        # avoids a jolt on the first frame. only wipe when we're leaving the
        # whole chase flow (i.e. heading into LOST_WAIT).
        if new_state == 'LOST_WAIT':
            smoothed_error = 0.0
        # but DO reset approach_error when starting a fresh chase — old
        # smoothed values from the previous ball would mess with the new one
        if new_state in ('CHASE', 'CHASE_BLUE'):
            approach_error = 0.0
        center_hold_start = None
        tag = 'MINI-SETTLE' if mini else 'SETTLE'
        print(f"-> {tag} {current_settle:.2f}s, then {new_state}")

    try:
        while True:
            sleep(0.05)   # ~20 Hz

            ret, image = camera.read()
            if not ret:
                print("Failed to retrieve image!")
                break

            color, ball = find_target(image, ignore_colors=caught_colors)

            # ============= SETTLE =============
            # short stop between every state. this whole pattern came out
            # of early testing — direct transitions caused the wheels to
            # carry residual velocity into the next state and the robot
            # would jerk or skip a state entirely. forcing a brief stop
            # in between fixed it.
            if state == 'SETTLE':
                stop()
                if time() >= settle_until:
                    # init for the state we're about to enter
                    if next_state == 'CENTER':
                        # keep smoothed_error to avoid jolt
                        center_hold_start = None
                    elif next_state == 'LOST_WAIT':
                        phase_end = time() + LOST_WAIT_SEC
                    elif next_state == 'CHASE_FINISH':
                        phase_end = time() + CHASE_FINISH_SEC
                    elif next_state == 'FORWARD_DRIVE':
                        pass   # nothing to set up
                    elif next_state == 'OBSTACLE_WAIT':
                        pass   # no timer, we sit until lidar clears
                    print(f"SETTLE done -> {next_state}")
                    state = next_state
                    next_state = None
                else:
                    remaining = settle_until - time()
                    print(f"SETTLE | {remaining:.2f}s left (-> {next_state})")
                continue

            # ============= FORWARD_DRIVE =============
            # cruise forward, watch for balls and obstacles.
            # check ball FIRST so we don't get stuck waiting on a "wall"
            # that's actually just a ball close to us. (bit of a hack but
            # it works because lidar can't really see ping pong balls
            # anyway — they're too small / smooth.)
            if state == 'FORWARD_DRIVE':
                if ball is not None:
                    x, y, w, h = ball
                    stop()
                    target_color = color
                    print(f"!! {color} BALL DETECTED while driving "
                          f"(w={w}px) -> SETTLE, then CENTER")
                    go_to_via_settle('CENTER')
                    continue
                # no ball — check lidar
                blocked, dist, ang = lidar_obstacle_in_front(lidar)
                if blocked:
                    stop()
                    print(f"!! LIDAR OBSTACLE (dist={dist:.2f}m @ {ang:+.1f}°) "
                          f"-> stop, wait for it to clear")
                    go_to_via_settle('OBSTACLE_WAIT')
                    continue
                # all clear, keep going
                drive_forward(FORWARD_DRIVE_SPEED)
                if dist is not None:
                    print(f"FORWARD_DRIVE | searching... nearest={dist:.2f}m "
                          f"@ {ang:+.1f}° (clear)")
                else:
                    print("FORWARD_DRIVE | searching... (no LIDAR data)")
                continue

            # ============= OBSTACLE_WAIT =============
            # something's in the way. sit here until it moves OR a ball
            # appears (which overrides — we'd rather chase the ball).
            if state == 'OBSTACLE_WAIT':
                stop()
                if ball is not None:
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
                # path is clear, get moving again
                if dist is not None:
                    print(f"OBSTACLE_WAIT | path clear (nearest={dist:.2f}m) "
                          f"-> FORWARD_DRIVE")
                else:
                    print("OBSTACLE_WAIT | path clear (no LIDAR data) "
                          "-> FORWARD_DRIVE")
                go_to_via_settle('FORWARD_DRIVE')
                continue

            # ============= CENTER =============
            # rotate in place until the ball is roughly in front of us.
            # "roughly" because the deadband is wide — CHASE will clean up
            # whatever residual heading error we leave behind.
            if state == 'CENTER':
                if ball is None:
                    # we lost it. probably stale frame / one-frame dropout.
                    stop()
                    smoothed_error = 0.0
                    center_hold_start = None
                    print(f"CENTER ({target_color}) | lost target")
                    target_color = None
                    go_to_via_settle('LOST_WAIT')
                    continue

                x, y, w, h = ball
                center_x = x + w / 2.0
                # normalize to roughly [-1, +1] across frame width
                raw_error = (center_x - size_w / 2.0) / (size_w / 2.0)
                smoothed_error = (CENTER_SMOOTHING * smoothed_error
                                  + (1.0 - CENTER_SMOOTHING) * raw_error)

                # inside deadband -> we're "centered enough", just hold for
                # CENTER_HOLD_SEC to make sure it sticks (sometimes the
                # error oscillates briefly across zero and we don't want
                # to declare done on a single frame)
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

                # outside deadband — turn proportional to error.
                # subtract sign*deadband so the turn doesn't jump straight
                # from 0 to MIN_TURN at the deadband edge — smoother handoff.
                center_hold_start = None
                sign = 1.0 if smoothed_error > 0 else -1.0
                effective_error = smoothed_error - sign * CENTER_DEADBAND
                turn = CENTER_TURN_GAIN * effective_error

                # floor it to MIN_TURN so the wheels actually rotate,
                # then cap to MAX_SPEED
                if abs(turn) < CENTER_MIN_TURN:
                    turn = CENTER_MIN_TURN * sign
                turn = max(-CENTER_MAX_SPEED, min(CENTER_MAX_SPEED, turn))

                # sign convention on this scuttle (verified by watching
                # which way the robot actually spun in test runs):
                #   err<0  -> ball on left  -> camera should pan left  -> L=-, R=+
                #   err>0  -> ball on right -> camera should pan right -> L=+, R=-
                # turn has same sign as err. so L = +turn, R = -turn.
                left_speed  = +turn
                right_speed = -turn
                sc.driveOpenLoop(np.array([left_speed, right_speed]))
                print(f"CENTER ({target_color}) | err={smoothed_error:+.2f}  "
                      f"L={left_speed:+.2f} R={right_speed:+.2f}")
                continue

            # ============= CHASE (pink) =============
            # drive at the ball with steering correction. exit when the
            # ball gets big enough OR drops out of frame (we've driven
            # over it). either way -> CHASE_FINISH for the final push.
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
                        # ball's right in front of us, push it in
                        stop()
                        print(f"CHASE | reached pink (width={w}px >= {PINK_STOP_WIDTH}) — pushing into scoop")
                        caught_colors.add('PINK')
                        print(f"CHASE | caught PINK; will ignore PINK from now on. caught={sorted(caught_colors)}")
                        target_color = None
                        go_to_via_settle('CHASE_FINISH')
                    else:
                        # closed-loop steering with the offset baked into
                        # the error. positive offset shifts the "centered"
                        # point to the LEFT side of frame, so the ball
                        # ends up on robot's left -> left compartment.
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

            # ============= CHASE_BLUE =============
            # same as CHASE but with the negative offset so the ball
            # lands in the right compartment instead. could probably DRY
            # this up with CHASE but it reads clearer as two states.
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

            # ============= CHASE_FINISH =============
            # straight forward push, no steering, for CHASE_FINISH_SEC.
            # this is what physically gets the ball into the scoop after
            # CHASE drops out — the ball goes under the camera's blind spot
            # before reaching the compartment, so we have to push blind.
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
            # short stationary pause after a chase finishes. mostly there
            # to give the wheels a sec to actually stop before we start
            # driving forward again.
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
        # kill the lidar thread cleanly. wrapped in try/except because
        # if lidar never connected, lidar.kill() throws.
        try:
            lidar.kill(lidar_thread)
        except Exception as e:
            print(f"LIDAR shutdown error (ignored): {e}")
        print("Exiting.")

if __name__ == '__main__':
    main()