# Project_Ball_Chase.py
# Track a pink ball: turn in place to keep it in view.
# Detection logic matches L3_image_filter.py.
#
# TO TUNE: edit PINK_* values below and re-run.
# If robot still jitters:  lower TURN_GAIN, or widen DEADBAND, or raise SMOOTHING.
# If robot loses the ball:  raise TURN_GAIN, or narrow DEADBAND.

import cv2
import numpy as np
import L2_speed_control as sc
import L2_inverse_kinematics as ik
import L2_kinematics as kin
import netifaces as ni
from time import sleep

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
# DETECTION SETTINGS
# -----------------------------
MIN_SIZE = 6        # ignore blobs where 0.5*w <= this (same as filter)

# -----------------------------
# TRACKING / ANTI-JITTER SETTINGS
# -----------------------------
# error is normalized: -1.0 (far left) to +1.0 (far right), 0 = centered.
DEADBAND   = 0.25   # |error| below this = don't move. Wider = calmer, but ball can drift more.
TURN_GAIN  = 15.0   # how hard to turn when outside deadband. Lower = gentler.
MIN_SPEED  = 3.0    # minimum wheel command when turning — overcomes static friction so wheels actually move.
MAX_SPEED  = 8.0    # clamp on wheel command magnitude.
SMOOTHING  = 0.6    # 0 = no smoothing (react every frame), 0.9 = very smooth (slow to react).

# -----------------------------
# BALL DETECTION (matches L3_image_filter.py)
# -----------------------------
def find_ball(hsv, kernel):
    thresh = cv2.inRange(
        hsv,
        (PINK_H_MIN, PINK_S_MIN, PINK_V_MIN),
        (PINK_H_MAX, PINK_S_MAX, PINK_V_MAX),
    )
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    if 0.5 * w <= MIN_SIZE:
        return None

    return (x, y, w, h)

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

    kernel = np.ones((5, 5), np.uint8)

    # smoothed error carries across frames so one noisy reading can't snap the wheels
    smoothed_error = 0.0

    try:
        while True:
            sleep(0.05)

            ret, image = camera.read()
            if not ret:
                print("Failed to retrieve image!")
                break

            image = cv2.resize(image, (size_w, size_h))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            ball = find_ball(hsv, kernel)

            # --- No ball: stop and reset the smoother so the robot doesn't lurch
            #     when the ball reappears. ---
            if ball is None:
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                smoothed_error = 0.0
                print("No ball. Stopped.")
                continue

            x, y, w, h = ball
            center_x = x + w / 2.0

            # raw error: -1.0 (far left) to +1.0 (far right)
            raw_error = (center_x - size_w / 2.0) / (size_w / 2.0)

            # low-pass filter: mix some of the new reading into the running value.
            # SMOOTHING near 1 means "trust old value more" -> smoother but laggier.
            smoothed_error = SMOOTHING * smoothed_error + (1.0 - SMOOTHING) * raw_error

            # --- Deadband: inside this, the ball is "close enough to center". Don't move. ---
            if abs(smoothed_error) < DEADBAND:
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                print(f"PINK w={w}px  error={smoothed_error:+.2f}  [centered, idle]")
                continue

            # --- Outside deadband: turn toward the ball. ---
            # Only the part of the error beyond the deadband drives the turn,
            # so the command starts from 0 at the edge (no sudden jump).
            sign = 1.0 if smoothed_error > 0 else -1.0
            effective_error = smoothed_error - sign * DEADBAND

            turn = TURN_GAIN * effective_error

            # Enforce minimum magnitude so we don't send a tiny command that can't overcome friction.
            if abs(turn) < MIN_SPEED:
                turn = MIN_SPEED * sign
            turn = max(-MAX_SPEED, min(MAX_SPEED, turn))

            # ball on right (error > 0) -> spin right: L forward, R backward
            left_speed  = +turn
            right_speed = -turn

            wheel_speed = np.array([left_speed, right_speed])
            sc.driveOpenLoop(wheel_speed)

            print(f"PINK w={w}px  error={smoothed_error:+.2f}  "
                  f"L={left_speed:+.2f} R={right_speed:+.2f}")

    except KeyboardInterrupt:
        pass

    finally:
        sc.driveOpenLoop(np.array([0.0, 0.0]))
        camera.release()
        print("Exiting.")

if __name__ == '__main__':
    main()