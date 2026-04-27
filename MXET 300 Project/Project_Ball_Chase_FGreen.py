# Project_Ball_Chase.py
# Detects a green box using HSV threshold + largest contour.
# Drives straight forward when it sees green, stops before hitting it, stays stopped.
#
# TO TUNE:
#   - edit the HSV values below to match your green
#   - STOP_WIDTH = pixel width at which the robot should stop.
#     Bigger number = stops closer to the box.
#     Smaller number = stops farther from the box.

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
# HSV VALUES — EDIT THESE TO TUNE COLOR (currently tuned for green)
# -----------------------------
PINK_H_MIN, PINK_S_MIN, PINK_V_MIN =  25, 195,  90
PINK_H_MAX, PINK_S_MAX, PINK_V_MAX =  40, 255, 140

# -----------------------------
# DETECTION SETTINGS
# -----------------------------
MIN_SIZE = 6     # from L3_image_filter: half-width threshold for a valid target

# -----------------------------
# BEHAVIOR SETTINGS
# -----------------------------
STOP_WIDTH         = 90    # px — stop as soon as the green box is this wide or wider.
                           # Raise this number to let the robot get closer before stopping.
                           # Lower it to stop farther away.
FWD_SPEED          = 3   # constant forward speed when driving

# Anti-restart latch: once we've stopped at the box, we only resume driving if
# the box convincingly leaves the view (lots of consecutive empty frames),
# OR it's detected small again (box was moved farther away on purpose).
LOST_FRAMES_TO_RESET = 40   # ~2 seconds at 20 fps of "no green" before we allow driving again
RESET_WIDTH          = 50   # if green reappears narrower than this, treat it as "new target, drive again"

# -----------------------------
# BALL DETECTION (logic from L3_image_filter.colorTracking)
# -----------------------------
def find_pink(hsv_image):
    """Return bounding rect (x,y,w,h) of the largest green blob, or None."""
    thresh = cv2.inRange(
        hsv_image,
        (PINK_H_MIN, PINK_S_MIN, PINK_V_MIN),
        (PINK_H_MAX, PINK_S_MAX, PINK_V_MAX),
    )

    kernel = np.ones((5, 5), np.uint8)
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
    print(f"HSV: ({PINK_H_MIN},{PINK_S_MIN},{PINK_V_MIN}) "
          f"to ({PINK_H_MAX},{PINK_S_MAX},{PINK_V_MAX})")
    print(f"Will stop when box width >= {STOP_WIDTH}px")

    # Latch state
    arrived = False           # once True, stay stopped until reset
    lost_counter = 0          # how many consecutive frames without green since arriving

    try:
        while True:
            sleep(0.05)

            ret, image = camera.read()
            if not ret:
                print("Failed to retrieve image!")
                break

            image = cv2.resize(image, (size_w, size_h))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            ball = find_pink(hsv)

            # --------------------------------------------------
            # State: ARRIVED (latched at the box)
            # --------------------------------------------------
            if arrived:
                # Stay stopped. Motors get zero every frame no matter what.
                sc.driveOpenLoop(np.array([0.0, 0.0]))

                if ball is None:
                    lost_counter += 1
                    if lost_counter >= LOST_FRAMES_TO_RESET:
                        arrived = False
                        lost_counter = 0
                        print("Green gone for a while — resuming search.")
                    else:
                        print(f"Arrived. No green ({lost_counter}/{LOST_FRAMES_TO_RESET}).")
                else:
                    x, y, w, h = ball
                    # If green reappears clearly smaller than stop size, something moved:
                    # treat it as a new chase.
                    if w < RESET_WIDTH:
                        arrived = False
                        lost_counter = 0
                        print(f"Green moved away (w={w}px < {RESET_WIDTH}px) — resuming chase.")
                    else:
                        lost_counter = 0   # still seeing it up close, keep latched
                        print(f"Arrived. Holding. (w={w}px)")
                continue

            # --------------------------------------------------
            # State: SEARCHING / DRIVING
            # --------------------------------------------------

            # No green: stop, but don't latch
            if ball is None:
                print("No green detected. Stopping.")
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                continue

            x, y, w, h = ball

            # Close enough: stop AND latch
            if w >= STOP_WIDTH:
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                arrived = True
                lost_counter = 0
                print(f"Green box reached (w={w}px >= {STOP_WIDTH}px). Stopped and holding.")
                continue

            # Drive straight forward — force both wheels to the SAME speed in open loop.
            # Bypasses closed-loop per-wheel correction that can cause tiny drift.
            wheel_speed = np.array([FWD_SPEED, FWD_SPEED])
            sc.driveOpenLoop(wheel_speed)

            print(f"GREEN | width={w}px (stop at {STOP_WIDTH}px)"
                  f" | wheels L={round(wheel_speed[0],2)} R={round(wheel_speed[1],2)}")

    except KeyboardInterrupt:
        pass

    finally:
        sc.driveOpenLoop(np.array([0.0, 0.0]))
        camera.release()
        print("Exiting.")

if __name__ == '__main__':
    main()