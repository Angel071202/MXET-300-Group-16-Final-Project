# L3_color_tracking_hardcoded_dual.py
# Detects either of 2 calibrated colors using HSV ranges.
# Behavior:
#   1) Search for target
#   2) Turn to face target
#   3) When target is centered, stop for 3 seconds
#   4) Move forward slowly so the vacuum can suck the balls
#   5) Stop again

import cv2
import numpy as np
import L2_speed_control as sc
import L2_inverse_kinematics as ik
import L2_kinematics as kin
import netifaces as ni
from time import sleep
from math import pi

# -----------------------------
# GET IP FOR MJPG CAMERA STREAM
# -----------------------------
def getIp():
    for interface in ni.interfaces()[1:]:
        try:
            ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
            return ip
        except KeyError:
            continue
    return 0

# -----------------------------
# CAMERA SETTINGS
# -----------------------------
stream_ip = getIp()
camera_input = 'http://' + stream_ip + ':8090/?action=stream' if stream_ip else 0

size_w = 240
size_h = 160
fov = 1.0   # camera field of view in radians (estimate)

# -----------------------------
# HARD-CODED COLOR 1 HSV VALUES
# Replace these with your calibrated values
# -----------------------------
#Blue
H1_MIN = 55
S1_MIN = 95
V1_MIN = 210

H1_MAX = 110
S1_MAX = 165
V1_MAX = 255

# -----------------------------
# HARD-CODED COLOR 2 HSV VALUES
# Replace these with your calibrated values
# -----------------------------
#Pink
H2_MIN = 0
S2_MIN = 55
V2_MIN = 250

H2_MAX = 110
S2_MAX = 165
V2_MAX = 255

# -----------------------------
# ROBOT BEHAVIOR SETTINGS
# -----------------------------
angle_margin = 0.12          # target considered centered if |angle| is below this
min_detect_width = 12        # ignore very small detections
slow_forward_speed = 0.10    # forward speed during vacuum approach
turn_gain = 1.1              # turning gain
pause_time = 3.0             # stop time after detecting target
forward_time = 2.0           # how long to move forward slowly

# Optional cooldown so it doesn't instantly retrigger again
cooldown_time = 1.0

def main():
    # Try local camera first
    camera = cv2.VideoCapture(0)

    # If local camera fails, try MJPG stream
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_input)

    if not camera.isOpened():
        print("Failed to open camera.")
        return

    camera.set(3, size_w)
    camera.set(4, size_h)

    try:
        while True:
            sleep(0.05)

            ret, image = camera.read()
            if not ret:
                print("Failed to retrieve image!")
                break

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width, channels = hsv.shape

            # -----------------------------------
            # CREATE MASKS FOR BOTH COLOR RANGES
            # -----------------------------------
            mask1 = cv2.inRange(
                hsv,
                (H1_MIN, S1_MIN, V1_MIN),
                (H1_MAX, S1_MAX, V1_MAX)
            )

            mask2 = cv2.inRange(
                hsv,
                (H2_MIN, S2_MIN, V2_MIN),
                (H2_MAX, S2_MAX, V2_MAX)
            )

            # Combine both masks
            thresh = cv2.bitwise_or(mask1, mask2)

            # Clean up noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            cnts = cv2.findContours(
                mask.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )[-2]

            # If no target found, stop
            if len(cnts) == 0:
                print("No targets")
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                continue

            # Use largest contour
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Ignore tiny noise blobs
            if w < min_detect_width:
                print("Target too small")
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                continue

            center = (int(x + 0.5 * w), int(y + 0.5 * h))
            angle = ((center[0] / width) - 0.5) * fov

            wheel_measured = kin.getPdCurrent()

            # -----------------------------------
            # IF TARGET IS CENTERED: STOP, WAIT, MOVE FORWARD
            # -----------------------------------
            if abs(angle) < angle_margin:
                sc.driveOpenLoop(np.array([0.0, 0.0]))
                print("Target detected and centered")
                print("Stopping for 3 seconds...")
                sleep(pause_time)

                print("Moving forward slowly...")
                sc.driveOpenLoop(np.array([slow_forward_speed, 0.0]))
                sleep(forward_time)

                sc.driveOpenLoop(np.array([0.0, 0.0]))
                print("Approach complete")
                sleep(cooldown_time)
                continue

            # -----------------------------------
            # OTHERWISE TURN TOWARD TARGET
            # -----------------------------------
            wheel_speed = ik.getPdTargets(np.array([0.0, -turn_gain * angle]))
            sc.driveClosedLoop(wheel_speed, wheel_measured, 0)

            print(
                "Angle:", round(angle, 3),
                "| Box width:", w,
                "| Target L/R:", wheel_speed[0], wheel_speed[1],
                "| Measured L/R:", wheel_measured[0], wheel_measured[1]
            )

    except KeyboardInterrupt:
        pass

    finally:
        sc.driveOpenLoop(np.array([0.0, 0.0]))
        camera.release()
        print("Exiting Color Tracking.")

if __name__ == '__main__':
    main()