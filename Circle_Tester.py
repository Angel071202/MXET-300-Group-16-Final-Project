# circularity_tester.py
# Test script for detecting colored balls using HSV + circularity

import cv2
import numpy as np
import netifaces as ni
from time import sleep

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

# -----------------------------
# COLOR 1 HSV
# Replace with your calibrated values
# -----------------------------
H1_MIN = 0
S1_MIN = 120
V1_MIN = 80

H1_MAX = 20
S1_MAX = 255
V1_MAX = 255

# -----------------------------
# COLOR 2 HSV
# Replace with your calibrated values
# -----------------------------
H2_MIN = 100
S2_MIN = 120
V2_MIN = 80

H2_MAX = 140
S2_MAX = 255
V2_MAX = 255

# -----------------------------
# CIRCLE FILTER SETTINGS
# -----------------------------
min_area = 120
max_area = 20000
min_width = 10
min_height = 10
aspect_ratio_tol = 0.35
min_circularity = 0.70

def find_best_ball(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    best_contour = None
    best_score = 0
    best_info = None

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < min_width or h < min_height:
            continue

        aspect_ratio = w / float(h)
        if abs(aspect_ratio - 1.0) > aspect_ratio_tol:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            best_contour = c
            best_info = (x, y, w, h, area, circularity, aspect_ratio)

    return best_contour, best_info

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_input)

    if not camera.isOpened():
        print("Failed to open camera.")
        return

    camera.set(3, size_w)
    camera.set(4, size_h)

    print("Press q to quit.")
    print("Show the colored ball in front of the camera.")

    try:
        while True:
            ret, image = camera.read()
            if not ret:
                print("Failed to retrieve image!")
                break

            image = cv2.resize(image, (size_w, size_h))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Make masks for both colors
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

            mask = cv2.bitwise_or(mask1, mask2)

            # Clean up noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contour, info = find_best_ball(mask)

            debug = image.copy()

            if contour is not None:
                x, y, w, h, area, circularity, aspect_ratio = info
                center = (int(x + w / 2), int(y + h / 2))

                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(debug, center, 3, (0, 0, 255), -1)

                text1 = "BALL DETECTED"
                text2 = f"A={int(area)} C={circularity:.2f} AR={aspect_ratio:.2f}"

                cv2.putText(debug, text1, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(debug, text2, (10, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print("BALL DETECTED | area =", int(area),
                      "| circularity =", round(circularity, 2),
                      "| aspect =", round(aspect_ratio, 2),
                      "| center =", center)
            else:
                cv2.putText(debug, "NO BALL", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("No ball detected")

            cv2.imshow("Camera", image)
            cv2.imshow("Mask", mask)
            cv2.imshow("Detection", debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Exiting tester.")

if __name__ == '__main__':
    main()