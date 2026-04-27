"""
Standalone LIDAR test — verifies that the SICK TiM561 LIDAR is connected,
streaming data, and that the obstacle detection logic in
Project_Ball_Chase_Foward.py would work.

Run this on the Pi:
    python3 test_lidar.py

Expected output (every 0.5s):
    SCAN OK | nearest in front cone: dist=1.85m angle=+12.3°  CLEAR
    SCAN OK | nearest in front cone: dist=0.42m angle=-3.5°  *** BLOCKED ***

If you see "WARNING: lidar.get() returned None" repeatedly, the LIDAR
is not streaming. Most common causes:
  1. ethernet cable not connected to the Pi
  2. forgot to run `sudo ip addr add 192.168.6.1XX/24 dev eth0`
  3. LIDAR power cable not plugged in
  4. wrong SENSOR_IP in L1_lidar.py (check the number on your LIDAR)
"""

import time
import numpy as np
from L1_lidar import Lidar

# Same parameters as Project_Ball_Chase_Foward.py
OBSTACLE_DIST_M = 0.5
FRONT_CONE_DEG  = 30


def lidar_obstacle_in_front(lidar):
    scan = lidar.get()
    if scan is None:
        return (False, None, None, "no scan yet")
    in_cone = scan[np.abs(scan[:, 1]) <= FRONT_CONE_DEG]
    if in_cone.shape[0] == 0:
        return (False, None, None, "no points in cone")
    valids = in_cone[in_cone[:, 0] > 0.016]
    if valids.shape[0] == 0:
        return (False, None, None, "no valid points")
    idx = np.argmin(valids[:, 0])
    dist = float(valids[idx, 0])
    ang  = float(valids[idx, 1])
    return (dist < OBSTACLE_DIST_M, dist, ang, "ok")


def main():
    print("=" * 60)
    print("LIDAR test")
    print(f"  obstacle threshold: < {OBSTACLE_DIST_M}m within ±{FRONT_CONE_DEG}°")
    print("=" * 60)

    print("Connecting to LIDAR...")
    lidar = Lidar()
    lidar.connect()
    thread = lidar.run()
    print("Waiting 2s for scans to start...")
    time.sleep(2.0)

    if lidar.get() is None:
        print("\n*** WARNING: lidar.get() is None after 2 seconds ***")
        print("The LIDAR is not streaming. Check connections.\n")

    print("\nReading scans every 0.5s. Ctrl+C to stop.\n")

    try:
        i = 0
        while True:
            i += 1
            blocked, dist, ang, status = lidar_obstacle_in_front(lidar)

            # Also dump raw scan summary every 5 readings for debugging
            scan = lidar.get()
            if scan is None:
                print(f"[{i:3d}] WARNING: lidar.get() returned None — no data")
            else:
                n_total = scan.shape[0]
                n_valid = int(np.sum(scan[:, 0] > 0.016))
                in_cone = scan[np.abs(scan[:, 1]) <= FRONT_CONE_DEG]
                n_cone = in_cone.shape[0]
                n_cone_valid = int(np.sum(in_cone[:, 0] > 0.016))

                if dist is None:
                    print(f"[{i:3d}] SCAN | total={n_total} valid={n_valid} "
                          f"cone={n_cone} cone_valid={n_cone_valid} | {status}")
                else:
                    flag = "*** BLOCKED ***" if blocked else "CLEAR"
                    print(f"[{i:3d}] SCAN | total={n_total} valid={n_valid} "
                          f"cone={n_cone} cone_valid={n_cone_valid} | "
                          f"nearest in cone: dist={dist:.2f}m "
                          f"angle={ang:+.1f}°  {flag}")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            lidar.kill(thread)
        except Exception as e:
            print(f"shutdown error (ignored): {e}")
        print("Done.")


if __name__ == "__main__":
    main()