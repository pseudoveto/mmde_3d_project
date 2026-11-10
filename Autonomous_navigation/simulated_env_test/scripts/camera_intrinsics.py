#!/usr/bin/env python3
import math
import numpy as np

def compute_pinhole_intrinsics(width=640, height=480, hfov_rad=math.pi / 2.0):
    # fx = (width / 2) / tan(hfov / 2)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    # Square pixel assumption in Gazebo
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    vfov_rad = 2.0 * math.atan((height / 2.0) / fy)
    vfov_deg = math.degrees(vfov_rad)

    print("================ CAMERA INTRINSICS ================")
    print(f"Resolution : {width} x {height}")
    print(f"HFOV       : {math.degrees(hfov_rad):.2f}° ({hfov_rad:.6f} rad)")
    print(f"VFOV       : {vfov_deg:.2f}° ({vfov_rad:.6f} rad)")
    print(f"fx, fy     : {fx:.4f}, {fy:.4f}")
    print(f"cx, cy     : {cx:.4f}, {cy:.4f}")
    print("Intrinsics Matrix K:\n", K)
    print("====================================================")
    return K

if __name__ == "__main__":
    compute_pinhole_intrinsics()
