#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXTURES_DIR = os.path.join(PKG_ROOT, "materials", "textures")

def get_box_sdf(name, x, y, z=0.5, sx=0.5, sy=0.5, sz=1.0):
    tex_path = os.path.join(TEXTURES_DIR, "box_wood.png")
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <collision name="col">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name="vis">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <pbr><metal><albedo_map>{tex_path}</albedo_map></metal></pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

def get_cylinder_sdf(name, x, y, z=0.5, radius=0.4, length=1.0):
    tex_path = os.path.join(TEXTURES_DIR, "cylinder_metal.png")
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <collision name="col">
        <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
      </collision>
      <visual name="vis">
        <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <pbr><metal><albedo_map>{tex_path}</albedo_map></metal></pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

class SynchronizedFrameCollector(Node):
    def __init__(self):
        super().__init__("frame_collector")
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None

        self.create_subscription(Image, "/camera/rgb/image_raw", self.rgb_cb, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_cb, 10)

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def capture_pair(self):
        self.latest_rgb = None
        self.latest_depth = None
        timeout = time.time() + 3.0
        while (self.latest_rgb is None or self.latest_depth is None) and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.latest_rgb is None or self.latest_depth is None:
            raise TimeoutError("Failed to collect synchronized RGB and Depth frames from Gazebo.")
        return self.latest_rgb.copy(), self.latest_depth.copy()

def gz_spawn(name, sdf_content):
    clean_sdf = sdf_content.replace('"', '\\"').replace("\n", " ")
    cmd = [
        "gz", "service", "-s", "/world/room_world/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'sdf: "{clean_sdf}"'
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def gz_delete(name):
    cmd = [
        "gz", "service", "-s", "/world/room_world/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'name: "{name}", type: 2'
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    rclpy.init()
    collector = SynchronizedFrameCollector()
    dataset_dir = os.path.join(PKG_ROOT, "results", "raw_data")
    os.makedirs(dataset_dir, exist_ok=True)

    test_matrix = []
    # Box obstacle: distances along optical axis (y=0)
    for d in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        test_matrix.append(("box", d, 0.0))
    # Cylinder obstacle: distances along optical axis (y=0) and off-axis (y = +/- 1.0)
    for d in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        for y_off in [0.0, -1.0, 1.0]:
            test_matrix.append(("cylinder", d, y_off))

    print(f"Total benchmark placements to evaluate: {len(test_matrix)}")

    for idx, (shape, dist, y_off) in enumerate(test_matrix):
        model_name = f"obstacle_{shape}_{idx}"
        print(f"[{idx+1}/{len(test_matrix)}] Spawning {shape} at dist={dist:.1f}m, y_offset={y_off:.1f}m...")
        
        # Ensure previous model removed
        gz_delete(model_name)
        time.sleep(0.2)

        if shape == "box":
            sdf = get_box_sdf(model_name, x=dist, y=y_off, z=0.5, sx=0.5, sy=0.5, sz=1.0)
        else:
            sdf = get_cylinder_sdf(model_name, x=dist, y=y_off, z=0.5, radius=0.4, length=1.0)

        gz_spawn(model_name, sdf)
        
        # Wait for simulation and rendering pipeline to settle
        time.sleep(0.8)

        # Collect N=10 frames
        N_FRAMES = 10
        for f_idx in range(N_FRAMES):
            rgb, depth = collector.capture_pair()
            base_filename = f"{shape}_d{dist:.1f}_y{y_off:.1f}_f{f_idx}"
            np.save(os.path.join(dataset_dir, f"{base_filename}_rgb.npy"), rgb)
            np.save(os.path.join(dataset_dir, f"{base_filename}_gt.npy"), depth)
            time.sleep(0.08)

        gz_delete(model_name)
        time.sleep(0.3)

    print("Benchmarking dataset acquisition completed successfully.")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
