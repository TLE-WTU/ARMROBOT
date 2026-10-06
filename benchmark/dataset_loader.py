"""
Standard Dataset & Sensor Data Loader for Grasp Benchmarking.

Supports:
1. Real-World RGB-D Sensor Captures (AnyGrasp / GraspNet RealSense/Kinect camera data).
2. Physical 3D CAD Mesh Scenes (PyBullet multi-view RGB-D rendering).
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image


class RealSensorDataLoader:
    """Loads and preprocesses real-world RGB-D camera captures."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_point_cloud(
        self,
        fx: float = 927.17,
        fy: float = 927.37,
        cx: float = 651.32,
        cy: float = 349.62,
        scale: float = 1000.0,
        depth_trunc: float = 1.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads color.png and depth.png, returning (points_xyz, colors_rgb, seg_mask).
        """
        color_path = os.path.join(self.data_dir, "color.png")
        depth_path = os.path.join(self.data_dir, "depth.png")
        mask_path = os.path.join(self.data_dir, "seg_mask.png")

        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"Missing color.png or depth.png in {self.data_dir}")

        colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        depths = np.array(Image.open(depth_path), dtype=np.float32)

        seg_mask = None
        if os.path.exists(mask_path):
            seg_mask = np.array(Image.open(mask_path))

        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        valid_mask = (points_z > 0.1) & (points_z < depth_trunc)
        points = np.stack([points_x, points_y, points_z], axis=-1)[valid_mask]
        colors_pts = colors[valid_mask]

        if seg_mask is not None:
            seg_mask_pts = seg_mask[valid_mask]
        else:
            seg_mask_pts = np.ones(len(points), dtype=np.int32)

        return points.astype(np.float32), colors_pts.astype(np.float32), seg_mask_pts
