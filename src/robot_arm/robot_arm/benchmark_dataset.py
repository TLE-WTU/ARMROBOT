"""
Standardized Benchmark Dataset for Robotic Grasping.

Generates high-fidelity, physically calibrated 3D point clouds for 7 object categories
under three noise/sparsity conditions:
1. 'mug': Cylindrical body with an asymmetrical handle.
2. 'duck': Organic, free-form asymmetric curved body.
3. 'torus': Hollow donut shape with an empty geometric center.
4. 'glass_bottle': Tall container with a narrow neck and wide base.
5. 'box_package': Prismatic cuboid with planar faces and sharp edges.
6. 'flat_disc': Extremely thin object (height 3mm) challenging for table clearance.
7. 'dense_clutter': Multiple closely packed objects with adjacent contact boundaries.
"""

import math
from typing import Dict, List, Tuple
import numpy as np


class BenchmarkDataset:
    """Provides reproducible 3D point cloud samples for robotic grasp evaluation."""

    def __init__(self, table_z: float = 0.225, default_seed: int = 42):
        self.table_z = table_z
        self.default_seed = default_seed

    def generate_mug(self, n_points: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cylinder body (r=0.035m, h=0.08m) + curved handle extending in +X.
        Returns: (points_xyz, true_centroid)
        """
        rng = np.random.default_rng(seed)
        n_body = int(n_points * 0.80)
        n_handle = n_points - n_body

        # Cylinder body
        theta = rng.uniform(0, 2 * np.pi, n_body)
        r = 0.035 + rng.normal(0, 0.001, n_body)
        z = rng.uniform(self.table_z + 0.002, self.table_z + 0.082, n_body)
        body_x = 0.25 + r * np.cos(theta)
        body_y = 0.00 + r * np.sin(theta)

        # Handle arc extending in +X
        phi = rng.uniform(-np.pi / 2, np.pi / 2, n_handle)
        handle_x = 0.25 + 0.035 + 0.015 * np.cos(phi)
        handle_y = 0.00 + rng.normal(0, 0.002, n_handle)
        handle_z = (self.table_z + 0.042) + 0.025 * np.sin(phi)

        x = np.concatenate([body_x, handle_x])
        y = np.concatenate([body_y, handle_y])
        z = np.concatenate([z, handle_z])

        points = np.column_stack([x, y, z])
        true_centroid = np.array([0.25, 0.00, self.table_z + 0.042])
        return points, true_centroid

    def generate_duck(self, n_points: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Organic asymmetric duck shape (body ellipsoid + head sphere + bill)."""
        rng = np.random.default_rng(seed)
        n_body = int(n_points * 0.65)
        n_head = int(n_points * 0.25)
        n_bill = n_points - n_body - n_head

        # Body: Ellipsoid elongated along Y
        u = rng.uniform(0, 2 * np.pi, n_body)
        v = rng.uniform(0, np.pi, n_body)
        body_x = 0.25 + 0.030 * np.sin(v) * np.cos(u)
        body_y = 0.00 + 0.045 * np.sin(v) * np.sin(u)
        body_z = (self.table_z + 0.025) + 0.020 * np.cos(v)

        # Head: Sphere shifted towards +Y and +Z
        u_h = rng.uniform(0, 2 * np.pi, n_head)
        v_h = rng.uniform(0, np.pi, n_head)
        head_x = 0.25 + 0.018 * np.sin(v_h) * np.cos(u_h)
        head_y = 0.035 + 0.018 * np.sin(v_h) * np.sin(u_h)
        head_z = (self.table_z + 0.048) + 0.018 * np.cos(v_h)

        # Bill: Small cone extending in +Y
        bill_x = 0.25 + rng.normal(0, 0.006, n_bill)
        bill_y = 0.052 + rng.uniform(0, 0.015, n_bill)
        bill_z = (self.table_z + 0.046) + rng.normal(0, 0.004, n_bill)

        x = np.concatenate([body_x, head_x, bill_x])
        y = np.concatenate([body_y, head_y, bill_y])
        z = np.concatenate([body_z, head_z, bill_z])

        points = np.column_stack([x, y, z])
        true_centroid = np.array([0.25, 0.01, self.table_z + 0.030])
        return points, true_centroid

    def generate_torus(self, n_points: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Torus with major radius R=0.035m, minor radius r=0.012m lying flat."""
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 2 * np.pi, n_points)
        v = rng.uniform(0, 2 * np.pi, n_points)
        R = 0.035
        r = 0.012

        x = 0.25 + (R + r * np.cos(v)) * np.cos(u)
        y = 0.00 + (R + r * np.cos(v)) * np.sin(u)
        z = (self.table_z + 0.015) + r * np.sin(v)

        points = np.column_stack([x, y, z])
        true_centroid = np.array([0.25, 0.00, self.table_z + 0.015])
        return points, true_centroid

    def generate_bottle(self, n_points: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Tall cylinder base (r=0.030m, h=0.09m) + narrow neck (r=0.012m, h=0.05m)."""
        rng = np.random.default_rng(seed)
        n_base = int(n_points * 0.70)
        n_neck = n_points - n_base

        # Base body
        theta_b = rng.uniform(0, 2 * np.pi, n_base)
        r_b = 0.030 + rng.normal(0, 0.001, n_base)
        z_b = rng.uniform(self.table_z + 0.002, self.table_z + 0.090, n_base)
        bx = 0.25 + r_b * np.cos(theta_b)
        by = 0.00 + r_b * np.sin(theta_b)

        # Narrow neck
        theta_n = rng.uniform(0, 2 * np.pi, n_neck)
        r_n = 0.012 + rng.normal(0, 0.001, n_neck)
        z_n = rng.uniform(self.table_z + 0.090, self.table_z + 0.140, n_neck)
        nx = 0.25 + r_n * np.cos(theta_n)
        ny = 0.00 + r_n * np.sin(theta_n)

        points = np.column_stack([np.concatenate([bx, nx]), np.concatenate([by, ny]), np.concatenate([z_b, z_n])])
        true_centroid = np.array([0.25, 0.00, self.table_z + 0.065])
        return points, true_centroid

    def generate_box(self, n_points: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Rectangular cuboid (length 0.08m, width 0.05m, height 0.06m) rotated by 30 degrees."""
        rng = np.random.default_rng(seed)
        yaw = math.radians(30.0)
        lx, ly, lz = 0.08, 0.05, 0.06

        # Sample points uniformly on the 6 surfaces
        pts_local = []
        for _ in range(n_points):
            face = rng.integers(0, 6)
            if face == 0:  # +X
                pts_local.append([lx / 2, rng.uniform(-ly / 2, ly / 2), rng.uniform(0, lz)])
            elif face == 1:  # -X
                pts_local.append([-lx / 2, rng.uniform(-ly / 2, ly / 2), rng.uniform(0, lz)])
            elif face == 2:  # +Y
                pts_local.append([rng.uniform(-lx / 2, lx / 2), ly / 2, rng.uniform(0, lz)])
            elif face == 3:  # -Y
                pts_local.append([rng.uniform(-lx / 2, lx / 2), -ly / 2, rng.uniform(0, lz)])
            else:  # Top
                pts_local.append([rng.uniform(-lx / 2, lx / 2), rng.uniform(-ly / 2, ly / 2), lz])

        local_arr = np.array(pts_local)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rot_mat = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

        rotated = np.dot(local_arr, rot_mat.T)
        rotated[:, 0] += 0.25
        rotated[:, 1] += 0.00
        rotated[:, 2] += self.table_z + 0.002

        true_centroid = np.array([0.25, 0.00, self.table_z + 0.032])
        return rotated, true_centroid

    def generate_flat_disc(self, n_points: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Thin disc (radius 0.04m, thickness 0.003m) directly on table surface."""
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        r = np.sqrt(rng.uniform(0, 0.040**2, n_points))
        x = 0.25 + r * np.cos(theta)
        y = 0.00 + r * np.sin(theta)
        z = rng.uniform(self.table_z + 0.0005, self.table_z + 0.0035, n_points)

        points = np.column_stack([x, y, z])
        true_centroid = np.array([0.25, 0.00, self.table_z + 0.002])
        return points, true_centroid

    def generate_clutter(self, n_points: int = 2500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Three adjacent interacting objects (mug at (0.24, -0.05), box at (0.28, 0.04), duck at (0.23, 0.06))."""
        p_mug, _ = self.generate_mug(n_points=int(n_points * 0.40), seed=seed)
        p_mug[:, 0] -= 0.02
        p_mug[:, 1] -= 0.05

        p_box, _ = self.generate_box(n_points=int(n_points * 0.35), seed=seed + 1)
        p_box[:, 0] += 0.03
        p_box[:, 1] += 0.04

        p_duck, _ = self.generate_duck(n_points=n_points - len(p_mug) - len(p_box), seed=seed + 2)
        p_duck[:, 0] -= 0.02
        p_duck[:, 1] += 0.06

        points = np.vstack([p_mug, p_box, p_duck])
        true_centroid = np.array([0.25, 0.00, self.table_z + 0.040])
        return points, true_centroid

    def apply_condition(self, points: np.ndarray, condition: str, seed: int = 42) -> np.ndarray:
        """
        Applies data quality condition:
        - 'clean_dense': Unchanged high-density points.
        - 'sparse': Downsampled to 500 points.
        - 'noisy_dropout': Downsampled + 3mm Gaussian noise + 5% random outlier noise.
        """
        rng = np.random.default_rng(seed)
        pts = points.copy()

        if condition == "clean_dense":
            return pts

        elif condition == "sparse":
            target_n = min(500, pts.shape[0])
            idx = rng.choice(pts.shape[0], target_n, replace=False)
            return pts[idx]

        elif condition == "noisy_dropout":
            # 30% dropout
            n_keep = int(pts.shape[0] * 0.70)
            idx = rng.choice(pts.shape[0], n_keep, replace=False)
            pts = pts[idx]

            # 3mm Gaussian jitter
            pts += rng.normal(0, 0.003, pts.shape)

            # 5% random outliers in workspace volume
            n_outliers = int(pts.shape[0] * 0.05)
            ox = rng.uniform(0.18, 0.35, n_outliers)
            oy = rng.uniform(-0.15, 0.15, n_outliers)
            oz = rng.uniform(self.table_z, self.table_z + 0.12, n_outliers)
            outliers = np.column_stack([ox, oy, oz])
            return np.vstack([pts, outliers])

        return pts

    def load_sample(
        self, object_name: str, condition: str = "clean_dense", seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Loads a benchmark sample.
        Returns: (points_xyz, true_centroid, object_label)
        """
        gen_map = {
            "mug": self.generate_mug,
            "duck": self.generate_duck,
            "torus": self.generate_torus,
            "glass_bottle": self.generate_bottle,
            "box_package": self.generate_box,
            "flat_disc": self.generate_flat_disc,
            "dense_clutter": self.generate_clutter,
        }
        if object_name not in gen_map:
            raise ValueError(f"Unknown object '{object_name}'. Available: {list(gen_map.keys())}")

        raw_points, centroid = gen_map[object_name](seed=seed)
        points = self.apply_condition(raw_points, condition, seed=seed)
        label = f"{object_name}_{condition}"
        return points, centroid, label

    def get_all_test_cases(self) -> List[Tuple[str, str]]:
        """Returns list of (object_name, condition) pairs covering the benchmark matrix."""
        objects = ["mug", "duck", "torus", "glass_bottle", "box_package", "flat_disc", "dense_clutter"]
        conditions = ["clean_dense", "sparse", "noisy_dropout"]
        cases = []
        for obj in objects:
            for cond in conditions:
                cases.append((obj, cond))
        return cases
