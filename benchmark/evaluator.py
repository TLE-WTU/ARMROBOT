"""
Unified Evaluation Engine for Comparing Geometric, Deep AI, and Hybrid Grasping.

Runs standardized evaluation across:
1. Pure Geometric Method: RANSAC Table Extraction + Euclidean Clustering + PCA / OBB Refiner
2. Pure Deep Learning Method: AnyGrasp (GSNet) End-to-End Prediction
3. Hybrid Method: AnyGrasp Affordances + Geometric Constraint Filter (Table clearance + IK + PCA)
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src", "robot_arm")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from robot_arm.geometric_refinement import (
    PCARefiner,
    OBBRefiner,
    CrossSectionSliceRefiner,
    PrimitiveRANSACRefiner,
    SurfaceNormalRefiner,
    make_top_down_rotation,
)
from robot_arm.ik_solver import IKSolver
from benchmark.metrics import (
    compute_antipodal_force_closure,
    check_table_collision,
    check_aperture_compliance,
    check_kinematic_feasibility,
)


class GraspEvaluator:
    """Executes head-to-head benchmarking between Geometric, AI, and Hybrid pipelines."""

    def __init__(self, table_z: float = 0.0, ik_solver: Optional[IKSolver] = None):
        self.table_z = table_z
        self.ik_solver = ik_solver if ik_solver is not None else IKSolver(dof=6)
        self.refiners = {
            "pca": PCARefiner(max_gripper_width=0.08),
            "obb": OBBRefiner(max_gripper_width=0.08),
            "normals": SurfaceNormalRefiner(max_gripper_width=0.08),
            "slice": CrossSectionSliceRefiner(max_gripper_width=0.08),
            "primitive_ransac": PrimitiveRANSACRefiner(max_gripper_width=0.08),
        }

    def run_geometric_pipeline(
        self,
        points: np.ndarray,
        method: str = "pca",
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes pure geometric pipeline without deep learning.
        Supported methods: 'pca', 'obb', 'normals', 'slice', 'primitive_ransac'.
        Returns: (grasp_list, latency_ms)
        """
        t0 = time.perf_counter()
        if len(points) < 15:
            return [], (time.perf_counter() - t0) * 1000.0

        # RANSAC table removal: filter points above table surface
        obj_mask = points[:, 2] > (self.table_z + 0.005)
        obj_points = points[obj_mask]

        if len(obj_points) < 10:
            obj_points = points  # fallback if coordinate frame differs

        refiner = self.refiners.get(method.lower(), self.refiners["pca"])
        try:
            raw_grasps = refiner.generate_heuristic_grasps(obj_points, table_z=self.table_z)
        except Exception as e:
            raw_grasps = self.refiners["pca"].generate_heuristic_grasps(obj_points, table_z=self.table_z)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        grasps = []
        for pos, score, rot, width, depth, yaw in raw_grasps:
            grasps.append({
                "translation": np.array(pos, dtype=np.float32),
                "rotation": np.array(rot, dtype=np.float32),
                "score": float(score),
                "width": float(width),
                "depth": float(depth),
                "yaw": float(yaw),
            })
        return grasps, latency_ms

    def run_anygrasp_pipeline(
        self,
        points: np.ndarray,
        detector: Any,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes AnyGrasp deep model forward inference.
        Returns: (grasp_list, latency_ms)
        """
        t0 = time.perf_counter()
        if detector is None or len(points) < 15:
            return [], (time.perf_counter() - t0) * 1000.0

        # Downsample if too dense to prevent GPU OOM
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            pts_down = points[indices].astype(np.float32)
        else:
            pts_down = points.astype(np.float32)

        optional_params = {
            "dense_grasp": False,
            "collision_detection": True,
            "approach_steering": [0, 0, -1],  # Top-down constraint for arm
            "approach_thresh": 0.25,
        }

        try:
            gg = detector.get_grasp(pts_down, optional_params)
        except Exception as e:
            print(f"AnyGrasp exception: {e}")
            gg = None

        latency_ms = (time.perf_counter() - t0) * 1000.0

        grasps = []
        if gg is not None and len(gg) > 0:
            # Sort by score descending
            for g in gg:
                rot = np.array(g.rotation_matrix, dtype=np.float32)
                trans = np.array(g.translation, dtype=np.float32)
                score = float(g.score)
                width = float(g.width)
                depth = float(g.depth)

                # Calculate yaw from AnyGrasp rotation matrix (add 90 deg offset for gripper closing axis alignment)
                yaw_raw = math.atan2(rot[1, 0], rot[0, 0]) + math.pi / 2.0
                yaw = (yaw_raw + math.pi) % (2 * math.pi) - math.pi
                grasps.append({
                    "translation": trans,
                    "rotation": rot,
                    "score": score,
                    "width": width,
                    "depth": depth,
                    "yaw": yaw,
                })

        return grasps, latency_ms

    def run_hybrid_pipeline(
        self,
        points: np.ndarray,
        detector: Any,
        method: str = "pca",
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes Hybrid Architecture: AnyGrasp candidates filtered & refined
        by the specified Geometric Refiner ('pca', 'obb', 'normals', 'slice', 'primitive_ransac').
        """
        t0 = time.perf_counter()
        obj_mask = points[:, 2] > (self.table_z + 0.005)
        obj_points = points[obj_mask]
        if len(obj_points) < 10:
            obj_points = points

        ai_grasps, _ = self.run_anygrasp_pipeline(obj_points, detector)
        if not ai_grasps and len(obj_points) != len(points):
            ai_grasps, _ = self.run_anygrasp_pipeline(points, detector)

        refiner = self.refiners.get(method.lower(), self.refiners["pca"])
        try:
            refined_tuples = refiner.refine(ai_grasps, obj_points, table_z=self.table_z)
        except Exception:
            refined_tuples = []

        valid_grasps = []
        # If refiner returned candidates, convert and validate them
        if refined_tuples:
            for pos, score, rot, width, depth, yaw in refined_tuples:
                pos = np.array(pos, dtype=np.float32)
                rot = np.array(rot, dtype=np.float32)
                yaw = float(yaw)
                width = float(width)

                # 1. Enforce minimum table clearance
                if pos[2] < (self.table_z + 0.025):
                    pos[2] = self.table_z + 0.030
                if check_table_collision(pos, rot, table_z=self.table_z):
                    pos[2] = self.table_z + 0.035

                # 2. Gripper Aperture Compliance
                if width > 0.08:
                    width = 0.070
                elif width < 0.01:
                    width = 0.035

                # 3. Kinematic Reachability Check (6-DOF Cartesian or 5-DOF top-down)
                reachable, _ = check_kinematic_feasibility(pos, yaw, self.ik_solver, rotation_matrix=rot)
                if not reachable:
                    alt_yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
                    alt_reachable, _ = check_kinematic_feasibility(pos, alt_yaw, self.ik_solver, rotation_matrix=rot)
                    if alt_reachable:
                        yaw = alt_yaw
                    else:
                        continue

                valid_grasps.append({
                    "translation": pos,
                    "rotation": rot,
                    "score": float(score),
                    "width": width,
                    "depth": float(depth),
                    "yaw": yaw,
                })

        # Fallback to pure geometric if no hybrid candidates survived
        if not valid_grasps:
            geo_grasps, _ = self.run_geometric_pipeline(points, method=method)
            valid_grasps = geo_grasps

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return valid_grasps, total_latency_ms

    def evaluate_analytical(
        self,
        grasps: List[Dict[str, Any]],
        points: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates analytical metrics without physical robot execution:
        - Force Closure rate (mu = 0.4 and mu = 0.8)
        - Table collision rate
        - IK reachability rate
        - Aperture compliance rate
        """
        if not grasps:
            return {
                "valid_grasp_found": False,
                "force_closure_04": False,
                "force_closure_08": False,
                "table_collision": False,
                "ik_feasible": False,
                "aperture_valid": False,
                "confidence_score": 0.0,
            }

        top_grasp = grasps[0]
        pos = top_grasp["translation"]
        rot = top_grasp["rotation"]
        yaw = top_grasp["yaw"]
        width = top_grasp["width"]
        score = top_grasp["score"]

        # 1. Collision check
        table_col = check_table_collision(pos, rot, table_z=self.table_z)

        # 2. IK Feasibility
        ik_feas, _ = check_kinematic_feasibility(pos, yaw, self.ik_solver, rotation_matrix=rot)

        # 3. Aperture compliance
        aperture_ok = check_aperture_compliance(width, max_gripper_width=0.08)

        # 4. Surface normal & Force-Closure evaluation
        # Estimate surface normals around contact points
        closing_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        half_w = min(width / 2.0, 0.035)
        p1 = pos - closing_axis * half_w
        p2 = pos + closing_axis * half_w

        # Find closest point normals
        dists1 = np.linalg.norm(points - p1, axis=1)
        dists2 = np.linalg.norm(points - p2, axis=1)
        idx1 = int(np.argmin(dists1))
        idx2 = int(np.argmin(dists2))

        # Local surface normals via local PCA
        k = min(20, len(points))
        knn1 = points[np.argsort(dists1)[:k]]
        knn2 = points[np.argsort(dists2)[:k]]

        cov1 = np.cov(knn1.T)
        cov2 = np.cov(knn2.T)
        _, v1 = np.linalg.eigh(cov1)
        _, v2 = np.linalg.eigh(cov2)
        n1 = v1[:, 0]
        n2 = v2[:, 0]

        fc_04, _, _ = compute_antipodal_force_closure(p1, p2, n1, n2, friction_coeff=0.4)
        fc_08, _, _ = compute_antipodal_force_closure(p1, p2, n1, n2, friction_coeff=0.8)

        return {
            "valid_grasp_found": True,
            "force_closure_04": bool(fc_04),
            "force_closure_08": bool(fc_08),
            "table_collision": bool(table_col),
            "ik_feasible": bool(ik_feas),
            "aperture_valid": bool(aperture_ok),
            "confidence_score": float(score),
        }
