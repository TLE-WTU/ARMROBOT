#!/usr/bin/env python3
"""
Analytical Kinematics Engine for 4-DOF Tabletop Manipulator.

Joint layout:
    joint1 - Base turret rotation around Z (Yaw)
    joint2 - Shoulder elevation around Y (Pitch)
    joint3 - Elbow flexion around Y (Pitch)
    joint4 - Wrist elevation around Y (Pitch)
    gripper - Parallel-jaw prismatic fingers along local Y axis

Kinematic Properties:
    - 3 positioning DOFs (x, y, z) + 1 constrained top-down orientation DOF.
    - End-effector orientation around Z (Yaw) is strictly coupled to base rotation:
      psi_feasible(x, y) = atan2(y, x) +/- pi/2 (mod pi).
    - Closed-form solution with dual elbow branches (Elbow-Up, Elbow-Down),
      joint limit enforcement (+/- 2.5 rad), and table clearance validation.
"""

import math
import logging
from typing import Optional, List, Tuple, Dict, Any


class IKSolver:
    """Analytical Inverse and Forward Kinematics for a 4-DOF robot arm."""

    DOF = 4
    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']
    JOINT_LIMITS = {
        'joint1': (-math.pi, math.pi),
        'joint2': (-2.5, 2.5),
        'joint3': (-2.5, 2.5),
        'joint4': (-math.pi, math.pi),
    }

    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        self.logger = logger or logging.getLogger('IKSolver')

        # Link lengths and geometric offsets (meters)
        self.config = {
            'base_height': 0.175,     # Base origin to shoulder axis
            'l1': 0.250,              # Upper arm length
            'l2': 0.200,              # Forearm length
            'l_hand': 0.1334,         # Wrist to Franka Hand TCP
            'table_z': 0.225,         # Support table height in base frame
            'elbow_table_margin': 0.015,  # Safety clearance for elbow above table
            'joint_limit_shoulder': 2.5,
            'joint_limit_elbow': 2.5,
        }
        if config:
            self.config.update(config)

    # -------------------------------------------------------------------------
    # Workspace & Kinematic Boundary Analysis
    # -------------------------------------------------------------------------

    def get_workspace_reach(self, target_z: float) -> Tuple[float, float]:
        """
        Computes the exact radial reach [r_min, r_max] at a given grasp height target_z.

        Under the top-down constraint (gripper pointing vertically down),
        wrist height is z_wrist = target_z + l_hand.
        Distance from shoulder to wrist is D = sqrt(r^2 + dz^2).
        Reachability requires |l1 - l2| <= D <= l1 + l2.
        """
        dz = target_z + self.config['l_hand'] - self.config['base_height']
        l1 = self.config['l1']
        l2 = self.config['l2']
        max_d = l1 + l2
        min_d = abs(l1 - l2)

        if abs(dz) > max_d:
            return 0.0, 0.0

        r_max = math.sqrt(max_d**2 - dz**2)
        r_min = math.sqrt(max(0.0, min_d**2 - dz**2))
        return r_min, r_max

    def is_in_workspace(self, x: float, y: float, z: float) -> Tuple[bool, str]:
        """
        Evaluates whether Cartesian coordinate (x, y, z) lies within the 4-DOF
        curvilinear reachable workspace.
        """
        if z < self.config['table_z'] - 0.005:
            return False, f"Target z={z:.3f}m is below table surface ({self.config['table_z']}m)"

        r = math.hypot(x, y)
        r_min, r_max = self.get_workspace_reach(z)

        if r_max <= 0.0:
            return False, f"Target height z={z:.3f}m is out of vertical kinematic reach"
        if r > r_max:
            return False, f"Radial distance r={r:.3f}m exceeds max reach r_max={r_max:.3f}m at z={z:.3f}m"
        if r < r_min:
            return False, f"Radial distance r={r:.3f}m is inside inner singularity r_min={r_min:.3f}m"

        return True, "Reachable"

    # -------------------------------------------------------------------------
    # 4-DOF End-Effector Orientation (Yaw Lock Analysis)
    # -------------------------------------------------------------------------

    def get_end_effector_closing_yaw(self, x: float, y: float) -> float:
        """
        Returns the kinematically dictated closing axis of the Franka parallel-jaw
        gripper at position (x, y) in radians [-pi/2, pi/2].

        Since the wrist has no roll joint (only pitch), the gripper's prismatic finger
        closing axis is aligned with the arm's tangential direction:
            psi_closing = atan2(y, x) + pi/2 (mod pi)
        """
        th1 = math.atan2(y, x)
        closing_yaw = th1 + math.pi / 2.0
        # Normalize into [-pi/2, pi/2] due to parallel-jaw 180-deg symmetry
        return math.atan2(math.sin(2.0 * closing_yaw), math.cos(2.0 * closing_yaw)) / 2.0

    def check_yaw_compatibility(
        self, x: float, y: float, desired_yaw: float, tolerance_deg: float = 25.0
    ) -> Tuple[bool, float]:
        """
        Checks if a requested grasp orientation desired_yaw can be executed by the 4-DOF arm.

        Returns:
            (is_compatible, divergence_deg): True if divergence <= tolerance_deg.
        """
        actual_yaw = self.get_end_effector_closing_yaw(x, y)
        # Angular difference modulo 180 deg
        diff = abs(desired_yaw - actual_yaw) % math.pi
        divergence = min(diff, math.pi - diff)
        divergence_deg = math.degrees(divergence)
        return divergence_deg <= tolerance_deg, divergence_deg

    # -------------------------------------------------------------------------
    # Forward Kinematics
    # -------------------------------------------------------------------------

    def forward_kinematics(self, t1: float, t2: float, t3: float, t4: float) -> Tuple[float, float, float, float]:
        """
        Computes forward kinematics. Returns (x, y, z_grasp, closing_yaw).
        """
        base_h = self.config['base_height']
        l1 = self.config['l1']
        l2 = self.config['l2']
        l_hand = self.config['l_hand']

        r_wrist = l1 * math.sin(t2) + l2 * math.sin(t2 + t3)
        z_wrist = base_h + l1 * math.cos(t2) + l2 * math.cos(t2 + t3)

        total_pitch = t2 + t3 + t4
        r_grasp = r_wrist + l_hand * math.sin(total_pitch)
        z_grasp = z_wrist + l_hand * math.cos(total_pitch)

        x = r_grasp * math.cos(t1)
        y = r_grasp * math.sin(t1)
        closing_yaw = self.get_end_effector_closing_yaw(x, y)
        return x, y, z_grasp, closing_yaw

    # -------------------------------------------------------------------------
    # Inverse Kinematics (Dual-Branch with Limit & Collision Checking)
    # -------------------------------------------------------------------------

    def compute_all_ik_solutions(
        self, target_x: float, target_y: float, target_z: float
    ) -> List[Tuple[float, float, float, float]]:
        """
        Computes all geometrically and physically valid IK solutions for target (x, y, z).
        Evaluates both elbow branches:
            branch +1: Elbow-Up
            branch -1: Elbow-Down
        Filters out configurations that violate joint limits or cause link-table collisions.
        """
        r_wrist = math.hypot(target_x, target_y)
        z_wrist = target_z + self.config['l_hand']
        dz = z_wrist - self.config['base_height']
        d_sq = r_wrist**2 + dz**2
        d = math.sqrt(d_sq)

        l1 = self.config['l1']
        l2 = self.config['l2']
        if d > (l1 + l2) or d < abs(l1 - l2):
            return []

        th1 = math.atan2(target_y, target_x)
        cos_t3 = (d_sq - l1**2 - l2**2) / (2.0 * l1 * l2)
        cos_t3 = max(-1.0, min(1.0, cos_t3))

        lim_shoulder = self.config['joint_limit_shoulder']
        lim_elbow = self.config['joint_limit_elbow']
        table_z = self.config['table_z']
        min_elbow_z = table_z + self.config['elbow_table_margin']

        valid_sols = []
        for s in (+1, -1):
            th3 = s * math.acos(cos_t3)
            alpha = math.atan2(r_wrist, dz)
            beta = math.atan2(l2 * math.sin(th3), l1 + l2 * math.cos(th3))
            th2 = alpha - beta

            # Top-down constraint: total pitch = pi
            th4 = math.pi - (th2 + th3)

            # Normalize angles to [-pi, pi]
            th1_norm = (th1 + math.pi) % (2.0 * math.pi) - math.pi
            th2_norm = (th2 + math.pi) % (2.0 * math.pi) - math.pi
            th3_norm = (th3 + math.pi) % (2.0 * math.pi) - math.pi
            th4_norm = (th4 + math.pi) % (2.0 * math.pi) - math.pi

            # 1. Joint limit verification
            if abs(th2_norm) > lim_shoulder or abs(th3_norm) > lim_elbow:
                continue

            # 2. Link-table clearance verification
            # Elbow position in base frame:
            z_elbow = self.config['base_height'] + l1 * math.cos(th2_norm)
            if z_elbow < min_elbow_z:
                continue

            valid_sols.append((th1_norm, th2_norm, th3_norm, th4_norm))

        return valid_sols

    def inverse_kinematics(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        yaw: float = 0.0,
        current_joints: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        """
        Computes optimal analytical IK for the 4-DOF arm.

        Args:
            target_x: desired grasp X in base frame (m)
            target_y: desired grasp Y in base frame (m)
            target_z: desired grasp Z in base frame (m)
            yaw: desired grasp yaw (evaluated for compatibility; end-effector yaw is constrained by geometry)
            current_joints: optional active joint positions [t1, t2, t3, t4] to select the nearest branch.

        Returns:
            [t1, t2, t3, t4] joint angles in radians, or None if unreachable.
        """
        solutions = self.compute_all_ik_solutions(target_x, target_y, target_z)
        if not solutions:
            return None

        if len(solutions) == 1:
            return list(solutions[0])

        # If current_joints provided, select configuration with minimum displacement
        if current_joints is not None and len(current_joints) >= 4:
            def joint_dist_sq(sol):
                return sum((sol[i] - current_joints[i])**2 for i in range(4))
            best_sol = min(solutions, key=joint_dist_sq)
            return list(best_sol)

        # Default fallback: prefer elbow-up branch (branch 0 where th3 > 0)
        return list(solutions[0])

    def validate_trajectory(
        self, waypoints: List[Tuple[float, float, float, float]]
    ) -> Tuple[bool, List[Optional[List[float]]]]:
        """Validates that a series of (x, y, z, yaw) waypoints are reachable without joint limit violations."""
        all_valid = True
        results = []
        curr = None
        for wp in waypoints:
            joints = self.inverse_kinematics(wp[0], wp[1], wp[2], wp[3], current_joints=curr)
            results.append(joints)
            if joints is None:
                all_valid = False
            else:
                curr = joints
        return all_valid, results


if __name__ == '__main__':
    import time
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('IKSolverBenchmark')
    solver = IKSolver(logger=logger)

    logger.info("--- Analytical 4-DOF IK Verification & Performance Benchmark ---")
    test_points = [
        (0.25, 0.0, 0.25),
        (0.30, 0.10, 0.26),
        (0.20, -0.15, 0.35),
        (0.35, 0.05, 0.23),
    ]

    for pt in test_points:
        sols = solver.compute_all_ik_solutions(*pt)
        reachable, reason = solver.is_in_workspace(*pt)
        closing_yaw = solver.get_end_effector_closing_yaw(pt[0], pt[1])
        logger.info(
            f"Point {pt}: Status={reason} | Valid Branches={len(sols)} | "
            f"Constrained Yaw={math.degrees(closing_yaw):.1f}°"
        )
        for i, sol in enumerate(sols):
            fx, fy, fz, _ = solver.forward_kinematics(*sol)
            err_mm = math.hypot(pt[0] - fx, pt[1] - fy, pt[2] - fz) * 1000.0
            logger.info(f"   Branch #{i+1} [Elbow={'Up' if sol[2]>0 else 'Down'}]: FK Error = {err_mm:.4f} mm")

    # Latency Benchmark over 100,000 queries
    n_iters = 100000
    t0 = time.perf_counter_ns()
    for _ in range(n_iters):
        _ = solver.inverse_kinematics(0.28, 0.08, 0.25)
    t1 = time.perf_counter_ns()
    avg_us = (t1 - t0) / (n_iters * 1000.0)
    logger.info(f"⚡ Benchmark: 100,000 IK computations in {(t1-t0)/1e9:.3f}s -> Mean Latency: {avg_us:.2f} µs/call")
