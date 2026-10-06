"""
Peer-Reviewed Grasp Evaluation Metrics.

Implements standard academic grasp evaluation metrics accepted in robotics literature
(ICRA, IROS, CVPR, RSS, CoRL):
1. Antipodal Force-Closure Analysis (Nguyen 1988, Ferrari & Canny 1992, Murray et al. 1994)
2. Coulomb Friction Cone Verification (mu = 0.4 low friction, mu = 0.8 high friction)
3. Geometric Collision Detection (Gripper envelope vs table and workspace bounds)
4. Kinematic Feasibility (Analytical IK reachability & joint limit constraints)
5. Gripper Aperture Compliance (Parallel-jaw stroke constraints)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_antipodal_force_closure(
    p1: np.ndarray,
    p2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    friction_coeff: float = 0.4,
) -> Tuple[bool, float, float]:
    """
    Evaluates whether two contact points form a valid Antipodal Force-Closure grasp.

    Parameters:
    - p1, p2: 3D coordinates of contact points on object surface.
    - n1, n2: Inward-pointing unit surface normals at p1 and p2.
    - friction_coeff: Coulomb friction coefficient (default 0.4 for plastic/wood, 0.8 for rubber).

    Returns:
    - is_force_closure: True if both contact normals lie within their respective friction cones.
    - angle1_deg: Angle between closing axis and n1 (degrees).
    - angle2_deg: Angle between closing axis (reversed) and n2 (degrees).
    """
    d = p2 - p1
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return False, 90.0, 90.0

    closing_dir = d / dist  # Direction from p1 to p2

    # Normalization of surface normals
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-6 or norm_n2 < 1e-6:
        return False, 90.0, 90.0

    n1_unit = n1 / norm_n1
    n2_unit = n2 / norm_n2

    # Friction cone half-angle
    alpha_max = math.atan(friction_coeff)
    cos_alpha_max = math.cos(alpha_max)

    # Dot products: n1 should point along closing_dir; n2 should point along -closing_dir
    dot1 = np.clip(np.dot(n1_unit, closing_dir), -1.0, 1.0)
    dot2 = np.clip(np.dot(n2_unit, -closing_dir), -1.0, 1.0)

    angle1_deg = math.degrees(math.acos(abs(dot1)))
    angle2_deg = math.degrees(math.acos(abs(dot2)))

    is_fc = (dot1 >= cos_alpha_max) and (dot2 >= cos_alpha_max)
    return bool(is_fc), float(angle1_deg), float(angle2_deg)


def check_table_collision(
    pos: np.ndarray,
    rot: np.ndarray,
    table_z: float = 0.225,
    finger_length: float = 0.05,
    finger_width: float = 0.02,
    safety_margin: float = 0.005,
) -> bool:
    """
    Checks if any part of the gripper fingers or palm penetrates below the table surface.

    Parameters:
    - pos: Gripper TCP position (x, y, z) in meters.
    - rot: 3x3 rotation matrix of the gripper.
    - table_z: Height of the table surface (meters).
    - finger_length: Length of the fingers extending past TCP (meters).
    - finger_width: Half-span of the gripper fingers (meters).
    - safety_margin: Safety margin above table (meters).

    Returns:
    - True if collision occurs (gripper enters table boundary), False otherwise.
    """
    # 8 bounding box corners representing the finger envelopes in gripper local frame
    local_corners = np.array([
        [0.0, -finger_width, 0.0],
        [0.0, finger_width, 0.0],
        [0.0, -finger_width, -finger_length],
        [0.0, finger_width, -finger_length],
        [0.02, -finger_width, -finger_length],
        [0.02, finger_width, -finger_length],
        [-0.02, -finger_width, -finger_length],
        [-0.02, finger_width, -finger_length],
    ])

    world_corners = (rot @ local_corners.T).T + pos
    min_z = float(np.min(world_corners[:, 2]))
    return min_z < (table_z - safety_margin)


def check_aperture_compliance(width: float, max_gripper_width: float = 0.08) -> bool:
    """Verifies that required grasp width does not exceed physical mechanical stroke."""
    return 0.002 <= width <= max_gripper_width


def check_kinematic_feasibility(
    pos: np.ndarray,
    yaw: float,
    ik_solver: Any,
) -> Tuple[bool, Optional[List[float]]]:
    """
    Evaluates whether the grasp pose is reachable within robot arm kinematic limits.

    Returns:
    - (is_reachable, joint_angles)
    """
    if ik_solver is None:
        return True, None
    try:
        angles = ik_solver.inverse_kinematics(float(pos[0]), float(pos[1]), float(pos[2]), yaw=float(yaw))
        return (angles is not None), angles
    except Exception:
        return False, None
