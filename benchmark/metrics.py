"""
Peer-Reviewed Grasp Evaluation Metrics.

Implements standard academic grasp evaluation metrics accepted in robotics literature
(ICRA, IROS, CVPR, RSS, CoRL):
1. Antipodal Force-Closure Analysis (Nguyen 1988, Ferrari & Canny 1992, Murray et al. 1994)
2. Coulomb Friction Cone Verification (mu = 0.4 low friction, mu = 0.8 high friction)
3. Geometric Collision Detection (Gripper envelope vs table and workspace bounds)
4. Kinematic Feasibility (Analytical IK reachability & joint limit constraints)
5. Gripper Aperture Compliance (Parallel-jaw stroke constraints)
6. GraspNet-Style Average Precision (AP_0.4, AP_0.8, AP_seen, AP_novel)
   Fang et al. "GraspNet-1Billion: A Large-Scale Benchmark." CVPR 2020.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np


# Public re-export: callers can do `from benchmark.metrics import compute_ap_score`
# without directly importing eval_ap.
def compute_ap_score(
    candidates: "Sequence",
    mu_values: Tuple[float, float] = (0.4, 0.8),
) -> "APResult":  # type: ignore[name-defined]
    """
    Compute GraspNet-style Average Precision for a sequence of GraspCandidate objects.

    Delegates to ``benchmark.eval_ap.compute_ap``.

    Parameters
    ----------
    candidates : sequence of benchmark.eval_ap.GraspCandidate
    mu_values : (float, float)
        Friction thresholds. Default: (0.4, 0.8).

    Returns
    -------
    benchmark.eval_ap.APResult
        Full AP result with ``ap``, ``ap_04``, ``ap_08``, ``ap_seen``, ``ap_novel``.
    """
    from benchmark.eval_ap import compute_ap
    return compute_ap(candidates, mu_values=mu_values)




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
    rotation_matrix: Optional[np.ndarray] = None,
) -> Tuple[bool, Optional[List[float]]]:
    """
    Evaluates whether the grasp pose is reachable within robot arm kinematic limits.

    Returns:
    - (is_reachable, joint_angles)
    """
    if ik_solver is None:
        return True, None
    try:
        if getattr(ik_solver, "dof", 5) == 6 and rotation_matrix is not None:
            angles = ik_solver.inverse_kinematics(
                float(pos[0]), float(pos[1]), float(pos[2]), yaw=float(yaw), rotation_matrix=rotation_matrix
            )
        else:
            angles = ik_solver.inverse_kinematics(float(pos[0]), float(pos[1]), float(pos[2]), yaw=float(yaw))
        return (angles is not None), angles
    except Exception:
        return False, None


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score confidence interval for binomial proportions.

    Args:
        k: number of successes
        n: total number of trials
        confidence: confidence level (default: 0.95, z = 1.95996)

    Returns:
        (p_hat, ci_lower, ci_upper)
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = float(k) / float(n)
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 2.57583  # 95% or 99%
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z**2) / (4.0 * n**2))
    return p, max(0.0, center - margin), min(1.0, center + margin)


def compute_mcnemar_statistic(b: int, c: int) -> Tuple[float, float]:
    """
    Computes McNemar's paired test statistic with Edwards' continuity correction.

    Args:
        b: Method 1 succeeds, Method 2 fails (discordant pairs)
        c: Method 1 fails, Method 2 succeeds (discordant pairs)

    Returns:
        (chi2_statistic, p_approx)
    """
    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0
    term = max(0.0, abs(b - c) - 1.0)
    chi2 = (term**2) / float(total_discordant)
    # 1-DOF chi-squared survival function approximation
    # p = erfc(sqrt(chi2 / 2))
    p_val = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
    return float(chi2), float(p_val)


def evaluate_4dof_orientability(
    x: float, y: float, target_yaw: float, tolerance_deg: float = 25.0
) -> Tuple[bool, float, float]:
    """
    Evaluates kinematic orientability for a 4-DOF manipulator.
    Since joint1 is the only Z-rotation, end-effector closing angle is locked to:
        psi_arm = atan2(y, x) + pi/2 (mod pi)

    Returns:
        (is_orientable, divergence_deg, alignment_score [0, 1])
    """
    th1 = math.atan2(y, x)
    arm_yaw = th1 + math.pi / 2.0
    diff = abs(target_yaw - arm_yaw) % math.pi
    divergence = min(diff, math.pi - diff)
    div_deg = math.degrees(divergence)
    alignment_score = math.cos(divergence) ** 2
    is_orientable = div_deg <= tolerance_deg
    return is_orientable, div_deg, alignment_score

