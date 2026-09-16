"""
Analytical Inverse Kinematics solver for 3 DoF RRR robot arm.

Robot configuration:
  - Joint 1: Revolute around Z (base rotation)
  - Joint 2: Revolute around Y (shoulder)
  - Joint 3: Revolute around Y (elbow)

Dimensions (in base_link frame):
  - Base height to shoulder (joint2): 0.175 m (base_height/2 = 0.025 + link1_height = 0.15)
  - Link2 (upper arm): 0.25 m
  - Link3 + Gripper (to grasp center): 0.23 m
      (link3_length = 0.20 + gripper_base_z = 0.01 + finger_z/2 = 0.02)
"""

import math
from typing import Optional, Tuple

# Robot dimensions in base_link frame
BASE_HEIGHT = 0.025 + 0.15  # 0.175 m (Z of joint2 in base_link frame)
L1 = 0.25                   # 0.25 m (link2 length)
L2 = 0.20 + 0.01 + 0.02     # 0.23 m (distance from joint3 to center of finger pads)

# Joint limits (radians)
JOINT1_MIN, JOINT1_MAX = -math.pi, math.pi
JOINT2_MIN, JOINT2_MAX = -2.5, 2.5
JOINT3_MIN, JOINT3_MAX = -2.5, 2.5


def check_limits(theta1: float, theta2: float, theta3: float) -> bool:
    """Check if joint angles are within limits."""
    return (
        JOINT1_MIN <= theta1 <= JOINT1_MAX
        and JOINT2_MIN <= theta2 <= JOINT2_MAX
        and JOINT3_MIN <= theta3 <= JOINT3_MAX
    )


def solve_ik(
    x: float, y: float, z: float, reach_down: bool = True
) -> Optional[Tuple[float, float, float]]:
    """
    Compute inverse kinematics for a target grasp position in base_link frame.

    The solver directly targets the grasp center between the gripper finger pads.

    Args:
        x: Target X position in base_link frame (forward)
        y: Target Y position in base_link frame (left)
        z: Target Z position in base_link frame (up)
        reach_down: If True (default), prefer positive elbow angle (theta3 > 0)
                   where the upper arm stays high and the forearm reaches
                   downward towards the object like a crane/excavator.

    Returns:
        Tuple of (theta1, theta2, theta3) in radians, or None if unreachable.
    """
    # Joint 1: base rotation around Z
    theta1 = math.atan2(y, x)

    # Project onto the arm's vertical motion plane
    r = math.sqrt(x * x + y * y)  # Horizontal distance from base Z axis
    z_adj = z - BASE_HEIGHT        # Height relative to shoulder joint (joint2)

    # Distance from shoulder joint to target
    d_sq = r * r + z_adj * z_adj
    d = math.sqrt(d_sq)

    # Check reachability
    if d > (L1 + L2) or d < abs(L1 - L2):
        return None

    # Law of cosines for elbow angle (joint3)
    cos_theta3 = (d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))

    # Primary configuration
    if reach_down:
        # Crane posture: theta3 > 0 bends forearm downward towards the table
        theta3_primary = math.acos(cos_theta3)
        theta3_alt = -theta3_primary
    else:
        theta3_primary = -math.acos(cos_theta3)
        theta3_alt = -theta3_primary

    # Shoulder angle calculation
    alpha = math.atan2(r, z_adj)

    # Helper to compute theta2 for a given theta3
    def get_theta2(t3: float) -> float:
        beta = math.atan2(L2 * math.sin(t3), L1 + L2 * math.cos(t3))
        return alpha - beta

    # Try primary solution
    theta2_primary = get_theta2(theta3_primary)
    if check_limits(theta1, theta2_primary, theta3_primary):
        return (theta1, theta2_primary, theta3_primary)

    # Fallback to alternative solution
    theta2_alt = get_theta2(theta3_alt)
    if check_limits(theta1, theta2_alt, theta3_alt):
        return (theta1, theta2_alt, theta3_alt)

    return None


def forward_kinematics(
    theta1: float, theta2: float, theta3: float
) -> Tuple[float, float, float]:
    """
    Compute forward kinematics: joint angles → grasp center position in base_link.

    Args:
        theta1, theta2, theta3: Joint angles in radians.

    Returns:
        Tuple of (x, y, z) position in base_link frame.
    """
    # In the arm's vertical plane
    r = L1 * math.sin(theta2) + L2 * math.sin(theta2 + theta3)
    z = BASE_HEIGHT + L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3)

    # Rotate by theta1 into 3D base_link frame
    x = r * math.cos(theta1)
    y = r * math.sin(theta1)

    return (x, y, z)


if __name__ == "__main__":
    print("=== IK Solver Self-Test ===")

    # Test grasp targets on table (table surface is at z=0.225 in base_link)
    targets = [
        ("Red box center", 0.25, 0.05, 0.240),
        ("Green cylinder center", 0.35, -0.05, 0.250),
        ("Blue sphere center", 0.30, 0.08, 0.245),
        ("Pre-grasp above red box", 0.25, 0.05, 0.320),
        ("Place position", 0.20, -0.15, 0.320),
    ]

    for name, x, y, z in targets:
        result = solve_ik(x, y, z)
        if result:
            t1, t2, t3 = result
            rx, ry, rz = forward_kinematics(t1, t2, t3)
            err = math.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
            pitch = math.degrees(t2 + t3)
            print(f"  {name:25s}: pos=({x:.3f}, {y:.3f}, {z:.3f}) → "
                  f"joints=({math.degrees(t1):6.1f}°, {math.degrees(t2):6.1f}°, {math.degrees(t3):6.1f}°) "
                  f"pitch={pitch:5.1f}° err={err*1000:.3f}mm")
        else:
            print(f"  {name:25s}: pos=({x:.3f}, {y:.3f}, {z:.3f}) → UNREACHABLE")
