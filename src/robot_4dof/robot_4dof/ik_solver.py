"""
Analytical Inverse Kinematics solver for 4 DoF RRRR robot arm.

Robot configuration:
  - Joint 1: Revolute around Z (base rotation)
  - Joint 2: Revolute around Y (shoulder)
  - Joint 3: Revolute around Y (elbow)
  - Joint 4: Revolute around Z (wrist roll — 4th DOF)

Dimensions (in base_link frame):
  - Base height to shoulder (joint2): 0.175 m (base_height/2 = 0.025 + link1_height = 0.15)
  - Link2 (upper arm): 0.25 m
  - Link3 (forearm): 0.20 m
  - Link4 (wrist) + Gripper (to grasp center): 0.06 m
      (link4_height = 0.03 + gripper_base_z = 0.01 + finger_z/2 = 0.02)

Note: Joint4 (wrist roll) rotates the gripper around the arm's local Z-axis at the
end of link3. It does NOT change the XYZ position of the grasp center, only the
gripper's yaw orientation. Therefore:
  - theta1, theta2, theta3 are solved identically to the 3DOF case for position.
  - theta4 is set directly from the desired gripper yaw angle.
"""

import math
from typing import Optional, Tuple

# Robot dimensions in base_link frame
BASE_HEIGHT = 0.025 + 0.15  # 0.175 m (Z of joint2 in base_link frame)
L1 = 0.25                   # 0.25 m (link2 length)
L2 = 0.20 + 0.03 + 0.01 + 0.02  # 0.26 m (link3 + link4_height + gripper_base_z + finger_z/2)

# Joint limits (radians)
JOINT1_MIN, JOINT1_MAX = -math.pi, math.pi
JOINT2_MIN, JOINT2_MAX = -2.5, 2.5
JOINT3_MIN, JOINT3_MAX = -2.5, 2.5
JOINT4_MIN, JOINT4_MAX = -math.pi, math.pi


def check_limits(theta1: float, theta2: float, theta3: float, theta4: float) -> bool:
    """Check if joint angles are within limits."""
    return (
        JOINT1_MIN <= theta1 <= JOINT1_MAX
        and JOINT2_MIN <= theta2 <= JOINT2_MAX
        and JOINT3_MIN <= theta3 <= JOINT3_MAX
        and JOINT4_MIN <= theta4 <= JOINT4_MAX
    )


def solve_ik(
    x: float, y: float, z: float, yaw: float = 0.0, reach_down: bool = True
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute inverse kinematics for a target grasp position in base_link frame.

    The solver directly targets the grasp center between the gripper finger pads.
    Joint4 (wrist roll) is set to the desired yaw angle for optimal grasp orientation.

    Args:
        x: Target X position in base_link frame (forward)
        y: Target Y position in base_link frame (left)
        z: Target Z position in base_link frame (up)
        yaw: Desired gripper yaw angle in radians (wrist roll, default=0.0)
        reach_down: If True (default), prefer positive elbow angle (theta3 > 0)
                   where the upper arm stays high and the forearm reaches
                   downward towards the object like a crane/excavator.

    Returns:
        Tuple of (theta1, theta2, theta3, theta4) in radians, or None if unreachable.
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

    # Joint4 (wrist roll): set directly from desired yaw
    # Relative to base rotation: theta4 = yaw - theta1
    # This makes the gripper's absolute yaw in the world equal to the desired yaw
    theta4 = yaw - theta1
    # Normalize theta4 to [-pi, pi]
    theta4 = math.atan2(math.sin(theta4), math.cos(theta4))

    # Try primary solution
    theta2_primary = get_theta2(theta3_primary)
    if check_limits(theta1, theta2_primary, theta3_primary, theta4):
        return (theta1, theta2_primary, theta3_primary, theta4)

    # Fallback to alternative solution
    theta2_alt = get_theta2(theta3_alt)
    if check_limits(theta1, theta2_alt, theta3_alt, theta4):
        return (theta1, theta2_alt, theta3_alt, theta4)

    return None


def forward_kinematics(
    theta1: float, theta2: float, theta3: float, theta4: float
) -> Tuple[float, float, float, float]:
    """
    Compute forward kinematics: joint angles → grasp center position + yaw in base_link.

    Args:
        theta1, theta2, theta3, theta4: Joint angles in radians.

    Returns:
        Tuple of (x, y, z, yaw) position and orientation in base_link frame.
    """
    # In the arm's vertical plane
    r = L1 * math.sin(theta2) + L2 * math.sin(theta2 + theta3)
    z = BASE_HEIGHT + L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3)

    # Rotate by theta1 into 3D base_link frame
    x = r * math.cos(theta1)
    y = r * math.sin(theta1)

    # Gripper absolute yaw = theta1 (base rotation) + theta4 (wrist roll)
    yaw = theta1 + theta4

    return (x, y, z, yaw)


if __name__ == "__main__":
    print("=== 4 DoF IK Solver Self-Test ===")

    # Test grasp targets on table (table surface is at z=0.225 in base_link)
    targets = [
        ("Red box center (yaw=0°)", 0.25, 0.05, 0.240, 0.0),
        ("Green cyl center (yaw=45°)", 0.35, -0.05, 0.250, math.radians(45)),
        ("Blue sphere (yaw=90°)", 0.30, 0.08, 0.245, math.radians(90)),
        ("Pre-grasp above red box", 0.25, 0.05, 0.320, 0.0),
        ("Place position", 0.20, -0.15, 0.320, 0.0),
    ]

    for name, x, y, z, yaw in targets:
        result = solve_ik(x, y, z, yaw=yaw)
        if result:
            t1, t2, t3, t4 = result
            rx, ry, rz, ryaw = forward_kinematics(t1, t2, t3, t4)
            err = math.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
            pitch = math.degrees(t2 + t3)
            print(f"  {name:30s}: pos=({x:.3f}, {y:.3f}, {z:.3f}) yaw={math.degrees(yaw):5.1f}° →")
            print(f"    joints=({math.degrees(t1):6.1f}°, {math.degrees(t2):6.1f}°, {math.degrees(t3):6.1f}°, {math.degrees(t4):6.1f}°)")
            print(f"    pitch={pitch:5.1f}° err={err*1000:.3f}mm yaw_out={math.degrees(ryaw):5.1f}°")
        else:
            print(f"  {name:30s}: pos=({x:.3f}, {y:.3f}, {z:.3f}) → UNREACHABLE")
