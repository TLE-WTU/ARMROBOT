"""
Analytical Inverse Kinematics solver for 4 DoF RRRR robot arm.

Robot configuration (Z-Y-Y-Y):
  - Joint 1: Revolute around Z (base rotation)
  - Joint 2: Revolute around Y (shoulder)
  - Joint 3: Revolute around Y (elbow)
  - Joint 4: Revolute around Y (wrist pitch — NEW 4th DOF)

This configuration ensures the gripper can ALWAYS point straight down (top-down grasp),
which keeps the fingers perfectly horizontal to avoid hitting the table.
As a 4-DOF arm, it cannot independently control Yaw. The gripper yaw is locked to the 
base rotation (theta1).

Dimensions (in base_link frame):
  - Base height to shoulder (joint2): 0.175 m
  - Link2 (upper arm): 0.25 m
  - Link3 (forearm): 0.20 m
  - Gripper length (from wrist joint4 to grasp center): 0.06 m
      (link4_height=0.03 + gripper_base_z=0.01 + finger_z/2=0.02)
"""

import math
from typing import Optional, Tuple

# Robot dimensions in base_link frame
BASE_HEIGHT = 0.175  # 0.175 m (Z of joint2 in base_link frame)
L1 = 0.25            # 0.25 m (link2 length)
L2 = 0.20            # 0.20 m (link3 length)
L_HAND = 0.06        # 0.06 m (distance from wrist joint to grasp center)

# Joint limits (radians)
JOINT1_MIN, JOINT1_MAX = -math.pi, math.pi
JOINT2_MIN, JOINT2_MAX = -2.5, 2.5
JOINT3_MIN, JOINT3_MAX = -2.5, 2.5
JOINT4_MIN, JOINT4_MAX = -2.5, 2.5


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

    The solver directly targets the grasp center. Since it's a Z-Y-Y-Y arm,
    we enforce the gripper to point perfectly straight down (top-down).
    The 'yaw' argument is ignored because a 4-DOF planar arm cannot independently 
    control yaw while pointing straight down.

    Args:
        x: Target X position in base_link frame (forward)
        y: Target Y position in base_link frame (left)
        z: Target Z position in base_link frame (up)
        yaw: Ignored in Z-Y-Y-Y configuration.
        reach_down: If True (default), prefer positive elbow angle (theta3 > 0).

    Returns:
        Tuple of (theta1, theta2, theta3, theta4) in radians, or None if unreachable.
    """
    # Joint 1: base rotation around Z
    theta1 = math.atan2(y, x)

    # Since the gripper must point straight down, the wrist joint4 must be 
    # exactly L_HAND above the target grasp position.
    r_target = math.sqrt(x * x + y * y)
    
    # Wrist position in the arm's 2D vertical plane
    r_wrist = r_target
    z_wrist = z + L_HAND
    
    # Position of wrist relative to shoulder (joint2)
    z_adj = z_wrist - BASE_HEIGHT

    # Distance from shoulder joint to wrist joint
    d_sq = r_wrist * r_wrist + z_adj * z_adj
    d = math.sqrt(d_sq)

    # Check reachability for the wrist
    if d > (L1 + L2) or d < abs(L1 - L2):
        return None

    # Law of cosines for elbow angle (joint3)
    cos_theta3 = (d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))

    # Primary configuration
    if reach_down:
        # Crane posture: theta3 > 0 bends forearm downward
        theta3_primary = math.acos(cos_theta3)
        theta3_alt = -theta3_primary
    else:
        theta3_primary = -math.acos(cos_theta3)
        theta3_alt = -theta3_primary

    # Shoulder angle calculation
    alpha = math.atan2(r_wrist, z_adj)

    def get_theta2(t3: float) -> float:
        beta = math.atan2(L2 * math.sin(t3), L1 + L2 * math.cos(t3))
        return alpha - beta

    def get_theta4(t2: float, t3: float) -> float:
        # To point straight down, the sum of all pitch angles must be exactly pi (180 deg)
        # Because theta=0 points straight UP (+Z).
        t4 = math.pi - (t2 + t3)
        # Normalize to [-pi, pi]
        t4 = math.atan2(math.sin(t4), math.cos(t4))
        return t4

    # Try primary solution
    theta2_primary = get_theta2(theta3_primary)
    theta4_primary = get_theta4(theta2_primary, theta3_primary)
    if check_limits(theta1, theta2_primary, theta3_primary, theta4_primary):
        return (theta1, theta2_primary, theta3_primary, theta4_primary)

    # Fallback to alternative solution
    theta2_alt = get_theta2(theta3_alt)
    theta4_alt = get_theta4(theta2_alt, theta3_alt)
    if check_limits(theta1, theta2_alt, theta3_alt, theta4_alt):
        return (theta1, theta2_alt, theta3_alt, theta4_alt)

    return None


def forward_kinematics(
    theta1: float, theta2: float, theta3: float, theta4: float
) -> Tuple[float, float, float, float]:
    """
    Compute forward kinematics: joint angles → grasp center position + yaw in base_link.
    """
    # Wrist position
    r_wrist = L1 * math.sin(theta2) + L2 * math.sin(theta2 + theta3)
    z_wrist = BASE_HEIGHT + L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3)
    
    # Grasp center position (add L_HAND along the final link's direction)
    total_pitch = theta2 + theta3 + theta4
    r_grasp = r_wrist + L_HAND * math.sin(total_pitch)
    z_grasp = z_wrist + L_HAND * math.cos(total_pitch)

    # Rotate by theta1 into 3D base_link frame
    x = r_grasp * math.cos(theta1)
    y = r_grasp * math.sin(theta1)

    # Gripper yaw is just theta1
    yaw = theta1

    return (x, y, z_grasp, yaw)


if __name__ == "__main__":
    print("=== 4 DoF IK Solver Self-Test ===")

    targets = [
        ("Red box center", 0.25, 0.05, 0.240, 0.0),
        ("Green cyl center", 0.35, -0.05, 0.250, 0.0),
        ("Blue sphere", 0.30, 0.08, 0.245, 0.0),
        ("Pre-grasp above red box", 0.25, 0.05, 0.320, 0.0),
        ("Place position", 0.20, -0.15, 0.320, 0.0),
    ]

    for name, x, y, z, yaw in targets:
        result = solve_ik(x, y, z, yaw=yaw)
        if result:
            t1, t2, t3, t4 = result
            rx, ry, rz, ryaw = forward_kinematics(t1, t2, t3, t4)
            err = math.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
            pitch = math.degrees(t2 + t3 + t4)
            print(f"  {name:30s}: pos=({x:.3f}, {y:.3f}, {z:.3f}) →")
            print(f"    joints=({math.degrees(t1):6.1f}°, {math.degrees(t2):6.1f}°, {math.degrees(t3):6.1f}°, {math.degrees(t4):6.1f}°)")
            print(f"    gripper_pitch={pitch:5.1f}° err={err*1000:.3f}mm yaw_out={math.degrees(ryaw):5.1f}°")
        else:
            print(f"  {name:30s}: pos=({x:.3f}, {y:.3f}, {z:.3f}) → UNREACHABLE")
