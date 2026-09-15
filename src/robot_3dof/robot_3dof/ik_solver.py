"""
Analytical Inverse Kinematics solver for 3 DoF RRR robot arm.

Robot configuration:
  - Joint 1: Revolute around Z (base rotation)
  - Joint 2: Revolute around Y (shoulder)
  - Joint 3: Revolute around Y (elbow)

Dimensions:
  - Base height (base_link + link1): 0.20 m
  - Link2 (upper arm): 0.25 m
  - Link3 (forearm): 0.20 m
"""

import math
from typing import Optional, Tuple

# Robot dimensions
BASE_HEIGHT = 0.05 + 0.15  # base_link height/2 + link1 height = 0.20
L1 = 0.25  # link2 length (upper arm)
L2 = 0.20  # link3 length (forearm)

# Joint limits
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
    x: float, y: float, z: float, elbow_up: bool = True
) -> Optional[Tuple[float, float, float]]:
    """
    Compute inverse kinematics for a target position in base_link frame.

    The end-effector position includes the gripper offset, so account for it
    when calling this function (subtract gripper length from z if needed).

    Args:
        x: Target X position in base_link frame (forward)
        y: Target Y position in base_link frame (left)
        z: Target Z position in base_link frame (up)
        elbow_up: If True, prefer elbow-up solution (arm reaches down).

    Returns:
        Tuple of (theta1, theta2, theta3) in radians, or None if unreachable.
    """
    # Joint 1: base rotation
    theta1 = math.atan2(y, x)

    # Project onto the arm plane (vertical plane containing the target)
    r = math.sqrt(x * x + y * y)  # Horizontal distance
    z_adj = z - BASE_HEIGHT  # Height relative to shoulder joint

    # Distance from shoulder joint to target in the arm plane
    d_sq = r * r + z_adj * z_adj
    d = math.sqrt(d_sq)

    # Check reachability
    if d > (L1 + L2) or d < abs(L1 - L2):
        return None

    # Law of cosines for elbow angle (joint3)
    cos_theta3 = (d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)

    # Clamp for numerical safety
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))

    if elbow_up:
        theta3 = -math.acos(cos_theta3)  # Elbow up (negative angle)
    else:
        theta3 = math.acos(cos_theta3)  # Elbow down (positive angle)

    # Shoulder angle (joint2)
    # alpha: angle from vertical (Z) to the target direction
    alpha = math.atan2(r, z_adj)

    # beta: angle offset due to link3
    beta = math.atan2(L2 * math.sin(abs(theta3)), L1 + L2 * math.cos(theta3))

    if elbow_up:
        theta2 = alpha + beta
    else:
        theta2 = alpha - beta

    # Check joint limits
    if not check_limits(theta1, theta2, theta3):
        # Try the other elbow configuration
        theta3_alt = -theta3
        if elbow_up:
            theta2_alt = alpha - beta
        else:
            theta2_alt = alpha + beta

        if check_limits(theta1, theta2_alt, theta3_alt):
            return (theta1, theta2_alt, theta3_alt)
        return None

    return (theta1, theta2, theta3)


def forward_kinematics(
    theta1: float, theta2: float, theta3: float
) -> Tuple[float, float, float]:
    """
    Compute forward kinematics: joint angles → end-effector position.

    Args:
        theta1, theta2, theta3: Joint angles in radians.

    Returns:
        Tuple of (x, y, z) position in base_link frame.
    """
    # In the arm plane
    r = L1 * math.sin(theta2) + L2 * math.sin(theta2 + theta3)
    z = BASE_HEIGHT + L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3)

    # Rotate by theta1 to get 3D position
    x = r * math.cos(theta1)
    y = r * math.sin(theta1)

    return (x, y, z)


if __name__ == "__main__":
    # Quick self-test
    print("=== IK Solver Self-Test ===")

    # Test 1: Forward → Inverse roundtrip
    test_angles = [
        (0.0, 0.5, -0.3),
        (0.3, 0.8, -0.5),
        (-0.5, 0.3, -0.2),
        (1.0, 0.4, -0.4),
    ]

    for t1, t2, t3 in test_angles:
        x, y, z = forward_kinematics(t1, t2, t3)
        result = solve_ik(x, y, z)
        if result is not None:
            x2, y2, z2 = forward_kinematics(*result)
            err = math.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
            print(f"  angles=({t1:.2f}, {t2:.2f}, {t3:.2f}) → "
                  f"pos=({x:.3f}, {y:.3f}, {z:.3f}) → "
                  f"IK=({result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}) "
                  f"err={err:.6f}")
        else:
            print(f"  angles=({t1:.2f}, {t2:.2f}, {t3:.2f}) → "
                  f"pos=({x:.3f}, {y:.3f}, {z:.3f}) → IK=UNREACHABLE")

    # Test 2: Table-height position (typical grasp target)
    print("\nTest grasp targets on table (z=0.28):")
    targets = [
        (0.25, 0.05, 0.28),
        (0.35, -0.05, 0.28),
        (0.30, 0.00, 0.30),
    ]
    for x, y, z in targets:
        result = solve_ik(x, y, z)
        if result:
            print(f"  target=({x}, {y}, {z}) → "
                  f"joints=({math.degrees(result[0]):.1f}°, "
                  f"{math.degrees(result[1]):.1f}°, "
                  f"{math.degrees(result[2]):.1f}°)")
        else:
            print(f"  target=({x}, {y}, {z}) → UNREACHABLE")
