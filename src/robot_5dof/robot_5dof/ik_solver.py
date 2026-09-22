import math
import numpy as np

# Robot Geometric Parameters
L1 = 0.25      # length of link2 (upper arm)
L2 = 0.20      # length of link3 (forearm)
L_HAND = 0.08  # link4(0.03) + link5(0.02) + gripper_base_z(0.01) + finger_center(0.02)
BASE_HEIGHT = 0.15 # link1 height

def forward_kinematics(t1, t2, t3, t4, t5):
    """
    Computes FK for the 5-DOF arm (Z-Y-Y-Y-Z).
    Returns (x, y, z, yaw) of the end-effector grasp center.
    """
    # Joint positions in world frame (Z up)
    # joint2 is at z = BASE_HEIGHT
    
    # Position of joint3
    x3 = L1 * math.sin(t2)
    z3 = BASE_HEIGHT + L1 * math.cos(t2)
    
    # Position of joint4
    x4 = x3 + L2 * math.sin(t2 + t3)
    z4 = z3 + L2 * math.cos(t2 + t3)
    
    # Position of grasp center
    # link4, link5, gripper total length is L_HAND
    # angle of the wrist is t2 + t3 + t4
    total_pitch = t2 + t3 + t4
    
    x_end_local = x4 + L_HAND * math.sin(total_pitch)
    z_end = z4 + L_HAND * math.cos(total_pitch)
    
    # Rotate by base yaw (t1) to get world X, Y
    x_end = x_end_local * math.cos(t1)
    y_end = x_end_local * math.sin(t1)
    
    # The gripper's total yaw in world frame is base_yaw + wrist_roll
    # However, this is only perfectly true if the gripper is pointing straight down
    yaw = t1 + t5
    # Normalize yaw
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    
    return x_end, y_end, z_end, yaw

def inverse_kinematics(target_x, target_y, target_z, yaw):
    """
    Computes IK for the 5-DOF arm.
    Constraint: The gripper MUST point straight down (pitch = pi or 180 degrees).
    Returns (t1, t2, t3, t4, t5) or None if unreachable.
    """
    # 1. Base rotation (t1)
    t1 = math.atan2(target_y, target_x)
    
    # 2. Compute wrist roll (t5)
    # total_yaw = t1 + t5  => t5 = yaw - t1
    t5 = yaw - t1
    # Normalize to [-pi, pi]
    t5 = math.atan2(math.sin(t5), math.cos(t5))
    
    # 3. Reduce to 2D planar problem
    r_target = math.sqrt(target_x**2 + target_y**2)
    
    # We want the gripper to point straight down.
    # Therefore, the wrist joint (joint4) must be exactly L_HAND straight up from the target.
    # The pitch angle of the hand is pi (180 degrees) from the local vertical.
    r_wrist = r_target # since straight down, r_wrist = r_target
    z_wrist = target_z + L_HAND
    
    # Calculate vector from joint2 to joint4
    dz = z_wrist - BASE_HEIGHT
    dr = r_wrist
    
    # Distance from joint2 to joint4
    D = math.sqrt(dr**2 + dz**2)
    
    # Check reachability
    if D > (L1 + L2) or D < abs(L1 - L2):
        print(f"[IK] Target unreachable: D={D:.3f} > Max={L1+L2:.3f}")
        return None
        
    # Cosine law for elbow angle (t3)
    cos_t3 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t3 = max(-1.0, min(1.0, cos_t3))
    
    # We prefer elbow UP configuration -> positive t3
    t3 = math.acos(cos_t3)
    
    # Calculate shoulder angle (t2)
    alpha = math.atan2(dz, dr) # angle from horizontal
    beta = math.atan2(L2 * math.sin(t3), L1 + L2 * math.cos(t3))
    
    # In our URDF, t2=0 means pointing straight UP (Z axis).
    # So the angle from Z axis is pi/2 - (alpha + beta)
    t2 = (math.pi / 2.0) - (alpha + beta)
    
    # 4. Enforce straight-down constraint for wrist pitch (t4)
    # The total pitch angle must be pi (180 degrees) to point straight down
    # total_pitch = t2 + t3 + t4 = pi
    t4 = math.pi - (t2 + t3)
    # Prevent wrist from folding backwards into forearm (>90 deg)
    if t4 > math.pi/2:
        print(f'[IK WARNING] Wrist pitch {math.degrees(t4):.1f}° clamped to 90° to prevent self-collision!')
        t4 = math.pi/2
    elif t4 < -math.pi/2:
        t4 = -math.pi/2
    
    return [t1, t2, t3, t4, t5]

if __name__ == '__main__':
    # Simple self-test
    print("Running IK Solver Self-Test (5-DOF)...")
    targets = [
        (0.25, 0.0, 0.25, 0.0),
        (0.30, 0.1, 0.26, math.pi/4),
        (0.20, -0.15, 0.35, -math.pi/2),
    ]
    for x, y, z, yaw in targets:
        joints = inverse_kinematics(x, y, z, yaw)
        if joints:
            fx, fy, fz, fyaw = forward_kinematics(*joints)
            err = math.sqrt((x-fx)**2 + (y-fy)**2 + (z-fz)**2)
            print(f"Target: ({x:.3f}, {y:.3f}, {z:.3f}, yaw={math.degrees(yaw):.1f}°) -> Error: {err*1000:.3f} mm")
            print(f"Joints: {np.degrees(joints).round(1)}°")
        else:
            print(f"Target unreachable: {x}, {y}, {z}")
