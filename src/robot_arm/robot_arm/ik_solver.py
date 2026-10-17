import math
import logging
from typing import Optional, List, Tuple

class IKSolver:
    """
    Analytical Inverse Kinematics solver for a 4-DOF Robot Arm.

    Joint layout:
        joint1 – base yaw  (rotation around Z-axis)
        joint2 – shoulder pitch
        joint3 – elbow pitch
        joint4 – wrist pitch (kept vertical so gripper faces down)

    Link geometry (defaults match Franka Hand URDF):
        base_height : distance from base_link origin to joint2 axis
        l1          : upper-arm length  (joint2 → joint3)
        l2          : forearm length    (joint3 → joint4)
        l_hand_4dof : hand + TCP length (joint4 → grasp point)
    """

    DOF = 4
    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

    def __init__(self, logger: logging.Logger = None, config: dict = None):
        self.logger = logger or logging.getLogger('IKSolver')

        # Sensible defaults for link lengths matching Franka Hand URDF
        self.config = {
            'base_height': 0.175,   # Distance from base_link origin to joint2 (0.025 + 0.15)
            'l1': 0.25,             # Upper arm length (joint2 to joint3)
            'l2': 0.20,             # Forearm length (joint3 to joint4)
            'l_hand_4dof': 0.1334,  # 4-DOF hand: link4(0.03) + Franka TCP(0.1034)
            'base_z_offset': 0.0,
        }
        if config:
            self.config.update(config)

    def forward_kinematics(self, t1: float, t2: float, t3: float, t4: float) -> Tuple[float, float, float, float]:
        """
        Compute forward kinematics for the 4-DOF arm.

        Args:
            t1: joint1 angle (base yaw, rad)
            t2: joint2 angle (shoulder pitch, rad)
            t3: joint3 angle (elbow pitch, rad)
            t4: joint4 angle (wrist pitch, rad)

        Returns:
            (x, y, z, yaw) of the grasp point in the base_link frame.
        """
        base_h = self.config['base_height'] + self.config['base_z_offset']
        l_hand = self.config['l_hand_4dof']

        r_wrist = self.config['l1'] * math.sin(t2) + self.config['l2'] * math.sin(t2 + t3)
        z_wrist = base_h + self.config['l1'] * math.cos(t2) + self.config['l2'] * math.cos(t2 + t3)

        total_pitch = t2 + t3 + t4
        r_grasp = r_wrist + l_hand * math.sin(total_pitch)
        z_grasp = z_wrist + l_hand * math.cos(total_pitch)

        x = r_grasp * math.cos(t1)
        y = r_grasp * math.sin(t1)
        return x, y, z_grasp, t1

    def inverse_kinematics(self, target_x: float, target_y: float, target_z: float, yaw: float = 0.0) -> Optional[List[float]]:
        """
        Compute analytical IK for the 4-DOF arm.

        The wrist pitch (t4) is computed so that the gripper points
        straight down (total_pitch = π), making the solution independent
        of the ``yaw`` parameter (kept for API compatibility).

        Args:
            target_x: desired grasp X in base_link frame (m)
            target_y: desired grasp Y in base_link frame (m)
            target_z: desired grasp Z in base_link frame (m)
            yaw: unused for 4-DOF (kept for interface compatibility)

        Returns:
            [t1, t2, t3, t4] joint angles in radians, or None if unreachable.
        """
        t1 = math.atan2(target_y, target_x)

        l_hand = self.config['l_hand_4dof']
        base_h = self.config['base_height'] + self.config['base_z_offset']
        l1 = self.config['l1']
        l2 = self.config['l2']

        # Back-project from grasp point to wrist (joint4) position
        r_wrist = math.sqrt(target_x**2 + target_y**2)
        z_wrist = target_z + l_hand

        z_adj = z_wrist - base_h
        d_sq = r_wrist**2 + z_adj**2
        d = math.sqrt(d_sq)

        if d > (l1 + l2) or d < abs(l1 - l2):
            self.logger.error(f"[IK] Target unreachable: D={d:.3f} out of bounds [{abs(l1-l2):.3f}, {l1+l2:.3f}]")
            return None

        cos_t3 = max(-1.0, min(1.0, (d_sq - l1**2 - l2**2) / (2.0 * l1 * l2)))
        t3 = math.acos(cos_t3)

        alpha = math.atan2(r_wrist, z_adj)
        beta = math.atan2(l2 * math.sin(t3), l1 + l2 * math.cos(t3))
        t2 = alpha - beta

        # Wrist pitch: keep gripper vertical (total pitch = π)
        t4 = math.pi - (t2 + t3)
        t4 = math.atan2(math.sin(t4), math.cos(t4))

        return [t1, t2, t3, t4]

    def validate_trajectory(self, waypoints: List[Tuple[float, float, float, float]]) -> Tuple[bool, List[Optional[List[float]]]]:
        """
        Validate that every (x, y, z, yaw) waypoint in the list is reachable.

        Args:
            waypoints: list of (x, y, z, yaw) tuples

        Returns:
            (all_valid, list_of_joint_configs) where list_of_joint_configs
            contains None for any unreachable waypoint.
        """
        all_valid = True
        results = []
        for wp in waypoints:
            joints = self.inverse_kinematics(*wp)
            results.append(joints)
            if joints is None:
                all_valid = False
        return all_valid, results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('IKSolverTest')
    logger.info("Running IK Solver Self-Test (4-DOF)...")
    solver = IKSolver(logger=logger)
    targets = [
        (0.25, 0.0,   0.25,  0.0),
        (0.30, 0.1,   0.26,  math.pi / 4),
        (0.20, -0.15, 0.35, -math.pi / 2),
    ]
    for x, y, z, yaw in targets:
        joints = solver.inverse_kinematics(x, y, z, yaw)
        if joints:
            fx, fy, fz, fyaw = solver.forward_kinematics(*joints)
            err = math.sqrt((x - fx)**2 + (y - fy)**2 + (z - fz)**2)
            logger.info(
                f"Target: ({x:.3f}, {y:.3f}, {z:.3f}, yaw={math.degrees(yaw):.1f}°) "
                f"-> FK error: {err * 1000:.3f} mm"
            )
        else:
            logger.error(f"Target unreachable: {x}, {y}, {z}")
