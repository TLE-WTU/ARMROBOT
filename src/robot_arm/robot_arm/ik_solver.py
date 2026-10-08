import math
import logging
from typing import Optional, List, Tuple
import numpy as np


class IKSolver:
    """
    Unified Analytical Inverse Kinematics solver for 3/4/5/6 DoF Robot Arm.
    Supports:
    - 3-DOF: [Base Yaw, Shoulder Pitch, Elbow Pitch]
    - 4-DOF: [Base Yaw, Shoulder Pitch, Elbow Pitch, Wrist Pitch]
    - 5-DOF: [Base Yaw, Shoulder Pitch, Elbow Pitch, Wrist Pitch, Wrist Roll]
    - 6-DOF: [Base Yaw, Shoulder Pitch, Elbow Pitch, Wrist Pitch, Wrist Yaw/Roll, Tool Roll] (Full SO(3) 6-DoF Dexterity)
    """

    def __init__(self, dof: int = 6, logger: Optional[logging.Logger] = None, config: Optional[dict] = None):
        if dof not in (3, 4, 5, 6):
            raise ValueError(f"Invalid dof: {dof}. Must be 3, 4, 5, or 6.")
        self.dof = dof
        self.logger = logger or logging.getLogger("IKSolver")

        # Sensible defaults for link lengths matching Franka Hand URDF
        self.config = {
            "base_height": 0.175,  # Distance from base_link origin to joint2 (0.025 + 0.15)
            "l1": 0.25,  # Upper arm length (joint2 to joint3)
            "l2": 0.20,  # Forearm length (joint3 to joint4)
            "l_hand": 0.1534,  # 5-DOF: link4(0.03) + link5(0.02) + Franka TCP(0.1034)
            "l_hand_6dof": 0.1734,  # 6-DOF: link4(0.03) + link5(0.02) + link6(0.02) + Franka TCP(0.1034)
            "l_hand_4dof": 0.1334,  # 4-DOF: link4(0.03) + Franka TCP(0.1034)
            "l2_3dof": 0.3034,  # 3-DOF: link3(0.20) + Franka TCP(0.1034)
            "base_z_offset": 0.0,
        }
        if config:
            self.config.update(config)

    def forward_kinematics(self, *joints: float) -> Tuple[float, float, float, float]:
        """
        Computes FK for 3, 4, 5, or 6 DOF arm.
        Returns (x, y, z, yaw) in base_link frame.
        """
        if self.dof == 3:
            t1, t2, t3 = joints
            base_h = self.config["base_height"] + self.config["base_z_offset"]
            r = self.config["l1"] * math.sin(t2) + self.config["l2_3dof"] * math.sin(t2 + t3)
            z = base_h + self.config["l1"] * math.cos(t2) + self.config["l2_3dof"] * math.cos(t2 + t3)
            x = r * math.cos(t1)
            y = r * math.sin(t1)
            return x, y, z, t1

        elif self.dof == 4:
            t1, t2, t3, t4 = joints
            base_h = self.config["base_height"] + self.config["base_z_offset"]
            r_wrist = self.config["l1"] * math.sin(t2) + self.config["l2"] * math.sin(t2 + t3)
            z_wrist = base_h + self.config["l1"] * math.cos(t2) + self.config["l2"] * math.cos(t2 + t3)
            total_pitch = t2 + t3 + t4
            l_hand = self.config.get("l_hand_4dof", self.config["l_hand"])
            r_grasp = r_wrist + l_hand * math.sin(total_pitch)
            z_grasp = z_wrist + l_hand * math.cos(total_pitch)
            x = r_grasp * math.cos(t1)
            y = r_grasp * math.sin(t1)
            return x, y, z_grasp, t1

        elif self.dof == 5:
            t1, t2, t3, t4, t5 = joints
            base_h = self.config["base_height"]
            x3 = self.config["l1"] * math.sin(t2)
            z3 = base_h + self.config["l1"] * math.cos(t2)
            x4 = x3 + self.config["l2"] * math.sin(t2 + t3)
            z4 = z3 + self.config["l2"] * math.cos(t2 + t3)
            total_pitch = t2 + t3 + t4
            x_end_local = x4 + self.config["l_hand"] * math.sin(total_pitch)
            z_end = z4 + self.config["l_hand"] * math.cos(total_pitch)
            x_end = x_end_local * math.cos(t1)
            y_end = x_end_local * math.sin(t1)
            yaw = (math.pi / 2.0) + t1 - t5
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            return x_end, y_end, z_end, yaw

        else:  # 6 DOF
            t1, t2, t3, t4, t5, t6 = joints
            base_h = self.config["base_height"]
            l1 = self.config["l1"]
            l2 = self.config["l2"]
            l_hand = self.config.get("l_hand_6dof", 0.1734)

            # Base rotation R01
            c1, s1 = math.cos(t1), math.sin(t1)
            R01 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])

            # Arm joints kinematics
            p2 = np.array([0.0, 0.0, base_h])
            v2 = np.array([l1 * math.sin(t2), 0.0, l1 * math.cos(t2)])
            v3 = np.array([l2 * math.sin(t2 + t3), 0.0, l2 * math.cos(t2 + t3)])
            pw = p2 + R01 @ (v2 + v3)

            # Spherical wrist orientation
            c23, s23 = math.cos(t2 + t3), math.sin(t2 + t3)
            Ry23 = np.array([[c23, 0.0, s23], [0.0, 1.0, 0.0], [-s23, 0.0, c23]])
            R03 = R01 @ Ry23

            c4, s4 = math.cos(t4), math.sin(t4)
            c5, s5 = math.cos(t5), math.sin(t5)
            c6, s6 = math.cos(t6), math.sin(t6)
            Ry4 = np.array([[c4, 0.0, s4], [0.0, 1.0, 0.0], [-s4, 0.0, c4]])
            Rx5 = np.array([[1.0, 0.0, 0.0], [0.0, c5, -s5], [0.0, s5, c5]])
            Rz6 = np.array([[c6, -s6, 0.0], [s6, c6, 0.0], [0.0, 0.0, 1.0]])

            R06 = R03 @ Ry4 @ Rx5 @ Rz6
            # Gripper TCP position (projected along tool +Z approach axis)
            ptcp = pw + l_hand * R06[:, 2]
            yaw = math.atan2(R06[1, 1], R06[0, 1])
            return float(ptcp[0]), float(ptcp[1]), float(ptcp[2]), float(yaw)

    def inverse_kinematics(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        yaw: float = 0.0,
        rotation_matrix: Optional[np.ndarray] = None,
    ) -> Optional[List[float]]:
        """
        Computes IK for 3, 4, 5, or 6 DOF arm.
        Returns a list of joint angles, or None if unreachable.
        """
        if self.dof == 3:
            t1 = math.atan2(target_y, target_x)
            r = math.sqrt(target_x**2 + target_y**2)
            base_h = self.config["base_height"] + self.config["base_z_offset"]
            z_adj = target_z - base_h
            d_sq = r**2 + z_adj**2
            d = math.sqrt(d_sq)
            l1 = self.config["l1"]
            l2 = self.config["l2_3dof"]
            if d > (l1 + l2) or d < abs(l1 - l2):
                self.logger.error(f"[IK 3-DOF] Target unreachable: D={d:.3f} out of bounds")
                return None
            cos_t3 = max(-1.0, min(1.0, (d_sq - l1**2 - l2**2) / (2.0 * l1 * l2)))
            t3 = math.acos(cos_t3)
            alpha = math.atan2(r, z_adj)
            beta = math.atan2(l2 * math.sin(t3), l1 + l2 * math.cos(t3))
            t2 = alpha - beta
            joints = [t1, t2, t3]
            fx, fy, fz, _ = self.forward_kinematics(*joints)
            if math.sqrt((target_x - fx) ** 2 + (target_y - fy) ** 2 + (target_z - fz) ** 2) > 0.005:
                self.logger.warning("[IK 3-DOF] Target position unreachable within 5mm tolerance")
                return None
            return joints

        elif self.dof == 4:
            t1 = math.atan2(target_y, target_x)
            r_target = math.sqrt(target_x**2 + target_y**2)
            r_wrist = r_target
            l_hand = self.config.get("l_hand_4dof", self.config["l_hand"])
            z_wrist = target_z + l_hand
            base_h = self.config["base_height"] + self.config["base_z_offset"]
            z_adj = z_wrist - base_h
            d_sq = r_wrist**2 + z_adj**2
            d = math.sqrt(d_sq)
            l1 = self.config["l1"]
            l2 = self.config["l2"]
            if d > (l1 + l2) or d < abs(l1 - l2):
                self.logger.error(f"[IK 4-DOF] Target unreachable: D={d:.3f} out of bounds")
                return None
            cos_t3 = max(-1.0, min(1.0, (d_sq - l1**2 - l2**2) / (2.0 * l1 * l2)))
            t3 = math.acos(cos_t3)
            alpha = math.atan2(r_wrist, z_adj)
            beta = math.atan2(l2 * math.sin(t3), l1 + l2 * math.cos(t3))
            t2 = alpha - beta
            t4 = math.pi - (t2 + t3)
            t4 = math.atan2(math.sin(t4), math.cos(t4))

            # Joint limit validation
            if abs(t2) > 2.5 or abs(t3) > 2.5 or abs(t4) > 2.2:
                self.logger.warning(f"[IK 4-DOF] Joint limit exceeded: t2={t2:.2f}, t3={t3:.2f}, t4={t4:.2f}")
                return None

            joints = [t1, t2, t3, t4]
            fx, fy, fz, _ = self.forward_kinematics(*joints)
            if math.sqrt((target_x - fx) ** 2 + (target_y - fy) ** 2 + (target_z - fz) ** 2) > 0.005:
                self.logger.warning("[IK 4-DOF] Target position unreachable within 5mm tolerance")
                return None
            return joints

        elif self.dof == 5:
            t1 = math.atan2(target_y, target_x)
            t5_raw = (math.pi / 2.0) + t1 - yaw
            t5 = math.atan2(math.sin(2.0 * t5_raw), math.cos(2.0 * t5_raw)) / 2.0
            r_target = math.sqrt(target_x**2 + target_y**2)
            r_wrist = r_target
            z_wrist = target_z + self.config["l_hand"]
            dz = z_wrist - self.config["base_height"]
            dr = r_wrist
            D = math.sqrt(dr**2 + dz**2)
            l1 = self.config["l1"]
            l2 = self.config["l2"]
            if D > (l1 + l2) or D < abs(l1 - l2):
                self.logger.error(f"[IK 5-DOF] Target unreachable: D={D:.3f} out of bounds")
                return None
            cos_t3 = max(-1.0, min(1.0, (D**2 - l1**2 - l2**2) / (2.0 * l1 * l2)))
            t3 = math.acos(cos_t3)
            alpha = math.atan2(dz, dr)
            beta = math.atan2(l2 * math.sin(t3), l1 + l2 * math.cos(t3))
            t2 = (math.pi / 2.0) - (alpha + beta)
            t4 = math.pi - (t2 + t3)

            # Strictly enforce physical pitch limit (±2.2 rad / ±126.1°)
            if abs(t4) > 2.2:
                self.logger.warning(
                    f"[IK 5-DOF] Wrist pitch {math.degrees(t4):.1f}° exceeds limit ±126.1° (2.2 rad)."
                )
                return None

            if abs(t2) > 2.5 or abs(t3) > 2.5 or abs(t5) > (math.pi / 2.0 + 1e-4):
                self.logger.warning(f"[IK 5-DOF] Joint limit exceeded: t2={t2:.2f}, t3={t3:.2f}, t5={t5:.2f}")
                return None

            joints = [t1, t2, t3, t4, t5]
            fx, fy, fz, _ = self.forward_kinematics(*joints)
            pos_err = math.sqrt((target_x - fx) ** 2 + (target_y - fy) ** 2 + (target_z - fz) ** 2)
            if pos_err > 0.005:
                self.logger.warning(f"[IK 5-DOF] FK verification error {pos_err*1000:.1f} mm exceeds 5.0 mm threshold")
                return None
            return joints

        else:  # 6-DOF Analytical Decoupled Kinematics (Full SO(3) Dexterity)
            base_h = self.config["base_height"]
            l1 = self.config["l1"]
            l2 = self.config["l2"]
            l_hand = self.config.get("l_hand_6dof", 0.1734)

            if rotation_matrix is not None:
                R_target = np.array(rotation_matrix, dtype=np.float64)
                # Find which column corresponds to approach (points down into table, negative Z component)
                z_comps = [R_target[2, i] for i in range(3)]
                min_idx = int(np.argmin(z_comps))
                if R_target[2, min_idx] < -0.3:
                    approach = R_target[:, min_idx]
                else:
                    approach = R_target[:, 2] if R_target[2, 2] < 0 else np.array([0.0, 0.0, -1.0])
            else:
                approach = np.array([0.0, 0.0, -1.0])
                c = math.cos(yaw)
                s = math.sin(yaw)
                col1 = np.array([c, s, 0.0])
                col2 = np.array([0.0, 0.0, -1.0])
                col0 = np.cross(col1, col2)
                R_target = np.column_stack([col0, col1, col2])

            # For top-down or near-vertical tabletop grasping:
            # approach vector points from wrist towards object (negative Z)
            # pw is wrist center position: pw = ptcp - l_hand * approach
            pw = np.array([target_x, target_y, target_z]) - l_hand * approach
            xw, yw, zw = pw[0], pw[1], pw[2]

            t1 = math.atan2(yw, xw)
            rw = math.sqrt(xw**2 + yw**2)
            dz = zw - base_h
            D = math.sqrt(rw**2 + dz**2)

            if D > (l1 + l2) or D < abs(l1 - l2):
                self.logger.error(f"[IK 6-DOF] Target unreachable: D={D:.3f} out of bounds")
                return None

            cos_t3 = max(-1.0, min(1.0, (D**2 - l1**2 - l2**2) / (2.0 * l1 * l2)))
            t3 = math.acos(cos_t3)
            alpha = math.atan2(dz, rw)
            beta = math.atan2(l2 * math.sin(t3), l1 + l2 * math.cos(t3))
            t2 = (math.pi / 2.0) - (alpha + beta)

            # Solve spherical wrist joints (t4, t5, t6) from relative rotation R36
            c1, s1 = math.cos(t1), math.sin(t1)
            R01 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
            c23, s23 = math.cos(t2 + t3), math.sin(t2 + t3)
            Ry23 = np.array([[c23, 0.0, s23], [0.0, 1.0, 0.0], [-s23, 0.0, c23]])
            R03 = R01 @ Ry23
            R36 = R03.T @ R_target

            # Check if approach is top-down (near vertical) or general 3D
            if abs(approach[2]) > 0.8:
                # Top-down analytical solution (avoids wrist singularity & ensures exact vertical orientation)
                t4 = math.pi - (t2 + t3)
                t5 = 0.0
                t6_raw = (math.pi / 2.0) + t1 - yaw
                t6 = math.atan2(math.sin(t6_raw), math.cos(t6_raw))
            else:
                s5 = max(-1.0, min(1.0, -R36[1, 2]))
                t5 = math.asin(s5)
                t4 = math.atan2(R36[0, 2], R36[2, 2])
                t6 = math.atan2(R36[1, 0], R36[1, 1])

            t4 = math.atan2(math.sin(t4), math.cos(t4))
            t5 = math.atan2(math.sin(t5), math.cos(t5))
            t6 = math.atan2(math.sin(t6), math.cos(t6))

            # Joint limits check
            if abs(t2) > 2.5 or abs(t3) > 2.5 or abs(t4) > 2.5 or abs(t5) > 2.5:
                self.logger.warning(f"[IK 6-DOF] Joint limit exceeded: t2={t2:.2f}, t3={t3:.2f}, t4={t4:.2f}, t5={t5:.2f}")
                return None

            joints = [t1, t2, t3, t4, t5, t6]
            fx, fy, fz, _ = self.forward_kinematics(*joints)
            pos_err = math.sqrt((target_x - fx) ** 2 + (target_y - fy) ** 2 + (target_z - fz) ** 2)
            if pos_err > 0.010:
                self.logger.warning(f"[IK 6-DOF] FK verification error {pos_err*1000:.1f} mm exceeds 10.0 mm threshold")
                return None
            return joints

    def validate_trajectory(
        self, waypoints: List[Tuple[float, float, float, float]]
    ) -> Tuple[bool, List[Optional[List[float]]]]:
        """
        Validates if a series of (x, y, z, yaw) waypoints are all reachable.
        Returns (all_valid, list_of_joint_configs)
        """
        all_valid = True
        results = []
        for wp in waypoints:
            joints = self.inverse_kinematics(*wp)
            results.append(joints)
            if joints is None:
                all_valid = False
        return all_valid, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IKSolverTest")
    for dof in [3, 4, 5, 6]:
        logger.info(f"Running IK Solver Self-Test ({dof}-DOF)...")
        solver = IKSolver(dof=dof, logger=logger)
        targets = [
            (0.25, 0.0, 0.25, 0.0),
            (0.30, 0.1, 0.26, math.pi / 4),
            (0.20, -0.15, 0.35, -math.pi / 2),
        ]
        for x, y, z, yaw in targets:
            joints = solver.inverse_kinematics(x, y, z, yaw)
            if joints:
                fx, fy, fz, fyaw = solver.forward_kinematics(*joints)
                err = math.sqrt((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2)
                logger.info(
                    f"Target: ({x:.3f}, {y:.3f}, {z:.3f}, yaw={math.degrees(yaw):.1f}°) -> Error: {err*1000:.3f} mm"
                )
            else:
                logger.error(f"Target unreachable: {x}, {y}, {z}")
