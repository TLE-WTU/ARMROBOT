"""
Physics Simulation Environment for Standardized Grasp Rollout Evaluation.

Uses PyBullet in headless (DIRECT) or GUI mode to evaluate real physical grasp rollouts
on canonical YCB benchmark models (YcbTomatoSoupCan, YcbMustardBottle, YcbGelatinBox,
YcbBanana, YcbFoamBrick) under strict 4-DOF manipulator kinematic constraints.

Standard Physical Grasp Protocol (ICRA/IROS / Dex-Net Benchmark Standard):
1. Spawn canonical YCB object on table with random horizontal orientation and position.
2. Step physics until fully settled.
3. Render RGB-D camera with realistic sensor noise (Gaussian sigma=2mm) and dropout (10%).
4. Grasp planner predicts grasp pose (pos x, y, z, yaw, target_width).
5. Kinematic gripper approaches from pre-grasp offset.
6. Enforce true 4-DOF executed yaw: psi_executed = atan2(y, x) + pi/2 (mod pi).
7. Lower gripper to candidate z with top-down clearance.
8. Check table collision via PyBullet physics contacts (p.getContactPoints).
9. Gripper fingers close with 100N contact force towards target_width.
10. Verify dual-finger physical contact on object.
11. Gripper lifts object vertically 8cm and holds for 120 simulation steps (0.5 second).
12. Physical Success Criteria:
    - Object lifted stably (delta_z >= 0.04m)
    - No collision with table surface
    - Dual-finger force-closure contact confirmed
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pybullet as p
import pybullet_data


class PhysicsGraspEnvironment:
    """Headless PyBullet environment for evaluating physical grasp rollouts."""

    YCB_DIR = os.path.join(os.path.dirname(__file__), "assets", "ycb_objects")

    OBJECT_MAP = {
        "ycbtomatosoupcan": "YcbTomatoSoupCan",
        "tomato_soup": "YcbTomatoSoupCan",
        "can": "YcbTomatoSoupCan",
        "ycbmustardbottle": "YcbMustardBottle",
        "mustard": "YcbMustardBottle",
        "bottle": "YcbMustardBottle",
        "ycbgelatinbox": "YcbGelatinBox",
        "gelatin": "YcbGelatinBox",
        "box": "YcbGelatinBox",
        "ycbbanana": "YcbBanana",
        "banana": "YcbBanana",
        "ycbfoambrick": "YcbFoamBrick",
        "foam_brick": "YcbFoamBrick",
        "brick": "YcbFoamBrick",
    }

    def __init__(self, gui: bool = False, table_z: float = 0.225, gripper_aperture_m: float = 0.080):
        self.gui = gui
        self.table_z = float(table_z)
        self.gripper_aperture_m = float(gripper_aperture_m)
        self.gripper_half_stroke = self.gripper_aperture_m / 2.0 + 0.002
        self.client_id = -1
        self.table_id = -1
        self.plane_id = -1
        self.gripper_id = -1
        self.constraint_id = -1
        mm_tag = int(round(self.gripper_aperture_m * 1000))
        self.gripper_urdf_path = f"/tmp/benchmark_floating_gripper_{mm_tag}mm.urdf"
        self._ensure_gripper_urdf()

    def _ensure_gripper_urdf(self):
        """Creates a floating Franka-like parallel-jaw gripper URDF with parameterized aperture."""
        origin_y = self.gripper_aperture_m / 2.0 + 0.004
        base_width_y = round(2.0 * origin_y + 0.004, 3)
        half_stroke = self.gripper_half_stroke

        urdf_text = f"""<?xml version="1.0"?>
<robot name="floating_gripper">
  <link name="base">
    <visual><geometry><box size="0.04 {base_width_y:.3f} 0.04"/></geometry><material name="dark"><color rgba="0.2 0.2 0.2 1"/></material></visual>
    <collision><geometry><box size="0.04 {base_width_y:.3f} 0.04"/></geometry></collision>
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001"/></inertial>
  </link>
  <link name="left_finger">
    <visual><geometry><box size="0.02 0.008 0.06"/></geometry><material name="white"><color rgba="0.9 0.9 0.9 1"/></material></visual>
    <collision><geometry><box size="0.02 0.008 0.06"/></geometry></collision>
    <inertial><mass value="0.1"/><inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/></inertial>
  </link>
  <joint name="left_joint" type="prismatic">
    <parent link="base"/><child link="left_finger"/>
    <origin xyz="0 {origin_y:.4f} -0.04" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-{half_stroke:.4f}" upper="0.0" effort="150" velocity="0.2"/>
  </joint>
  <link name="right_finger">
    <visual><geometry><box size="0.02 0.008 0.06"/></geometry><material name="white"><color rgba="0.9 0.9 0.9 1"/></material></visual>
    <collision><geometry><box size="0.02 0.008 0.06"/></geometry></collision>
    <inertial><mass value="0.1"/><inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/></inertial>
  </link>
  <joint name="right_joint" type="prismatic">
    <parent link="base"/><child link="right_finger"/>
    <origin xyz="0 -{origin_y:.4f} -0.04" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="{half_stroke:.4f}" effort="150" velocity="0.2"/>
  </joint>
</robot>
"""
        with open(self.gripper_urdf_path, "w", encoding="utf-8") as f:
            f.write(urdf_text)

    def connect(self):
        """Connects to PyBullet physics server."""
        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

    def disconnect(self):
        """Disconnects from PyBullet physics server."""
        if self.client_id >= 0:
            p.disconnect()
            self.client_id = -1

    def reset_scene(
        self,
        object_type: str = "YcbTomatoSoupCan",
        target_pos: Tuple[float, float] = (0.35, 0.0),
        yaw_deg: float = 0.0,
    ) -> Tuple[int, np.ndarray]:
        """
        Resets table and spawns target canonical YCB object.
        Returns: (object_body_id, settled_object_pos)
        """
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

        table_half_thick = 0.025
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_half_thick])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_half_thick], rgbaColor=[0.85, 0.85, 0.85, 1.0])
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0.35, 0.0, self.table_z - table_half_thick],
        )
        p.changeDynamics(self.table_id, -1, lateralFriction=0.8, restitution=0.0)

        yaw_rad = math.radians(yaw_deg)
        orn = p.getQuaternionFromEuler([0, 0, yaw_rad])

        x, y = target_pos
        drop_z = self.table_z + 0.06

        canon_name = self.OBJECT_MAP.get(object_type.lower(), object_type)
        ycb_urdf = os.path.join(self.YCB_DIR, canon_name, "model.urdf")

        if os.path.exists(ycb_urdf):
            obj_id = p.loadURDF(ycb_urdf, [x, y, drop_z], orn)
            p.changeDynamics(obj_id, -1, lateralFriction=1.0, restitution=0.0)
        else:
            raise FileNotFoundError(
                f"Canonical YCB asset not found: '{ycb_urdf}'. "
                f"Please ensure YCB models are available in {self.YCB_DIR}."
            )

        for _ in range(80):
            p.stepSimulation()

        settled_pos, _ = p.getBasePositionAndOrientation(obj_id)
        return obj_id, np.array(settled_pos, dtype=np.float32)

    def capture_pointcloud(
        self,
        cam_eye: Tuple[float, float, float] = (0.35, 0.0, 0.70),
        cam_target: Tuple[float, float, float] = (0.35, 0.0, 0.225),
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        depth_noise_sigma: float = 0.002,
        dropout_ratio: float = 0.10,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders synthetic RGB-D from top-down / angled view and transforms into base coordinates.
        Injects realistic depth noise (Gaussian sigma) and random dropout (reflection artifacts).
        Returns: (points_base, colors)
        """
        if rng is None:
            rng = np.random.default_rng()

        cam_up = [0, 1, 0]
        view_matrix = p.computeViewMatrix(list(cam_eye), list(cam_target), cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(fov, float(width) / height, 0.1, 2.5)

        img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(img_arr[2], (height, width, 4))[:, :, :3] / 255.0
        depth_buffer = np.reshape(img_arr[3], (height, width))

        far, near = 2.5, 0.1
        depth = far * near / (far - (far - near) * depth_buffer)

        focal = (height / 2.0) / math.tan(math.radians(fov / 2.0))
        cx, cy = width / 2.0, height / 2.0

        xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
        pz = depth
        px = (xmap - cx) / focal * pz
        py = (ymap - cy) / focal * pz

        mask = (pz > 0.1) & (pz < 1.5)
        p_cam = np.ones((np.sum(mask), 4), dtype=np.float32)
        p_cam[:, 0] = px[mask]
        p_cam[:, 1] = -py[mask]
        p_cam[:, 2] = -pz[mask]

        view_matrix_np = np.array(view_matrix).reshape(4, 4).T
        cam2world = np.linalg.inv(view_matrix_np)

        p_world = (cam2world @ p_cam.T).T[:, :3].astype(np.float32)
        colors = rgb[mask].astype(np.float32)

        if depth_noise_sigma > 0 and len(p_world) > 0:
            noise = rng.normal(0.0, depth_noise_sigma, size=p_world.shape).astype(np.float32)
            p_world = p_world + noise

        if dropout_ratio > 0 and len(p_world) > 0:
            keep_mask = rng.random(len(p_world)) >= dropout_ratio
            p_world = p_world[keep_mask]
            colors = colors[keep_mask]

        ws_mask = (
            (p_world[:, 0] > 0.10) & (p_world[:, 0] < 0.55) &
            (p_world[:, 1] > -0.30) & (p_world[:, 1] < 0.30) &
            (p_world[:, 2] >= self.table_z - 0.01) & (p_world[:, 2] < self.table_z + 0.35)
        )
        return p_world[ws_mask], colors[ws_mask]

    def capture_dual_dynamic_pointclouds(
        self,
        target_pos: Tuple[float, float] = (0.35, 0.0),
        depth_noise_sigma: float = 0.002,
        dropout_ratio: float = 0.10,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Captures and fuses point clouds from 2 camera viewpoints."""
        if rng is None:
            rng = np.random.default_rng()

        tx, ty = target_pos
        cam1_eye = (tx + 0.10, ty - 0.08, self.table_z + 0.40)
        cam1_target = (tx, ty, self.table_z + 0.05)
        pts1, col1 = self.capture_pointcloud(
            cam1_eye, cam1_target, fov=65.0, depth_noise_sigma=depth_noise_sigma, dropout_ratio=dropout_ratio, rng=rng
        )

        cam2_eye = (tx - 0.12, ty + 0.10, self.table_z + 0.45)
        cam2_target = (tx, ty, self.table_z + 0.05)
        pts2, col2 = self.capture_pointcloud(
            cam2_eye, cam2_target, fov=65.0, depth_noise_sigma=depth_noise_sigma, dropout_ratio=dropout_ratio, rng=rng
        )

        if len(pts1) == 0:
            return pts2, col2
        if len(pts2) == 0:
            return pts1, col1

        fused_pts = np.vstack([pts1, pts2])
        fused_col = np.vstack([col1, col2])

        if len(fused_pts) > 6000:
            indices = rng.choice(len(fused_pts), 6000, replace=False)
            fused_pts = fused_pts[indices]
            fused_col = fused_col[indices]

        return fused_pts, fused_col

    def execute_grasp(
        self,
        grasp_pos: np.ndarray,
        yaw_rad: float,
        target_obj_id: int,
        target_width: float = 0.04,
        enforce_4dof_kinematics: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes physical approach, closing, dual-contact verification, and vertical lift.
        Enforces true 4-DOF physical arm kinematic orientation:
        psi_executed = atan2(y, x) + pi/2 (mod pi).

        Fingertip mechanics:
        - Target contact height z_contact = grasp_pos[2].
        - Gripper base is positioned at z_gripper_base = z_contact + 0.065m so that
          the finger pads enclose the target region from above without palm penetration.
        - Fingertips reach down to z_gripper_base - 0.070m.
        """
        init_obj_pos, _ = p.getBasePositionAndOrientation(target_obj_id)

        # 1. Determine true 4-DOF executed yaw
        th1 = math.atan2(grasp_pos[1], grasp_pos[0])
        feasible_yaw = (th1 + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi

        if enforce_4dof_kinematics:
            executed_yaw = feasible_yaw
        else:
            executed_yaw = yaw_rad

        diff = abs(yaw_rad - feasible_yaw) % math.pi
        mismatch_deg = math.degrees(min(diff, math.pi - diff))

        # 2. Compute gripper base target height
        z_contact = float(grasp_pos[2])
        gripper_base_z = z_contact + 0.065
        pre_grasp_z = gripper_base_z + 0.12

        gripper_orn = p.getQuaternionFromEuler([0, 0, executed_yaw])

        # 3. Spawn floating gripper at pre-grasp pose
        self.gripper_id = p.loadURDF(
            self.gripper_urdf_path,
            [grasp_pos[0], grasp_pos[1], pre_grasp_z],
            gripper_orn,
            useFixedBase=False,
        )
        p.changeDynamics(self.gripper_id, 0, lateralFriction=2.0)
        p.changeDynamics(self.gripper_id, 1, lateralFriction=2.0)

        cid = p.createConstraint(
            self.gripper_id, -1, -1, -1, p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0],
            [grasp_pos[0], grasp_pos[1], pre_grasp_z],
            gripper_orn,
        )

        # 4. Open fingers fully (targetPosition = 0.0)
        p.setJointMotorControl2(self.gripper_id, 0, p.POSITION_CONTROL, targetPosition=0.0, force=50)
        p.setJointMotorControl2(self.gripper_id, 1, p.POSITION_CONTROL, targetPosition=0.0, force=50)
        for _ in range(30):
            p.stepSimulation()

        # 5. Lower gripper to grasp pose
        for step in range(50):
            cur_z = pre_grasp_z - (pre_grasp_z - gripper_base_z) * (step / 50.0)
            p.changeConstraint(cid, [grasp_pos[0], grasp_pos[1], cur_z], gripper_orn, maxForce=200)
            p.stepSimulation()

        # 6. Physical table collision check via PyBullet contact query
        contacts_table = p.getContactPoints(bodyA=self.gripper_id, bodyB=self.table_id)
        fingertip_min_z = gripper_base_z - 0.070
        table_collision = (len(contacts_table) > 0) or (fingertip_min_z < (self.table_z - 0.003))

        # 7. Close fingers to grasp object with 100N contact force
        for _ in range(60):
            p.setJointMotorControl2(self.gripper_id, 0, p.POSITION_CONTROL, targetPosition=-self.gripper_half_stroke, force=100)
            p.setJointMotorControl2(self.gripper_id, 1, p.POSITION_CONTROL, targetPosition=self.gripper_half_stroke, force=100)
            p.stepSimulation()

        # 8. Verify dual-finger physical contact on target object
        c_left = p.getContactPoints(bodyA=self.gripper_id, bodyB=target_obj_id, linkIndexA=0)
        c_right = p.getContactPoints(bodyA=self.gripper_id, bodyB=target_obj_id, linkIndexA=1)
        has_dual_contact = (len(c_left) > 0) and (len(c_right) > 0)

        # 9. Lift object 8cm vertically
        lift_target_z = gripper_base_z + 0.08
        for step in range(60):
            cur_z = gripper_base_z + 0.08 * (step / 60.0)
            p.changeConstraint(cid, [grasp_pos[0], grasp_pos[1], cur_z], gripper_orn, maxForce=300)
            p.setJointMotorControl2(self.gripper_id, 0, p.POSITION_CONTROL, targetPosition=-self.gripper_half_stroke, force=100)
            p.setJointMotorControl2(self.gripper_id, 1, p.POSITION_CONTROL, targetPosition=self.gripper_half_stroke, force=100)
            p.stepSimulation()

        # 10. Hold for 240 simulation steps (1.0 second at 240 Hz) to test dynamic stability
        for _ in range(240):
            p.stepSimulation()

        final_obj_pos, _ = p.getBasePositionAndOrientation(target_obj_id)
        lift_height = final_obj_pos[2] - init_obj_pos[2]

        success = (lift_height >= 0.04) and (not table_collision) and has_dual_contact

        # Cleanup
        p.removeConstraint(cid)
        p.removeBody(self.gripper_id)

        return {
            "success": bool(success),
            "table_collision": bool(table_collision),
            "has_dual_contact": bool(has_dual_contact),
            "initial_obj_z": float(init_obj_pos[2]),
            "final_obj_z": float(final_obj_pos[2]),
            "lift_height": float(lift_height),
            "executed_yaw": float(executed_yaw),
            "yaw_mismatch_deg": float(mismatch_deg),
        }
