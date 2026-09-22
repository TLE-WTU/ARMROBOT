#!/usr/bin/python3
"""
Pick-and-Place Orchestrator Node for 5 DoF Robot.

Listens for grasp poses from the grasp detection node, computes IK (5 DOF),
and executes the pick-and-place sequence:
  1. Open gripper
  2. Move to pre-grasp pose (above target)
  3. Move to grasp pose
  4. Close gripper
  5. Lift object
  6. Move to place position
  7. Open gripper
  8. Retreat to home
"""

import math
import threading
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState

from robot_5dof.ik_solver import inverse_kinematics, forward_kinematics


class State(Enum):
    IDLE = auto()
    INITIALIZING = auto()
    MOVING_TO_PRE_GRASP = auto()
    MOVING_TO_GRASP = auto()
    CLOSING_GRIPPER = auto()
    LIFTING = auto()
    MOVING_TO_PLACE = auto()
    OPENING_GRIPPER = auto()
    RETREATING = auto()


class PickPlaceNode(Node):
    """Pick-and-place orchestrator for 5 DoF robot with gripper."""

    def __init__(self):
        super().__init__("pick_place_node")

        # Parameters
        self.declare_parameter("pre_grasp_offset_z", 0.025)
        self.declare_parameter("post_grasp_lift_z", 0.035)
        self.declare_parameter("place_position.x", 0.20)
        self.declare_parameter("place_position.y", -0.15)
        self.declare_parameter("place_position.z", 0.250)
        self.declare_parameter("gripper_open_position", 0.03)
        self.declare_parameter("gripper_close_position", 0.002)
        self.declare_parameter("move_duration_sec", 2.0)
        self.declare_parameter("gripper_duration_sec", 1.0)

        self.pre_grasp_offset = self.get_parameter("pre_grasp_offset_z").value
        self.lift_height = self.get_parameter("post_grasp_lift_z").value
        self.place_pos = (
            self.get_parameter("place_position.x").value,
            self.get_parameter("place_position.y").value,
            self.get_parameter("place_position.z").value,
        )
        self.gripper_open = self.get_parameter("gripper_open_position").value
        self.gripper_close = self.get_parameter("gripper_close_position").value
        self.move_duration = self.get_parameter("move_duration_sec").value
        self.gripper_duration = self.get_parameter("gripper_duration_sec").value

        # State machine and lock
        self.state = State.INITIALIZING
        self.lock = threading.Lock()

        # Current joint state (5 DOF + gripper)
        self.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Callback group for parallel callbacks
        self.cb_group = ReentrantCallbackGroup()

        # Subscribe to grasp pose
        self.grasp_sub = self.create_subscription(
            PoseStamped, "/anygrasp/best_grasp", self._grasp_callback, 10,
            callback_group=self.cb_group
        )

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10,
            callback_group=self.cb_group
        )

        # Action clients
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self.cb_group
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
            callback_group=self.cb_group
        )

        # Start connection & initialization in a background thread
        init_thread = threading.Thread(target=self._init_system, daemon=True)
        init_thread.start()

    def _init_system(self):
        """Wait for action servers and move to home position."""
        self.get_logger().info("Waiting for controllers to be available...")
        while not self.trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("Waiting for /joint_trajectory_controller...")

        while not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("Waiting for /gripper_controller...")

        self.get_logger().info("✅ Action servers connected! Waiting for controller activation...")
        time.sleep(1.0)
        for attempt in range(5):
            if self._move_to_home():
                break
            self.get_logger().info(f"Retrying home move (attempt {attempt+2}/5)...")
            time.sleep(1.0)

        with self.lock:
            self.state = State.IDLE
        self.get_logger().info("🤖 Pick-and-Place Node (5 DoF) ready! Waiting for grasp poses...")

    def _joint_state_callback(self, msg: JointState):
        """Update current joint positions."""
        for i, name in enumerate(msg.name):
            if name == "joint1" and i < len(msg.position):
                self.current_joints[0] = msg.position[i]
            elif name == "joint2" and i < len(msg.position):
                self.current_joints[1] = msg.position[i]
            elif name == "joint3" and i < len(msg.position):
                self.current_joints[2] = msg.position[i]
            elif name == "joint4" and i < len(msg.position):
                self.current_joints[3] = msg.position[i]
            elif name == "joint5" and i < len(msg.position):
                self.current_joints[4] = msg.position[i]

    def _extract_yaw_from_quaternion(self, q) -> float:
        """Extract yaw angle from quaternion orientation (Z-axis rotation)."""
        # Convert quaternion to yaw (rotation around Z-axis)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _grasp_callback(self, msg: PoseStamped):
        """Handle new grasp pose from detection node."""
        with self.lock:
            if self.state != State.IDLE:
                return
            self.state = State.MOVING_TO_PRE_GRASP

        pos = msg.pose.position
        yaw = self._extract_yaw_from_quaternion(msg.pose.orientation)
        self.get_logger().info(
            f"🎯 Target grasp received at x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}, yaw={math.degrees(yaw):.1f}°"
        )

        # Run the pick and place in a background worker thread
        worker = threading.Thread(
            target=self._execute_pick_and_place,
            args=(pos.x, pos.y, pos.z, yaw),
            daemon=True
        )
        worker.start()

    def _execute_pick_and_place(self, gx: float, gy: float, gz: float, yaw: float):
        try:
            # Table is at z=0.250 in world frame, which is z=0.225 in base_link frame (base_height/2 = 0.025).
            table_top_z = 0.225
            # Grasp center should ensure finger tips (20mm lower) are at least 2mm above the table surface
            min_grasp_z = table_top_z + 0.022
            
            # Calculate maximum safe Z at this (gx, gy) reach
            L1 = 0.25
            L2 = 0.20
            L_HAND = 0.08
            BASE_HEIGHT = 0.15
            
            r_target = math.sqrt(gx**2 + gy**2)
            max_reach = L1 + L2
            if r_target > max_reach:
                self._abort(f"Target too far: r={r_target:.3f} > max={max_reach:.3f}")
                return
                
            # dz_max = sqrt((L1+L2)^2 - r^2)
            dz_max = math.sqrt(max_reach**2 - r_target**2)
            z_wrist_max = BASE_HEIGHT + dz_max
            target_z_max = z_wrist_max - L_HAND
            
            # Maximum grasp Z to avoid wrist folding backwards into forearm (t4 > 90 deg)
            max_grasp_z = min(0.290, target_z_max - 0.03)
            
            grasp_z = max(min_grasp_z, min(max_grasp_z, gz))
            pre_grasp_z = min(grasp_z + self.pre_grasp_offset, target_z_max - 0.02)
            lift_z = min(grasp_z + self.lift_height, target_z_max - 0.01)
            
            px, py, pz = self.place_pos
            pre_place_z = pz + self.pre_grasp_offset
            
            # ==========================================
            # PRE-CHECK ENTIRE TRAJECTORY BEFORE MOVING!
            # ==========================================
            j_pre = inverse_kinematics(gx, gy, pre_grasp_z, yaw=yaw)
            j_grasp = inverse_kinematics(gx, gy, grasp_z, yaw=yaw)
            j_lift = inverse_kinematics(gx, gy, lift_z, yaw=yaw)
            j_pre_place = inverse_kinematics(px, py, pre_place_z, yaw=0.0)
            j_place = inverse_kinematics(px, py, pz, yaw=0.0)
            
            if not all([j_pre, j_grasp, j_lift, j_pre_place, j_place]):
                self.get_logger().error(f"❌ Trajectory Pre-check failed! Some points are unreachable.")
                self.get_logger().error(f"  Pre-grasp: {j_pre is not None} (z={pre_grasp_z:.3f})")
                self.get_logger().error(f"  Grasp:     {j_grasp is not None} (z={grasp_z:.3f})")
                self.get_logger().error(f"  Lift:      {j_lift is not None} (z={lift_z:.3f})")
                self.get_logger().error(f"  Pre-place: {j_pre_place is not None} (z={pre_place_z:.3f})")
                self.get_logger().error(f"  Place:     {j_place is not None} (z={pz:.3f})")
                self._abort("Kinematic trajectory planning failed.")
                return

            # 1. Open gripper
            self.get_logger().info("[1/8] Opening gripper...")
            if not self._send_gripper(self.gripper_open):
                self._abort("Failed to open gripper")
                return

            # 2. Move to pre-grasp
            self.get_logger().info(f"[2/8] Moving smoothly to pre-grasp ({gx:.3f}, {gy:.3f}, {pre_grasp_z:.3f}, yaw={math.degrees(yaw):.1f}°)...")
            if not self._send_trajectory(list(j_pre)):
                self._abort("Failed to reach pre-grasp position")
                return

            # 3. Move down to grasp
            self.get_logger().info(f"[3/8] Lowering down to grasp ({gx:.3f}, {gy:.3f}, {grasp_z:.3f})...")
            if not self._send_trajectory(list(j_grasp)):
                self._abort("Failed to reach grasp position")
                return

            # 4. Close gripper
            self.get_logger().info("[4/8] Closing gripper firmly around object...")
            if not self._send_gripper(self.gripper_close):
                self._abort("Failed to close gripper")
                return
            time.sleep(0.5)

            # 5. Lift object
            self.get_logger().info(f"[5/8] Lifting object smoothly to z={lift_z:.3f}...")
            if not self._send_trajectory(list(j_lift)):
                self._abort("Failed to lift object")
                return

            # 6. Move to pre-place
            self.get_logger().info(f"[6/8] Moving to place position ({px:.3f}, {py:.3f}, {pre_place_z:.3f})...")
            if not self._send_trajectory(list(j_pre_place)):
                self._abort("Failed to move to pre-place position")
                return

            # Lower to place
            if not self._send_trajectory(list(j_place)):
                self._abort("Failed to lower to place position")
                return

            # 7. Open gripper
            self.get_logger().info("[7/8] Opening gripper to release object...")
            if not self._send_gripper(self.gripper_open):
                self._abort("Failed to release object")
                return
            time.sleep(0.5)
            
            # Rise to pre-place before home
            if not self._send_trajectory(list(j_pre_place)):
                self.get_logger().warn("Failed to rise after place, returning home anyway")

            # 8. Return to home
            self.get_logger().info("[8/8] Returning to home position...")
            if not self._move_to_home():
                self._abort("Failed to return home")
                return

            self.get_logger().info("✅ Pick and Place sequence completed successfully!")
            
            with self.lock:
                self.state = State.IDLE

        except Exception as e:
            self.get_logger().error(f"Exception during pick and place: {e}")
            self._abort(str(e))


    def _send_trajectory(self, joint_positions: list, duration: float = None) -> bool:
        """
        Send smooth multi-point joint trajectory with S-curve (cosine) velocity profile.
        Eliminates jerk, motor stress, and sudden start/stop snapping.
        Now handles 4 joints: [joint1, joint2, joint3, joint4].
        """
        n_joints = 5
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]

        start_positions = list(self.current_joints)

        # Dynamic duration proportional to maximum joint displacement
        if duration is None:
            max_disp = max(abs(joint_positions[i] - start_positions[i]) for i in range(n_joints))
            # Smooth speed ~ 0.6 rad/s, min 1.5s
            duration = max(1.5, max_disp / 0.6)

        num_points = 20
        deltas = [joint_positions[i] - start_positions[i] for i in range(n_joints)]

        for k in range(1, num_points + 1):
            tau = k / float(num_points)
            t = tau * duration

            # S-curve smooth cosine interpolation:
            # s(tau) = 0.5 * (1 - cos(pi * tau))
            # v(tau) = (pi / (2 * duration)) * sin(pi * tau) * delta
            s = 0.5 * (1.0 - math.cos(math.pi * tau))
            v_factor = (math.pi / (2.0 * duration)) * math.sin(math.pi * tau)

            point = JointTrajectoryPoint()
            point.positions = [start_positions[i] + s * deltas[i] for i in range(n_joints)]
            point.velocities = [v_factor * deltas[i] for i in range(n_joints)]

            sec = int(t)
            nsec = int((t - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            goal.trajectory.points.append(point)

        event = threading.Event()
        goal_handle = None

        def goal_response_cb(future):
            nonlocal goal_handle
            try:
                goal_handle = future.result()
            except Exception as e:
                self.get_logger().error(f"Trajectory goal response error: {e}")
            finally:
                event.set()

        future = self.trajectory_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)

        if not event.wait(timeout=10.0) or goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected or timed out")
            return False

        result_event = threading.Event()
        result = None

        def get_result_cb(future):
            nonlocal result
            try:
                result = future.result()
            except Exception as e:
                self.get_logger().error(f"Trajectory result error: {e}")
            finally:
                result_event.set()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(get_result_cb)

        if not result_event.wait(timeout=duration + 5.0) or result is None:
            self.get_logger().error("Trajectory execution timed out")
            return False

        return True

    def _send_gripper(self, position: float) -> bool:
        """Send gripper command and wait for completion."""
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 50.0

        event = threading.Event()
        goal_handle = None

        def goal_response_cb(future):
            nonlocal goal_handle
            try:
                goal_handle = future.result()
            except Exception as e:
                self.get_logger().error(f"Gripper goal response error: {e}")
            finally:
                event.set()

        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)

        if not event.wait(timeout=10.0) or goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected or timed out")
            return False

        result_event = threading.Event()
        result = None

        def get_result_cb(future):
            nonlocal result
            try:
                result = future.result()
            except Exception as e:
                self.get_logger().error(f"Gripper result error: {e}")
            finally:
                result_event.set()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(get_result_cb)

        if not result_event.wait(timeout=self.gripper_duration + 5.0) or result is None:
            self.get_logger().error("Gripper execution timed out")
            return False

        return True

    def _move_to_home(self):
        """Move robot to natural crane ready pose with gripper pointing straight down."""
        # theta4 must compensate so total pitch = pi (gripper points down)
        # theta4 = pi - (theta2 + theta3)
        import math
        t2_home = 0.0
        t3_home = 0.5
        t4_home = 0.5
        home_joints = [0.0, t2_home, t3_home, t4_home, 0.0]  # [base, shoulder, elbow, wrist_pitch, wrist_roll]
        return self._send_trajectory(home_joints, duration=2.5)

    def _abort(self, reason: str):
        """Abort current operation."""
        self.get_logger().error(f"❌ ABORT: {reason}")


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
