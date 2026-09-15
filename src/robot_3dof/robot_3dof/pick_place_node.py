#!/usr/bin/env python3
"""
Pick-and-Place Orchestrator Node.

Listens for grasp poses from the grasp detection node, computes IK,
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

from robot_3dof.ik_solver import solve_ik, forward_kinematics


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
    """Pick-and-place orchestrator for 3 DoF robot with gripper."""

    def __init__(self):
        super().__init__("pick_place_node")

        # Parameters
        self.declare_parameter("pre_grasp_offset_z", 0.08)
        self.declare_parameter("post_grasp_lift_z", 0.10)
        self.declare_parameter("place_position.x", 0.20)
        self.declare_parameter("place_position.y", -0.15)
        self.declare_parameter("place_position.z", 0.35)
        self.declare_parameter("gripper_open_position", 0.03)
        self.declare_parameter("gripper_close_position", 0.005)
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

        # Current joint state
        self.current_joints = [0.0, 0.0, 0.0]

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

        self.get_logger().info("✅ Action servers connected! Moving to home position...")
        self._move_to_home()

        with self.lock:
            self.state = State.IDLE
        self.get_logger().info("🤖 Pick-and-Place Node ready! Waiting for grasp poses...")

    def _joint_state_callback(self, msg: JointState):
        """Update current joint positions."""
        for i, name in enumerate(msg.name):
            if name == "joint1" and i < len(msg.position):
                self.current_joints[0] = msg.position[i]
            elif name == "joint2" and i < len(msg.position):
                self.current_joints[1] = msg.position[i]
            elif name == "joint3" and i < len(msg.position):
                self.current_joints[2] = msg.position[i]

    def _grasp_callback(self, msg: PoseStamped):
        """Handle new grasp pose from detection node."""
        with self.lock:
            if self.state != State.IDLE:
                return
            self.state = State.MOVING_TO_PRE_GRASP

        pos = msg.pose.position
        self.get_logger().info(
            f"🎯 Target grasp received at x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}"
        )

        # Run the pick and place in a background worker thread
        worker = threading.Thread(
            target=self._execute_pick_and_place,
            args=(pos.x, pos.y, pos.z),
            daemon=True
        )
        worker.start()

    def _execute_pick_and_place(self, gx: float, gy: float, gz: float):
        """Execute the full pick-and-place sequence."""
        try:
            # gz is the object centroid. Finger pad center is 0.025m below end_effector.
            # Aligning end_effector at gz + 0.025 puts the finger pads directly around the object.
            gripper_offset = 0.025
            grasp_z = gz + gripper_offset
            pre_grasp_z = grasp_z + self.pre_grasp_offset

            # 1. Open gripper
            self.get_logger().info("[1/8] Opening gripper...")
            if not self._send_gripper(self.gripper_open):
                self._abort("Failed to open gripper")
                return

            # 2. Move to pre-grasp
            self.get_logger().info(
                f"[2/8] Moving to pre-grasp ({gx:.3f}, {gy:.3f}, {pre_grasp_z:.3f})..."
            )
            joints = solve_ik(gx, gy, pre_grasp_z)
            if joints is None:
                self._abort(f"Pre-grasp position unreachable: ({gx:.3f}, {gy:.3f}, {pre_grasp_z:.3f})")
                return
            if not self._send_trajectory(list(joints)):
                self._abort("Failed to reach pre-grasp position")
                return

            # 3. Move down to grasp
            self.get_logger().info(
                f"[3/8] Moving down to grasp ({gx:.3f}, {gy:.3f}, {grasp_z:.3f})..."
            )
            joints = solve_ik(gx, gy, grasp_z)
            if joints is None:
                self._abort(f"Grasp position unreachable: ({gx:.3f}, {gy:.3f}, {grasp_z:.3f})")
                return
            if not self._send_trajectory(list(joints)):
                self._abort("Failed to reach grasp position")
                return

            # 4. Close gripper
            self.get_logger().info("[4/8] Closing gripper around object...")
            if not self._send_gripper(self.gripper_close):
                self._abort("Failed to close gripper")
                return

            # Short pause to let physics stabilize grasp
            time.sleep(0.5)

            # 5. Lift object
            lift_z = grasp_z + self.lift_height
            self.get_logger().info(f"[5/8] Lifting object to z={lift_z:.3f}...")
            joints = solve_ik(gx, gy, lift_z)
            if joints is None:
                self._abort(f"Lift position unreachable: ({gx:.3f}, {gy:.3f}, {lift_z:.3f})")
                return
            if not self._send_trajectory(list(joints)):
                self._abort("Failed to lift")
                return

            # 6. Move to place position
            px, py, pz = self.place_pos
            self.get_logger().info(
                f"[6/8] Moving to place position ({px:.3f}, {py:.3f}, {pz:.3f})..."
            )
            joints = solve_ik(px, py, pz)
            if joints is None:
                self._abort(f"Place position unreachable: ({px:.3f}, {py:.3f}, {pz:.3f})")
                return
            if not self._send_trajectory(list(joints)):
                self._abort("Failed to move to place")
                return

            # 7. Open gripper to release
            self.get_logger().info("[7/8] Opening gripper to release object...")
            if not self._send_gripper(self.gripper_open):
                self._abort("Failed to release gripper")
                return

            # 8. Retreat to home
            self.get_logger().info("[8/8] Retreating to home position...")
            self._move_to_home()

            self.get_logger().info("🎉 ✅ Pick-and-place sequence complete!")

        finally:
            with self.lock:
                self.state = State.IDLE

    def _send_trajectory(self, joint_positions: list) -> bool:
        """Send joint trajectory goal and wait for completion."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3"]

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0, 0.0, 0.0]
        duration_sec = int(self.move_duration)
        duration_nsec = int((self.move_duration - duration_sec) * 1e9)
        point.time_from_start = Duration(sec=duration_sec, nanosec=duration_nsec)

        goal.trajectory.points = [point]

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

        if not result_event.wait(timeout=self.move_duration + 5.0) or result is None:
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
        """Move robot to home position (slightly bent ready pose)."""
        home_joints = [0.0, 0.4, -0.4]
        self._send_trajectory(home_joints)

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
