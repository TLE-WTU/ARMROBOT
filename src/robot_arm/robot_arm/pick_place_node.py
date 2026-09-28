#!/usr/bin/python3
"""
Unified Pick-and-Place Orchestrator Node.

Supports 3, 4, and 5 DOF arms using a single configurable FSM and unified IK solver.
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

from robot_arm.ik_solver import IKSolver

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
    def __init__(self):
        super().__init__("pick_place_node")

        self.declare_parameter("dof", 5)
        self.declare_parameter("pre_grasp_offset_z", 0.060)
        self.declare_parameter("post_grasp_lift_z", 0.060)
        self.declare_parameter("place_position.x", 0.20)
        self.declare_parameter("place_position.y", -0.15)
        self.declare_parameter("place_position.z", 0.250)
        self.declare_parameter("gripper_open_position", 0.035)
        self.declare_parameter("gripper_close_position", 0.005)
        self.declare_parameter("move_duration_sec", 2.0)
        self.declare_parameter("gripper_duration_sec", 1.0)
        
        # New parameters for magic numbers
        self.declare_parameter("table_top_z", 0.225)
        self.declare_parameter("min_grasp_z_margin", 0.022)

        self.dof = self.get_parameter("dof").value
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
        self.table_top_z = self.get_parameter("table_top_z").value
        self.min_grasp_z_margin = self.get_parameter("min_grasp_z_margin").value

        self.ik_solver = IKSolver(dof=self.dof, logger=self.get_logger())
        self.joint_names = [f"joint{i+1}" for i in range(self.dof)]

        self.state = State.INITIALIZING
        self.lock = threading.Lock()
        self.current_joints = [0.0] * self.dof

        self.cb_group = ReentrantCallbackGroup()
        self.grasp_sub = self.create_subscription(
            PoseStamped, "/anygrasp/best_grasp", self._grasp_callback, 10,
            callback_group=self.cb_group
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10,
            callback_group=self.cb_group
        )
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self.cb_group
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, "/gripper_controller/gripper_cmd",
            callback_group=self.cb_group
        )

        init_thread = threading.Thread(target=self._init_system, daemon=True)
        init_thread.start()

    def _init_system(self):
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
        self.get_logger().info(f"🤖 Pick-and-Place Node ({self.dof} DoF) ready! Waiting for grasp poses...")

    def _joint_state_callback(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                if i < len(msg.position):
                    self.current_joints[idx] = msg.position[i]

    def _extract_yaw_from_quaternion(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _grasp_callback(self, msg: PoseStamped):
        with self.lock:
            if self.state != State.IDLE:
                return
            self.state = State.MOVING_TO_PRE_GRASP

        pos = msg.pose.position
        yaw = self._extract_yaw_from_quaternion(msg.pose.orientation)
        self.get_logger().info(
            f"🎯 Target grasp received at x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}, yaw={math.degrees(yaw):.1f}°"
        )
        worker = threading.Thread(
            target=self._execute_pick_and_place,
            args=(pos.x, pos.y, pos.z, yaw),
            daemon=True
        )
        worker.start()

    def _execute_pick_and_place(self, gx: float, gy: float, gz: float, yaw: float):
        try:
            min_grasp_z = self.table_top_z + self.min_grasp_z_margin
            l1 = self.ik_solver.config['l1']
            l2 = self.ik_solver.config.get('l2', self.ik_solver.config.get('l2_3dof', 0.20))
            if self.dof == 4:
                l_hand = self.ik_solver.config.get('l_hand_4dof', self.ik_solver.config.get('l_hand', 0.0))
            elif self.dof == 5:
                l_hand = self.ik_solver.config.get('l_hand', 0.0)
            else:
                l_hand = 0.0
            base_height = self.ik_solver.config['base_height'] + self.ik_solver.config.get('base_z_offset', 0)
            
            r_target = math.sqrt(gx**2 + gy**2)
            max_reach = l1 + l2
            if r_target > max_reach:
                self._abort(f"Target too far: r={r_target:.3f} > max={max_reach:.3f}")
                return
                
            dz_max = math.sqrt(max_reach**2 - r_target**2)
            z_wrist_max = base_height + dz_max
            target_z_max = z_wrist_max - l_hand if self.dof > 3 else z_wrist_max
            
            max_grasp_z = min(0.350, target_z_max - 0.01)
            grasp_z = max(min_grasp_z, min(max_grasp_z, gz))
            pre_grasp_z = min(grasp_z + self.pre_grasp_offset, target_z_max)
            lift_z = min(grasp_z + self.lift_height, target_z_max)
            
            px, py, pz = self.place_pos
            pre_place_z = pz + self.pre_grasp_offset
            
            j_pre = self.ik_solver.inverse_kinematics(gx, gy, pre_grasp_z, yaw=yaw)
            j_grasp = self.ik_solver.inverse_kinematics(gx, gy, grasp_z, yaw=yaw)
            j_lift = self.ik_solver.inverse_kinematics(gx, gy, lift_z, yaw=yaw)
            j_pre_place = self.ik_solver.inverse_kinematics(px, py, pre_place_z, yaw=0.0)
            j_place = self.ik_solver.inverse_kinematics(px, py, pz, yaw=0.0)
            
            if not all([j_pre, j_grasp, j_lift, j_pre_place, j_place]):
                self.get_logger().error(f"❌ Trajectory Pre-check failed! Some points are unreachable.")
                self._abort("Kinematic trajectory planning failed.")
                return

            self.get_logger().info("[1/8] Opening gripper...")
            if not self._send_gripper(self.gripper_open):
                self._abort("Failed to open gripper")
                return

            self.get_logger().info(f"[2/8] Moving to pre-grasp ({gx:.3f}, {gy:.3f}, {pre_grasp_z:.3f})...")
            if not self._send_trajectory(j_pre):
                self._abort("Failed to reach pre-grasp")
                return

            self.get_logger().info(f"[3/8] Lowering to grasp ({gx:.3f}, {gy:.3f}, {grasp_z:.3f})...")
            if not self._send_trajectory(j_grasp):
                self._abort("Failed to reach grasp")
                return

            self.get_logger().info("[4/8] Closing gripper...")
            if not self._send_gripper(self.gripper_close):
                self._abort("Failed to close gripper")
                return
            time.sleep(0.5)

            self.get_logger().info(f"[5/8] Lifting object to z={lift_z:.3f}...")
            if not self._send_trajectory(j_lift):
                self._abort("Failed to lift")
                return

            self.get_logger().info(f"[6/8] Moving to place position ({px:.3f}, {py:.3f}, {pre_place_z:.3f})...")
            if not self._send_trajectory(j_pre_place):
                self._abort("Failed to move to pre-place")
                return

            if not self._send_trajectory(j_place):
                self._abort("Failed to lower to place")
                return

            self.get_logger().info("[7/8] Opening gripper...")
            if not self._send_gripper(self.gripper_open):
                self._abort("Failed to release")
                return
            time.sleep(0.5)
            
            if not self._send_trajectory(j_pre_place):
                self.get_logger().warn("Failed to rise after place, returning home anyway")

            self.get_logger().info("[8/8] Returning home...")
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
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names

        start_positions = list(self.current_joints)
        if duration is None:
            max_disp = max(abs(joint_positions[i] - start_positions[i]) for i in range(self.dof))
            duration = max(1.5, max_disp / 0.6)

        num_points = 20
        deltas = [joint_positions[i] - start_positions[i] for i in range(self.dof)]

        for k in range(1, num_points + 1):
            tau = k / float(num_points)
            t = tau * duration
            s = 0.5 * (1.0 - math.cos(math.pi * tau))
            v_factor = (math.pi / (2.0 * duration)) * math.sin(math.pi * tau)

            point = JointTrajectoryPoint()
            point.positions = [start_positions[i] + s * deltas[i] for i in range(self.dof)]
            point.velocities = [v_factor * deltas[i] for i in range(self.dof)]
            
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
                self.get_logger().error(f"Trajectory goal error: {e}")
            finally:
                event.set()

        future = self.trajectory_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)
        if not event.wait(10.0) or goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected/timeout")
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
        if not result_event.wait(duration + 5.0) or result is None:
            self.get_logger().error("Trajectory execution timed out")
            return False

        return True

    def _send_gripper(self, position: float) -> bool:
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
                self.get_logger().error(f"Gripper goal error: {e}")
            finally:
                event.set()

        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)
        if not event.wait(10.0) or goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected/timeout")
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
        if not result_event.wait(self.gripper_duration + 5.0) or result is None:
            self.get_logger().error("Gripper execution timed out")
            return False
        return True

    def _move_to_home(self) -> bool:
        home_joints = [0.0] * self.dof
        if self.dof >= 3:
            home_joints[1] = 0.0
            home_joints[2] = 0.5
        if self.dof >= 4:
            home_joints[3] = 0.5
        return self._send_trajectory(home_joints, duration=2.5)

    def _abort(self, reason: str):
        self.get_logger().error(f"❌ ABORT: {reason}")
        with self.lock:
            self.state = State.IDLE

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
