#!/usr/bin/python3
"""
Grasp Detection Node — wraps AnyGrasp SDK with a fallback heuristic mode.

When use_anygrasp=false (default):
  Uses simple point cloud clustering + centroid to generate top-down grasp poses.
  No GPU or license required.

When use_anygrasp=true:
  Loads AnyGrasp SDK and runs full grasp detection on point cloud data.
  Requires: NVIDIA GPU, CUDA, AnyGrasp license + checkpoint.

Provides ROS2 service: /anygrasp/detect
Publishes: /anygrasp/grasp_markers (visualization_msgs/MarkerArray)
Subscribes: /camera/points (sensor_msgs/PointCloud2)
"""

import os
import pickle
import socket
import struct
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from builtin_interfaces.msg import Duration

import tf2_ros


class GraspDetectionNode(Node):
    """Grasp detection node with AnyGrasp SDK or fallback heuristic."""

    def __init__(self):
        super().__init__("grasp_detection_node")

        # Declare parameters
        self.declare_parameter("use_anygrasp", False)
        self.declare_parameter("test_scenario", "default")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("socket_path", "/tmp/anygrasp_ipc.sock")
        self.declare_parameter("max_gripper_width", 0.06)
        self.declare_parameter("gripper_height", 0.04)
        self.declare_parameter("top_down_grasp", True)
        self.declare_parameter("point_cloud_topic", "/camera/points")
        self.declare_parameter("camera_frame", "camera_optical_link")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("min_points", 50)
        self.declare_parameter("voxel_size", 0.005)
        self.declare_parameter("workspace_bounds.x", [0.10, 0.50])
        self.declare_parameter("workspace_bounds.y", [-0.20, 0.20])
        self.declare_parameter("workspace_bounds.z", [0.25, 0.45])

        # Read parameters
        self.use_anygrasp = self.get_parameter("use_anygrasp").value
        self.test_scenario = self.get_parameter("test_scenario").value
        self.socket_path = self.get_parameter("socket_path").value
        self.max_gripper_width = self.get_parameter("max_gripper_width").value
        self.min_points = self.get_parameter("min_points").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        pc_topic = self.get_parameter("point_cloud_topic").value

        ws_x = self.get_parameter("workspace_bounds.x").value
        ws_y = self.get_parameter("workspace_bounds.y").value
        ws_z = self.get_parameter("workspace_bounds.z").value
        self.workspace_bounds = {
            "x": ws_x, "y": ws_y, "z": ws_z
        }

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Latest point cloud storage
        self.latest_pc: PointCloud2 = None

        # Subscriber
        cb_group = ReentrantCallbackGroup()
        self.pc_sub = self.create_subscription(
            PointCloud2, pc_topic, self._pc_callback, 10,
            callback_group=cb_group
        )

        # Publisher for visualization
        self.marker_pub = self.create_publisher(
            MarkerArray, "/anygrasp/grasp_markers", 10
        )

        # Publisher for segmented workspace point cloud
        self.ws_pc_pub = self.create_publisher(
            PointCloud2, "/anygrasp/workspace_cloud", 10
        )

        # Service for grasp detection (simple request/response via topic)
        # Using a timer-triggered detection for demo simplicity
        self.grasp_poses_pub = self.create_publisher(
            PoseStamped, "/anygrasp/best_grasp", 10
        )

        # Periodic detection (every 2 seconds)
        self.detect_timer = self.create_timer(2.0, self._detect_callback)

        if self.use_anygrasp:
            self._init_anygrasp()

        mode = "AnyGrasp AI (IPC Service)" if self.use_anygrasp else "Heuristic Fallback"
        self.get_logger().info(f"Grasp Detection Node started — mode: {mode}")

    def _init_anygrasp(self):
        """Check connection to AnyGrasp IPC Service."""
        self.get_logger().info(f"AnyGrasp mode enabled via IPC socket: {self.socket_path}")
        if os.path.exists(self.socket_path):
            self.get_logger().info("✅ Found active AnyGrasp IPC socket!")
        else:
            self.get_logger().info("ℹ️ AnyGrasp IPC socket not yet created. Node will connect once service starts.")

    def _pc_callback(self, msg: PointCloud2):
        """Store latest point cloud."""
        self.latest_pc = msg

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to Nx3 numpy array of XYZ points."""
        points = []
        # Determine field offsets
        x_off = y_off = z_off = None
        for field in msg.fields:
            if field.name == "x":
                x_off = field.offset
            elif field.name == "y":
                y_off = field.offset
            elif field.name == "z":
                z_off = field.offset

        if x_off is None or y_off is None or z_off is None:
            self.get_logger().warn("PointCloud2 missing x/y/z fields")
            return np.array([]).reshape(0, 3)

        point_step = msg.point_step
        data = msg.data

        for i in range(msg.width * msg.height):
            offset = i * point_step
            x = struct.unpack_from("f", data, offset + x_off)[0]
            y = struct.unpack_from("f", data, offset + y_off)[0]
            z = struct.unpack_from("f", data, offset + z_off)[0]

            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])

        return np.array(points).reshape(-1, 3)

    def _transform_points_to_base(self, points: np.ndarray, source_frame: str = None) -> np.ndarray:
        """Transform points from camera frame to base frame using TF2."""
        frame = source_frame if source_frame else self.camera_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return points

        # Extract translation and rotation
        t = transform.transform.translation
        q = transform.transform.rotation

        # Quaternion to rotation matrix
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])

        # Apply transform
        return (R @ points.T).T + trans

    def _quat_to_rot(self, x, y, z, w) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ])

    def _filter_workspace(self, points: np.ndarray) -> np.ndarray:
        """Filter points to workspace bounds (in base frame)."""
        if points.shape[0] == 0:
            return points

        bounds = self.workspace_bounds
        mask = (
            (points[:, 0] >= bounds["x"][0]) & (points[:, 0] <= bounds["x"][1]) &
            (points[:, 1] >= bounds["y"][0]) & (points[:, 1] <= bounds["y"][1]) &
            (points[:, 2] >= bounds["z"][0]) & (points[:, 2] <= bounds["z"][1])
        )
        return points[mask]

    def _heuristic_grasp_detection(
        self, points: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Simple heuristic: cluster points and create top-down grasp poses.
        Returns list of (position_xyz, confidence).
        """
        if points.shape[0] < self.min_points:
            return []

        # Cluster points using scipy.spatial.KDTree
        try:
            from scipy.spatial import KDTree
            tree = KDTree(points)
            visited = np.zeros(len(points), dtype=bool)
            clusters = []
            cluster_radius = 0.015  
            for i in range(len(points)):
                if visited[i]:
                    continue
                neighbors = tree.query_ball_point(points[i], r=cluster_radius)
                if len(neighbors) < self.min_points:
                    continue
                cluster_indices = set(neighbors)
                queue = list(neighbors)
                while queue:
                    curr = queue.pop()
                    if visited[curr]:
                        continue
                    visited[curr] = True
                    cluster_indices.add(curr)
                    sub_nbrs = tree.query_ball_point(points[curr], r=cluster_radius)
                    if len(sub_nbrs) >= self.min_points // 2:
                        for n in sub_nbrs:
                            if not visited[n] and n not in cluster_indices:
                                cluster_indices.add(n)
                                queue.append(n)

                cluster_pts = points[list(cluster_indices)]
                if len(cluster_pts) >= self.min_points:
                    clusters.append(cluster_pts)

            grasps = []
            for cl in clusters:
                c = np.mean(cl, axis=0)
                conf = min(0.95, 0.5 + 0.5 * (len(cl) / 200.0))
                grasps.append((c, conf, np.eye(3), 0.04, 0.04))

            if grasps:
                if self.test_scenario == "transparent_bottle":
                    # In transparent bottle scenario, prioritize ghost grasp (Y < -0.03) to demonstrate the failure
                    grasps.sort(key=lambda g: 0 if g[0][1] < -0.03 else 1)
                else:
                    # Sort by distance to base (prefer nearest reachable object)
                    grasps.sort(key=lambda g: np.linalg.norm(g[0][:2]))
                return grasps
        except Exception as e:
            self.get_logger().warn(f"Clustering error: {e}, falling back to overall centroid")

        centroid = np.mean(points, axis=0)
        grasp_pos = np.array([centroid[0], centroid[1], centroid[2]])
        return [(grasp_pos, 0.8, np.eye(3), 0.04, 0.04)]

    def _rot_to_quat(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return float(qx), float(qy), float(qz), float(qw)

    def _call_anygrasp_service(self, points: np.ndarray) -> List[dict]:
        """Send point cloud to AnyGrasp service via Unix domain socket."""
        if not os.path.exists(self.socket_path):
            self.get_logger().debug(f"AnyGrasp IPC socket {self.socket_path} not ready yet.")
            return None

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(3.0)
        try:
            client.connect(self.socket_path)
            req = {
                "points": points,
                "optional_params": {
                    "dense_grasp": False,
                    "collision_detection": False,
                    "approach_steering": [0, 0, -1],
                    "approach_thresh": 0.4,
                }
            }
            payload = pickle.dumps(req, protocol=4)
            client.sendall(struct.pack("!I", len(payload)) + payload)

            # Read response length
            raw_len = client.recv(4)
            if not raw_len:
                return None
            msg_len = struct.unpack("!I", raw_len)[0]

            # Read response payload
            data = bytearray()
            while len(data) < msg_len:
                packet = client.recv(min(65536, msg_len - len(data)))
                if not packet:
                    break
                data.extend(packet)

            resp = pickle.loads(data)
            if resp.get("status") == "ok":
                return resp.get("grasps", [])
            else:
                self.get_logger().warn(f"AnyGrasp service error: {resp.get('message')}")
                return None
        except Exception as e:
            self.get_logger().warn(f"AnyGrasp IPC error: {e}")
            return None
        finally:
            client.close()

    def _create_pointcloud2(self, points: np.ndarray, frame_id: str) -> PointCloud2:
        """Construct PointCloud2 message from Nx3 numpy array."""
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        return msg

    def _anygrasp_detection(
        self, points: np.ndarray
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float]]:
        """Run AnyGrasp inference on point cloud via IPC service."""
        grasps_data = self._call_anygrasp_service(points)
        if grasps_data is None or len(grasps_data) == 0:
            self.get_logger().warn(
                "AnyGrasp service unavailable or returned 0 grasps; using heuristic fallback"
            )
            return self._heuristic_grasp_detection(points)

        results = []
        for g in grasps_data:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            rot = np.array(g["rotation"], dtype=np.float32)
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))
            results.append((pos, score, rot, width, depth))

        # Sort by confidence score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _apply_scenario_pointcloud_effects(self, points: np.ndarray) -> np.ndarray:
        """
        Simulate realistic sensor noise & optical degradation for benchmark scenarios:
        1. 'transparent_bottle': IR light transmits through glass body -> missing depth (dropout) + refraction ghost points.
        2. 'flat_object': Isolates ultra-thin disc (2.5mm height) on flat table.
        3. 'dense_clutter': Tests crowded objects.
        """
        if points.shape[0] == 0:
            return points

        if self.test_scenario == "transparent_bottle":
            # Glass bottle is located at X ~ 0.28, Y ~ 0.00, body Z in [0.255, 0.320]
            # 1. Simulate optical transmission: 85% of body points are dropped (missing depth / NaN)
            in_bottle_x = (points[:, 0] >= 0.25) & (points[:, 0] <= 0.31)
            in_bottle_y = (points[:, 1] >= -0.04) & (points[:, 1] <= 0.04)
            in_body_z = (points[:, 2] >= 0.255) & (points[:, 2] <= 0.320)
            bottle_body_mask = in_bottle_x & in_bottle_y & in_body_z

            filtered_points = points.copy()
            body_indices = np.where(bottle_body_mask)[0]
            if len(body_indices) > 0:
                drop_mask = np.random.rand(len(body_indices)) < 0.85
                drop_indices = body_indices[drop_mask]
                filtered_points = np.delete(filtered_points, drop_indices, axis=0)

            # 2. Simulate refraction & specular reflection: Ghost points floating in empty air
            n_ghost = 50
            np.random.seed(42)  # Consistent demonstration
            ghost_x = np.random.normal(0.280, 0.008, n_ghost)
            ghost_y = np.random.normal(-0.065, 0.010, n_ghost)  # Displaced by 6.5cm into air
            ghost_z = np.random.normal(0.275, 0.012, n_ghost)
            ghost_pts = np.column_stack([ghost_x, ghost_y, ghost_z])

            return np.vstack([filtered_points, ghost_pts])

        elif self.test_scenario == "flat_object":
            # In dedicated world, only flat disc is present at (0.28, 0.0, 0.251)
            return points

        elif self.test_scenario == "dense_clutter":
            # In dedicated world, mug and duck touch at (0.27, -0.02) and (0.27, 0.025)
            return points

        return points

    def _diagnose_grasp(
        self, pos: np.ndarray, rot: np.ndarray, width: float, depth: float, score: float
    ) -> dict:
        """
        Diagnose whether a grasp candidate suffers from known failure modes:
        - FAIL_GHOST_GRASP: AnyGrasp hallucinated grasp in empty air from optical refraction.
        - FAIL_TABLE_COLLISION: Gripper finger penetrates the rigid table surface (Z < 0.248m).
        - FAIL_FLAT_OBJECT: Object lacks vertical clearance for antipodal grasping.
        - FAIL_CLUTTER_COLLISION: Gripper fingers penetrate adjacent clutter objects.
        - OPTIMAL: Valid collision-free grasp.
        """
        u_x = rot[:, 0]
        norm_x = np.linalg.norm(u_x)
        u_x = u_x / norm_x if norm_x > 1e-4 else np.array([0.0, 0.0, -1.0])

        d = max(0.02, min(depth, 0.05))
        # Finger tip reaches down in approach direction
        finger_tip_z = pos[2] + d * u_x[2] if u_x[2] < 0 else pos[2] - 0.02

        table_surface_z = 0.250

        # 1. Ghost grasp detection (Refraction artifact outside bottle body)
        if self.test_scenario == "transparent_bottle":
            dist_to_bottle = np.hypot(pos[0] - 0.28, pos[1] - 0.0)
            if dist_to_bottle > 0.035 and pos[1] < -0.03:
                return {
                    "status": "FAIL_GHOST_GRASP",
                    "tag": "⚠️ [FAIL: GHOST GRASP]",
                    "desc": "Ảo giác điểm ma do khúc xạ quang học (Refraction)",
                    "is_failure": True,
                    "color": (1.0, 0.0, 0.8, 0.95),  # Magenta
                }

        # 2. Flat object low affordance
        if self.test_scenario == "flat_object":
            if pos[2] <= table_surface_z + 0.010:
                return {
                    "status": "FAIL_FLAT_OBJECT",
                    "tag": "⛔ [FAIL: FLAT OBJECT]",
                    "desc": "Vật quá dẹt (2.5mm), thiếu khe hở luồn ngón kẹp (No Clearance)",
                    "is_failure": True,
                    "color": (1.0, 0.5, 0.0, 0.95),  # Orange
                }

        # 3. Table collision
        if finger_tip_z < (table_surface_z - 0.002) or pos[2] < (table_surface_z + 0.004):
            penetration_mm = max(0.0, (table_surface_z - finger_tip_z) * 1000.0)
            return {
                "status": "FAIL_TABLE_COLLISION",
                "tag": "💥 [FAIL: TABLE COLLISION]",
                "desc": f"Ngón kẹp đâm sâu xuống bàn ({penetration_mm:.1f}mm < 250mm)",
                "is_failure": True,
                "color": (1.0, 0.1, 0.1, 0.95),  # Red
            }

        # 4. Dense clutter collision
        if self.test_scenario == "dense_clutter":
            # If grasping near the junction of mug & duck
            if abs(pos[1]) < 0.03 and width > 0.035:
                return {
                    "status": "FAIL_CLUTTER_COLLISION",
                    "tag": "⚡ [FAIL: CLUTTER COLLISION]",
                    "desc": "Ngón kẹp va chạm vật lân cận (Semantic Blindness)",
                    "is_failure": True,
                    "color": (1.0, 0.2, 0.0, 0.95),  # Red-Orange
                }

        return {
            "status": "OPTIMAL",
            "tag": "✅ [OPTIMAL]",
            "desc": "Điểm gắp an toàn hợp lệ",
            "is_failure": False,
            "color": (0.1, 0.9, 0.2, 0.95),  # Green
        }

    def _detect_callback(self):
        """Periodic detection callback."""
        if self.latest_pc is None:
            return

        # Convert point cloud
        points_cam = self._pointcloud2_to_xyz(self.latest_pc)
        if points_cam.shape[0] == 0:
            return

        # Transform to base frame
        source_frame = self.latest_pc.header.frame_id if self.latest_pc.header.frame_id else self.camera_frame
        points_base = self._transform_points_to_base(points_cam, source_frame=source_frame)

        # Filter workspace
        points_ws = self._filter_workspace(points_base)

        # Apply benchmark scenario sensor & optical effects
        points_ws = self._apply_scenario_pointcloud_effects(points_ws)

        if points_ws.shape[0] < self.min_points:
            self.get_logger().debug(
                f"Not enough points in workspace: {points_ws.shape[0]}"
            )
            return

        # Publish segmented workspace point cloud for RViz2
        self.ws_pc_pub.publish(self._create_pointcloud2(points_ws, self.base_frame))

        # Run detection
        if self.use_anygrasp:
            grasps = self._anygrasp_detection(points_ws)
        else:
            grasps = self._heuristic_grasp_detection(points_ws)

        if not grasps:
            self.get_logger().debug("No grasps detected")
            return

        # Diagnose each detected grasp candidate
        diagnosed_grasps = []
        for g in grasps:
            pos, score, rot, width, depth = g
            diag = self._diagnose_grasp(pos, rot, width, depth, score)
            diagnosed_grasps.append((g, diag))

        # Publish best grasp
        best_pos = grasps[0][0]
        best_conf = grasps[0][1]
        best_rot = grasps[0][2]
        best_width = grasps[0][3]

        grasp_msg = PoseStamped()
        grasp_msg.header.stamp = self.get_clock().now().to_msg()
        grasp_msg.header.frame_id = self.base_frame
        grasp_msg.pose.position = Point(
            x=float(best_pos[0]),
            y=float(best_pos[1]),
            z=float(best_pos[2])
        )

        if self.use_anygrasp and best_conf > 0:
            qx, qy, qz, qw = self._rot_to_quat(best_rot)
            grasp_msg.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        else:
            grasp_msg.pose.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

        self.grasp_poses_pub.publish(grasp_msg)

        # Publish visualization markers (3D Gripper Wireframe, Labels, Arrows)
        self._publish_markers(diagnosed_grasps)

        # Print clean formatted ASCII diagnostic table in terminal
        method_str = "AnyGrasp AI" if self.use_anygrasp else "Heuristic Fallback"
        scen_str = f" [Scenario: {self.test_scenario.upper()}]" if self.test_scenario != "default" else ""
        print("\n" + "═" * 86)
        print(f"🤖 [{method_str}{scen_str}] Detected {len(grasps)} Grasps on 3D Objects:")
        print(f" {'Rank':<5} │ {'Score':<7} │ {'Position (X, Y, Z)':<24} │ {'Width':<7} │ {'Diagnostic / Failure Analysis'}")
        print("─" * 86)
        for idx, (g, diag) in enumerate(diagnosed_grasps[:5]):
            p = g[0]
            s = g[1]
            w = g[3]
            tag = "★ " if idx == 0 else "  "
            desc = diag["desc"]
            print(f" {tag}#{idx+1:<3} │ {s:<7.4f} │ [{p[0]:5.2f}, {p[1]:5.2f}, {p[2]:5.2f}] │ {w*100:4.1f}cm │ {desc}")
        print("═" * 86 + "\n")

    def _publish_markers(
        self, diagnosed_grasps: list
    ):
        """Publish 3D Gripper wireframe, text labels, and approach arrows to RViz2."""
        marker_array = MarkerArray()

        for i, (item, diag) in enumerate(diagnosed_grasps):
            pos = item[0]
            conf = item[1]
            rot = item[2]
            width = item[3] if len(item) > 3 else 0.04
            depth = item[4] if len(item) > 4 else 0.04

            # Use color from diagnostic status
            cr, cg, cb, ca = diag["color"]
            color = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=float(ca))

            # Extract gripper axes from rotation matrix
            # In GraspNet: rot[:, 0] is approach, rot[:, 1] is open/close
            u_x = rot[:, 0]
            u_y = rot[:, 1]
            norm_x = np.linalg.norm(u_x)
            u_x = u_x / norm_x if norm_x > 1e-4 else np.array([0.0, 0.0, -1.0])
            norm_y = np.linalg.norm(u_y)
            u_y = u_y / norm_y if norm_y > 1e-4 else np.array([0.0, 1.0, 0.0])

            w2 = max(0.015, min(width, 0.065)) / 2.0
            d = max(0.02, min(depth, 0.05))

            # ── 1. 3D Gripper Jaws Wireframe (Marker.LINE_LIST) ──
            b_center = pos - d * u_x
            l_base = b_center - w2 * u_y
            r_base = b_center + w2 * u_y
            l_tip = l_base + d * u_x
            r_tip = r_base + d * u_x
            stem = b_center - 0.03 * u_x

            jaw_marker = Marker()
            jaw_marker.header.stamp = self.get_clock().now().to_msg()
            jaw_marker.header.frame_id = self.base_frame
            jaw_marker.ns = "gripper_jaws"
            jaw_marker.id = i
            jaw_marker.type = Marker.LINE_LIST
            jaw_marker.action = Marker.ADD
            jaw_marker.scale.x = 0.003  # 3mm line thickness
            jaw_marker.color = color
            jaw_marker.lifetime = Duration(sec=2, nanosec=0)

            def to_pt(arr):
                return Point(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

            # Base crossbar
            jaw_marker.points.extend([to_pt(l_base), to_pt(r_base)])
            # Left finger
            jaw_marker.points.extend([to_pt(l_base), to_pt(l_tip)])
            # Right finger
            jaw_marker.points.extend([to_pt(r_base), to_pt(r_tip)])
            # Wrist stem
            jaw_marker.points.extend([to_pt(b_center), to_pt(stem)])

            marker_array.markers.append(jaw_marker)

            # ── 2. 3D Text Label (Marker.TEXT_VIEW_FACING) ──
            text_marker = Marker()
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.header.frame_id = self.base_frame
            text_marker.ns = "grasp_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2] + 0.035)
            )
            text_marker.scale.z = 0.013  # Font size
            if diag["is_failure"]:
                text_marker.color = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=1.0)
                text_marker.text = f"#{i+1}: {diag['tag']}"
            else:
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.2 if i == 0 else 0.8, a=1.0)
                text_marker.text = f"#{i+1}: S={conf:.3f} W={width*100:.1f}cm {diag['tag']}"

            text_marker.lifetime = Duration(sec=2, nanosec=0)
            marker_array.markers.append(text_marker)

            # ── 3. Approach Vector Arrow (Marker.ARROW) ──
            arrow_marker = Marker()
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.header.frame_id = self.base_frame
            arrow_marker.ns = "approach_arrows"
            arrow_marker.id = i
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            start_pt = Point(
                x=float(pos[0] - 0.05 * u_x[0]),
                y=float(pos[1] - 0.05 * u_x[1]),
                z=float(pos[2] - 0.05 * u_x[2])
            )
            end_pt = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            arrow_marker.points = [start_pt, end_pt]
            arrow_marker.scale.x = 0.003
            arrow_marker.scale.y = 0.006
            arrow_marker.scale.z = 0.008
            arrow_marker.color = color
            arrow_marker.lifetime = Duration(sec=2, nanosec=0)
            marker_array.markers.append(arrow_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = GraspDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
