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
                grasps.append((c, conf))

            if grasps:
                # Sort by distance to base (prefer nearest reachable object)
                grasps.sort(key=lambda g: np.linalg.norm(g[0][:2]))
                return grasps
        except Exception as e:
            self.get_logger().warn(f"Clustering error: {e}, falling back to overall centroid")

        centroid = np.mean(points, axis=0)
        grasp_pos = np.array([centroid[0], centroid[1], centroid[2]])
        return [(grasp_pos, 0.8)]

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

    def _anygrasp_detection(
        self, points: np.ndarray
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """Run AnyGrasp inference on point cloud via IPC service."""
        grasps_data = self._call_anygrasp_service(points)
        if grasps_data is None or len(grasps_data) == 0:
            self.get_logger().warn(
                "AnyGrasp service unavailable or returned 0 grasps; using heuristic fallback"
            )
            h_grasps = self._heuristic_grasp_detection(points)
            return [(pos, conf, np.eye(3)) for pos, conf in h_grasps]

        results = []
        for g in grasps_data:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            rot = np.array(g["rotation"], dtype=np.float32)
            results.append((pos, score, rot))

        # Sort by confidence score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

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

        if points_ws.shape[0] < self.min_points:
            self.get_logger().debug(
                f"Not enough points in workspace: {points_ws.shape[0]}"
            )
            return

        # Run detection
        if self.use_anygrasp:
            grasps = self._anygrasp_detection(points_ws)
        else:
            h_grasps = self._heuristic_grasp_detection(points_ws)
            grasps = [(pos, conf, np.eye(3)) for pos, conf in h_grasps]

        if not grasps:
            self.get_logger().debug("No grasps detected")
            return

        # Publish best grasp
        best_pos, best_conf, best_rot = grasps[0]
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
            # Top-down orientation (gripper pointing down)
            grasp_msg.pose.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

        self.grasp_poses_pub.publish(grasp_msg)

        # Publish visualization markers
        self._publish_markers(grasps)

        method_str = "AnyGrasp AI" if self.use_anygrasp else "Heuristic"
        self.get_logger().info(
            f"[{method_str}] Detected {len(grasps)} grasp(s). "
            f"Best: ({best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}) "
            f"score={best_conf:.3f}"
        )

    def _publish_markers(
        self, grasps: List[Tuple[np.ndarray, float, np.ndarray]]
    ):
        """Publish grasp poses as RViz markers."""
        marker_array = MarkerArray()

        for i, item in enumerate(grasps):
            pos, conf = item[0], item[1]
            rot = item[2] if len(item) > 2 else np.eye(3)

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.base_frame
            marker.ns = "grasps"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Determine approach direction vector
            if self.use_anygrasp:
                # In GraspNet, approach vector is rot[:, 0]
                approach_vec = rot[:, 0]
                norm = np.linalg.norm(approach_vec)
                if norm > 1e-4:
                    approach_vec = approach_vec / norm
                else:
                    approach_vec = np.array([0, 0, -1])
            else:
                approach_vec = np.array([0, 0, -1])

            start = Point(
                x=float(pos[0] - 0.05 * approach_vec[0]),
                y=float(pos[1] - 0.05 * approach_vec[1]),
                z=float(pos[2] - 0.05 * approach_vec[2])
            )
            end = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            marker.points = [start, end]

            marker.scale.x = 0.005  # Shaft diameter
            marker.scale.y = 0.01   # Head diameter
            marker.scale.z = 0.01   # Head length

            # Color: green for high confidence, red for low
            conf_norm = min(1.0, max(0.0, conf * 5.0)) if self.use_anygrasp else float(conf)
            marker.color = ColorRGBA(
                r=float(1.0 - conf_norm),
                g=float(conf_norm),
                b=0.0,
                a=0.8
            )
            marker.lifetime = Duration(sec=2, nanosec=0)

            marker_array.markers.append(marker)

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
