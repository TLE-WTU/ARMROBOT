#!/usr/bin/env python3
"""
Grasp Detection Node for Unified Robot Arm (3/4/5 DoF) — wraps AnyGrasp SDK with a fallback heuristic mode.

When use_anygrasp=false (default):
  Uses simple point cloud clustering + centroid to generate top-down grasp poses.
  Uses PCA to determine optimal gripper yaw orientation.
  No GPU or license required.

When use_anygrasp=true:
  Loads AnyGrasp SDK and runs full grasp detection on point cloud data.
  Requires: NVIDIA GPU, CUDA, AnyGrasp license + checkpoint.

Provides ROS2 service: /anygrasp/detect
Publishes: /anygrasp/grasp_markers (visualization_msgs/MarkerArray)
Subscribes: /camera/points (sensor_msgs/PointCloud2)
"""

import os
import sys

# Ensure execution under Python with ROS 2 support (Ubuntu 24.04 uses Python 3.12 for ROS 2 Jazzy)
if sys.version_info >= (3, 13) and "conda" in sys.executable.lower():
    for _sys_py in ("/usr/bin/python3.12", "/usr/bin/python3"):
        if os.path.exists(_sys_py):
            os.execv(_sys_py, [_sys_py] + sys.argv)

import json
import base64
import socket
import struct
import time
import csv
import math
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup
    from sensor_msgs.msg import PointCloud2, PointField
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    from builtin_interfaces.msg import Duration
    from rcl_interfaces.msg import ParameterDescriptor
    import tf2_ros
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:
        def __init__(self, *args, **kwargs):
            pass
    ReentrantCallbackGroup = None
    PointCloud2 = PointField = PoseStamped = Point = Quaternion = None
    Marker = MarkerArray = ColorRGBA = Duration = ParameterDescriptor = None
    tf2_ros = None

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None

from robot_arm.geometric_refinement import get_geometric_refiner, BaseGeometricRefiner
from robot_arm.perception_pipeline import (
    filter_workspace,
    compute_grasp_yaw_pca,
    yaw_to_quaternion_tuple,
    ransac_plane_segmentation,
    adaptive_geometric_reduction,
    enforce_virtual_safety_floor,
    is_in_drop_zone,
)


def encode_json(obj: Any) -> Any:
    """Encode numpy arrays into base64 JSON representation."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(obj.tobytes()).decode('utf-8'),
            "dtype": str(obj.dtype),
            "shape": obj.shape
        }
    elif isinstance(obj, dict):
        return {k: encode_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(encode_json(v) for v in obj)
    return obj


def decode_json(obj: Any) -> Any:
    """Decode numpy arrays from base64 JSON representation."""
    if isinstance(obj, dict) and "__ndarray__" in obj:
        b = base64.b64decode(obj["__ndarray__"])
        return np.frombuffer(b, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    elif isinstance(obj, dict):
        return {k: decode_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_json(v) for v in obj]
    return obj


class GraspDetectionNode(Node):
    """Grasp detection node with AnyGrasp SDK or fallback heuristic for 3/4/5 DoF."""

    def __init__(self):
        super().__init__("grasp_detection_node")

        # Declare parameters
        self.declare_parameter("dof", 5)
        self.declare_parameter("use_anygrasp", False, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("grasp_mode", "heuristic", ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("geometric_refinement", "pca", ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("test_scenario", "default")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("socket_path", "/tmp/anygrasp_ipc.sock")
        self.declare_parameter("max_gripper_width", 0.07)
        self.declare_parameter("gripper_height", 0.04)
        self.declare_parameter("top_down_grasp", True)
        self.declare_parameter("point_cloud_topic", "/camera/points")
        self.declare_parameter("camera_frame", "camera_optical_link")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("min_points", 30)
        self.declare_parameter("voxel_size", 0.005)
        self.declare_parameter("workspace_bounds.x", [0.10, 0.50])
        self.declare_parameter("workspace_bounds.y", [-0.20, 0.20])
        self.declare_parameter("workspace_bounds.z", [0.22, 0.45])
        self.declare_parameter("drop_zone.x_max", 0.24)
        self.declare_parameter("drop_zone.y_max", -0.12)

        # RANSAC & Benchmark parameters
        self.declare_parameter("enable_ransac", True, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("ransac_distance_threshold", 0.008)
        self.declare_parameter("ransac_max_iterations", 150)
        self.declare_parameter("target_point_count", 1024)
        self.declare_parameter("benchmark_csv_path", "")

        self.dof = self.get_parameter("dof").value
        raw_mode = self.get_parameter("grasp_mode").value
        self.grasp_mode = str(raw_mode).lower() if raw_mode else "heuristic"
        raw_anygrasp = self.get_parameter("use_anygrasp").value
        self.use_anygrasp = raw_anygrasp.lower() in ("true", "1", "yes") if isinstance(raw_anygrasp, str) else bool(raw_anygrasp)
        if self.grasp_mode == "hybrid" and self.use_anygrasp:
            self.use_anygrasp = True
        self.test_scenario = str(self.get_parameter("test_scenario").value)
        self.socket_path = str(self.get_parameter("socket_path").value)
        self.max_gripper_width = float(self.get_parameter("max_gripper_width").value)
        self.drop_zone_x_max = float(self.get_parameter("drop_zone.x_max").value)
        self.drop_zone_y_max = float(self.get_parameter("drop_zone.y_max").value)

        raw_geo = self.get_parameter("geometric_refinement").value
        self.geometric_refinement_name = str(raw_geo).lower() if raw_geo else "pca"
        try:
            self.refiner = get_geometric_refiner(self.geometric_refinement_name, max_gripper_width=self.max_gripper_width)
        except Exception:
            self.refiner = get_geometric_refiner("pca", max_gripper_width=self.max_gripper_width)
        self.min_points = int(self.get_parameter("min_points").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)

        raw_ransac = self.get_parameter("enable_ransac").value
        self.enable_ransac = raw_ransac.lower() in ("true", "1", "yes") if isinstance(raw_ransac, str) else bool(raw_ransac)
        self.ransac_distance_threshold = float(self.get_parameter("ransac_distance_threshold").value)
        self.ransac_max_iterations = int(self.get_parameter("ransac_max_iterations").value)
        self.target_point_count = int(self.get_parameter("target_point_count").value)
        
        csv_path = str(self.get_parameter("benchmark_csv_path").value)
        self.benchmark_csv_path = csv_path if csv_path else "/tmp/anygrasp_benchmark.csv"

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

        self.latest_pc: Optional[PointCloud2] = None

        cb_group = ReentrantCallbackGroup()
        self.pc_sub = self.create_subscription(
            PointCloud2, pc_topic, self._pc_callback, 10,
            callback_group=cb_group
        )

        self.marker_pub = self.create_publisher(MarkerArray, "/anygrasp/grasp_markers", 10)
        self.ws_pc_pub = self.create_publisher(PointCloud2, "/anygrasp/workspace_cloud", 10)
        self.table_pc_pub = self.create_publisher(PointCloud2, "/anygrasp/table_cloud", 10)
        self.object_pc_pub = self.create_publisher(PointCloud2, "/anygrasp/object_cloud", 10)
        self.grasp_poses_pub = self.create_publisher(PoseStamped, "/anygrasp/best_grasp", 10)

        # Periodic detection
        self.detect_timer = self.create_timer(2.0, self._detect_callback)

        if self.use_anygrasp:
            self._init_anygrasp()

        self._init_benchmark_csv()

        if self.grasp_mode == "hybrid":
            mode_desc = "Hybrid Ensemble (AnyGrasp AI + Geometric PCA/RANSAC)"
        elif self.use_anygrasp or self.grasp_mode == "anygrasp":
            mode_desc = "AnyGrasp AI (Deep Learning Only)"
        else:
            mode_desc = "Heuristic Geometric (RANSAC + PCA Fallback)"
        self.get_logger().info(f"🚀 Grasp Detection Node ({self.dof} DoF) started — mode: {mode_desc}")

    def _init_anygrasp(self):
        self.get_logger().info(f"AnyGrasp mode enabled via IPC socket: {self.socket_path}")
        if self._check_anygrasp_health():
            self.get_logger().info("✅ Found active AnyGrasp IPC socket!")
        else:
            self.get_logger().info("ℹ️ AnyGrasp IPC socket not yet healthy. Node will connect once service starts.")

    def _init_benchmark_csv(self):
        try:
            csv_dir = os.path.dirname(self.benchmark_csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            if not os.path.exists(self.benchmark_csv_path):
                with open(self.benchmark_csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "scenario", "ransac_enabled", "raw_points",
                        "object_points", "reduction_pct", "ransac_ms", "reduction_ms",
                        "inference_ms", "total_ms", "fps", "grasps_count",
                        "best_score", "best_yaw_deg", "table_collision"
                    ])
                self.get_logger().info(f"📊 Benchmark logger initialized: {self.benchmark_csv_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to initialize benchmark CSV: {e}")

    def _record_benchmark_row(
        self, raw_pts: int, obj_pts: int, reduct_pct: float, ransac_ms: float,
        reduct_ms: float, infer_ms: float, total_ms: float, fps: float,
        grasps_count: int, best_score: float, best_yaw_deg: float, collision_flag: bool
    ):
        try:
            with open(self.benchmark_csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{self.get_clock().now().nanoseconds / 1e9:.2f}",
                    self.test_scenario, self.enable_ransac, raw_pts, obj_pts,
                    f"{reduct_pct:.1f}", f"{ransac_ms:.2f}", f"{reduct_ms:.2f}",
                    f"{infer_ms:.2f}", f"{total_ms:.2f}", f"{fps:.1f}", grasps_count,
                    f"{best_score:.4f}", f"{best_yaw_deg:.1f}", collision_flag,
                ])
        except Exception as e:
            self.get_logger().warn(f"Failed to record benchmark row: {e}")

    def _is_in_drop_zone(self, pos: np.ndarray) -> bool:
        """Returns True if position falls inside the designated placement drop zone."""
        return is_in_drop_zone(pos, x_max=self.drop_zone_x_max, y_max=self.drop_zone_y_max)

    def _ransac_plane_segmentation(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        return ransac_plane_segmentation(
            points,
            max_iters=self.ransac_max_iterations,
            thresh=self.ransac_distance_threshold
        )

    def _adaptive_geometric_reduction(self, points: np.ndarray, target_k: int = 1024) -> np.ndarray:
        return adaptive_geometric_reduction(
            points,
            voxel_size=self.voxel_size,
            target_k=target_k
        )

    def _enforce_virtual_safety_floor(self, grasps: list, plane_model: Tuple[float, float, float, float]) -> list:
        return enforce_virtual_safety_floor(grasps, plane_model, safety_margin=0.005)

    def _pc_callback(self, msg: PointCloud2):
        self.latest_pc = msg

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        x_off = y_off = z_off = None
        for field in msg.fields:
            if field.name == "x": x_off = field.offset
            elif field.name == "y": y_off = field.offset
            elif field.name == "z": z_off = field.offset
        if x_off is None or y_off is None or z_off is None:
            return np.array([]).reshape(0, 3)
        if len(msg.data) == 0:
            return np.array([]).reshape(0, 3)
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            raw_sampled = raw[::2]
            x = raw_sampled[:, x_off:x_off+4].copy().view(np.float32).reshape(-1)
            y = raw_sampled[:, y_off:y_off+4].copy().view(np.float32).reshape(-1)
            z = raw_sampled[:, z_off:z_off+4].copy().view(np.float32).reshape(-1)
            valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isinf(x) | np.isinf(y) | np.isinf(z))
            return np.column_stack([x[valid], y[valid], z[valid]])
        except Exception as e:
            self.get_logger().warn(f"Fast point cloud unpacking failed: {e}")
            return np.array([]).reshape(0, 3)

    def _transform_points_to_base(self, points: np.ndarray, source_frame: str = None) -> np.ndarray:
        frame = source_frame if source_frame else self.camera_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return points
        t = transform.transform.translation
        q = transform.transform.rotation
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])
        return (R @ points.T).T + trans

    def _quat_to_rot(self, x, y, z, w) -> np.ndarray:
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ])

    def _filter_workspace(self, points: np.ndarray) -> np.ndarray:
        return filter_workspace(points, self.workspace_bounds)

    def _compute_grasp_yaw_pca(self, cluster_points: np.ndarray) -> float:
        return compute_grasp_yaw_pca(cluster_points)
        
    def _cluster_points(self, points: np.ndarray, radius: float, min_pts: int) -> List[np.ndarray]:
        """Cluster points using KDTree (if available) or simple fallback."""
        clusters = []
        if KDTree is None:
            return clusters
        try:
            tree = KDTree(points)
            visited = np.zeros(len(points), dtype=bool)
            for i in range(len(points)):
                if visited[i]:
                    continue
                neighbors = tree.query_ball_point(points[i], r=radius)
                if len(neighbors) < min_pts:
                    continue
                cluster_indices = set(neighbors)
                queue = list(neighbors)
                while queue:
                    curr = queue.pop()
                    if visited[curr]:
                        continue
                    visited[curr] = True
                    cluster_indices.add(curr)
                    sub_nbrs = tree.query_ball_point(points[curr], r=radius)
                    if len(sub_nbrs) >= min_pts // 2:
                        for n in sub_nbrs:
                            if not visited[n] and n not in cluster_indices:
                                cluster_indices.add(n)
                                queue.append(n)
                cluster_pts = points[list(cluster_indices)]
                if len(cluster_pts) >= min_pts:
                    clusters.append(cluster_pts)
        except Exception as e:
            self.get_logger().warn(f"Clustering error: {e}")
        return clusters

    def _heuristic_grasp_detection(self, points: np.ndarray) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if points.shape[0] < self.min_points:
            return []
            
        cluster_radius = 0.020 if self.test_scenario == "flat_object" else 0.025
        req_min_pts = 20 if self.test_scenario == "flat_object" else 15
        
        clusters = self._cluster_points(points, cluster_radius, req_min_pts)
        
        grasps = []
        if clusters:
            for cl in clusters:
                c = np.mean(cl, axis=0)
                if self._is_in_drop_zone(c):
                    continue
                conf = min(0.95, 0.5 + 0.5 * (len(cl) / 200.0))
                yaw = self._compute_grasp_yaw_pca(cl)
                col0 = np.array([0.0, 0.0, -1.0])
                col1 = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                col2 = np.cross(col0, col1)
                rot = np.column_stack([col0, col1, col2])
                grasps.append((c, conf, rot, 0.04, 0.04, yaw))

            if grasps:
                if self.test_scenario == "transparent_bottle":
                    grasps.sort(key=lambda g: 0 if g[0][1] < -0.03 else 1)
                else:
                    grasps.sort(key=lambda g: -g[0][2])
                return grasps

        centroid = np.mean(points, axis=0)
        if self._is_in_drop_zone(centroid):
            return []
        grasp_pos = np.array([centroid[0], centroid[1], centroid[2]])
        yaw = self._compute_grasp_yaw_pca(points)
        col0 = np.array([0.0, 0.0, -1.0])
        col1 = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        col2 = np.cross(col0, col1)
        rot = np.column_stack([col0, col1, col2])
        return [(grasp_pos, 0.8, rot, 0.04, 0.04, yaw)]

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        x, y, z, w = yaw_to_quaternion_tuple(yaw)
        return Quaternion(x=x, y=y, z=z, w=w)

    def _send_ipc_request(self, req: dict) -> Optional[dict]:
        """Send JSON payload to IPC socket with retry backoff."""
        if not os.path.exists(self.socket_path):
            return None

        max_retries = 3
        for attempt in range(max_retries):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(3.0)
            try:
                client.connect(self.socket_path)
                payload_str = json.dumps(encode_json(req))
                payload_bytes = payload_str.encode('utf-8')
                client.sendall(struct.pack("!I", len(payload_bytes)) + payload_bytes)

                raw_len = client.recv(4)
                if not raw_len:
                    return None
                msg_len = struct.unpack("!I", raw_len)[0]

                data = bytearray()
                while len(data) < msg_len:
                    packet = client.recv(min(65536, msg_len - len(data)))
                    if not packet:
                        break
                    data.extend(packet)

                resp = decode_json(json.loads(data.decode('utf-8')))
                return resp
            except (socket.error, socket.timeout) as e:
                self.get_logger().warn(f"IPC connection attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (2 ** attempt)) # Exponential backoff
            finally:
                client.close()
        return None

    def _check_anygrasp_health(self) -> bool:
        """Check if AnyGrasp IPC service is responding."""
        resp = self._send_ipc_request({"command": "health_check"})
        return resp is not None and resp.get("status") == "ok"

    def _call_anygrasp_service(self, points: np.ndarray) -> List[dict]:
        req = {
            "command": "infer",
            "points": points,
            "optional_params": {
                "dense_grasp": False,
                "collision_detection": False,
                "approach_steering": [0, 0, -1],
                "approach_thresh": 0.4,
            }
        }
        resp = self._send_ipc_request(req)
        if resp and resp.get("status") == "ok":
            return resp.get("grasps", [])
        return None

    def _create_pointcloud2(self, points: np.ndarray, frame_id: str) -> PointCloud2:
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

    def _anygrasp_detection(self, points: np.ndarray) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        grasps_data = self._call_anygrasp_service(points)
        if grasps_data is None or len(grasps_data) == 0:
            self.get_logger().warn("AnyGrasp service unavailable or returned 0 grasps; using heuristic fallback")
            return self._heuristic_grasp_detection(points)

        clusters = self._cluster_points(points, 0.025, 15)

        results = []
        for g in grasps_data:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            rot = np.array(g["rotation"], dtype=np.float32)
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))
            yaw = math.atan2(rot[1, 1], rot[0, 1])

            if self._is_in_drop_zone(pos):
                continue

            if clusters:
                centroids = [np.mean(cluster, axis=0) for cluster in clusters]
                dists = [np.linalg.norm(pos[:2] - c[:2]) for c in centroids]
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < 0.05:
                    cl = clusters[best_idx]
                    c = centroids[best_idx]
                    pos = np.array([c[0], c[1], c[2]], dtype=np.float32)

                    if cl.shape[0] >= 10:
                        cl_xy = cl[:, :2] - c[:2]
                        cov = np.cov(cl_xy.T)
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        if eigvals[0] > 1e-6 and (eigvals[1] / eigvals[0]) > 1.3:
                            minor_axis = eigvecs[:, 0]
                            pca_yaw = math.atan2(minor_axis[1], minor_axis[0])
                            diff1 = abs(math.atan2(math.sin(yaw - pca_yaw), math.cos(yaw - pca_yaw)))
                            diff2 = abs(math.atan2(math.sin(yaw - (pca_yaw + math.pi)), math.cos(yaw - (pca_yaw + math.pi))))
                            yaw = pca_yaw if diff1 <= diff2 else (pca_yaw + math.pi)

            results.append((pos, score, rot, width, depth, yaw))

        if not results:
            return self._heuristic_grasp_detection(points)

        if self.test_scenario == "transparent_bottle":
            results.sort(key=lambda x: 0 if x[0][1] < -0.03 else 1)
        elif self.test_scenario != "default":
            pass
        else:
            results.sort(key=lambda x: (round(float(x[0][2]), 2), x[1]), reverse=True)
        return results

    def _hybrid_grasp_detection(self, points: np.ndarray) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        """
        Hybrid Grasp Synthesis: Fuses AnyGrasp Deep Learning affordances with Selected Geometric Refiner.
        - Deep Learning Stream: predicts 6D contact normals, multi-view grasp poses, and grasp width.
        - Geometric Stream: provides centroid grounding, table collision avoidance, and selected geometric filter (PCA/OBB/Normals/Slice/RANSAC).
        """
        clusters = self._cluster_points(points, 0.025, 15)

        # 1. Obtain baseline geometric proposals using configured refiner
        geo_grasps = self.refiner.generate_heuristic_grasps(points, clusters=clusters, table_z=0.225)
        geo_grasps = [g for g in geo_grasps if not self._is_in_drop_zone(g[0])]

        # 2. Query AnyGrasp Deep Learning service
        ai_grasps_raw = self._call_anygrasp_service(points)

        if ai_grasps_raw is None or len(ai_grasps_raw) == 0:
            self.get_logger().info(f"ℹ️ [Hybrid] AnyGrasp service idle or 0 grasps -> using {self.refiner.name} proposals.")
            return geo_grasps

        # 3. Refine AnyGrasp candidates using the configured geometric refiner
        fused_grasps = self.refiner.refine(ai_grasps_raw, points, clusters=clusters, table_z=0.225)
        fused_grasps = [g for g in fused_grasps if not self._is_in_drop_zone(g[0])]

        # 4. Include distinct geometric candidates that AI might have missed
        combined = list(fused_grasps)
        for gg in geo_grasps:
            g_pos = gg[0]
            if not self._is_in_drop_zone(g_pos) and not any(np.linalg.norm(g_pos[:2] - fg[0][:2]) < 0.035 for fg in fused_grasps):
                combined.append((gg[0], min(0.75, gg[1]), gg[2], gg[3], gg[4], gg[5]))

        if self.test_scenario == "transparent_bottle":
            combined.sort(key=lambda x: 0 if x[0][1] < -0.03 else 1)
        elif self.test_scenario != "default":
            pass
        else:
            combined.sort(key=lambda x: (round(float(x[0][2]), 2), x[1]), reverse=True)

        return combined

    def _apply_scenario_pointcloud_effects(self, points: np.ndarray) -> np.ndarray:
        if self.test_scenario == "transparent_bottle":
            in_bottle_x = (points[:, 0] >= 0.24) & (points[:, 0] <= 0.32)
            in_bottle_y = (points[:, 1] >= -0.05) & (points[:, 1] <= 0.05)
            in_bottle_z = (points[:, 2] >= 0.252) & (points[:, 2] <= 0.360)
            bottle_mask = in_bottle_x & in_bottle_y & in_bottle_z
            filtered_points = points.copy()
            bottle_indices = np.where(bottle_mask)[0]
            if len(bottle_indices) > 0:
                np.random.seed(42)
                drop_mask = np.random.rand(len(bottle_indices)) < 0.95
                drop_indices = bottle_indices[drop_mask]
                filtered_points = np.delete(filtered_points, drop_indices, axis=0)

            n_ghost = 100
            np.random.seed(42)
            ghost_x = np.random.normal(0.280, 0.004, n_ghost)
            ghost_y = np.random.normal(-0.065, 0.004, n_ghost)
            ghost_z = np.random.normal(0.275, 0.004, n_ghost)
            ghost_pts = np.column_stack([ghost_x, ghost_y, ghost_z])
            return np.vstack([filtered_points, ghost_pts])

        elif self.test_scenario == "flat_object":
            n_disc = 120
            np.random.seed(42)
            theta = np.random.uniform(0, 2 * np.pi, n_disc)
            r = np.sqrt(np.random.uniform(0, 0.026**2, n_disc))
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            dz = np.random.uniform(0.2255, 0.2275, n_disc)
            disc_pts = np.column_stack([0.28 + dx, 0.0 + dy, dz])
            return disc_pts
        elif self.test_scenario == "dense_clutter":
            return points
        return points

    def _diagnose_grasp(self, pos: np.ndarray, rot: np.ndarray, width: float, depth: float, score: float) -> dict:
        u_x = rot[:, 0]
        norm_x = np.linalg.norm(u_x)
        u_x = u_x / norm_x if norm_x > 1e-4 else np.array([0.0, 0.0, -1.0])
        d = max(0.02, min(depth, 0.05))
        finger_tip_z = pos[2] + d * u_x[2] if u_x[2] < 0 else pos[2] - 0.02
        table_surface_z = 0.225

        if self.test_scenario == "transparent_bottle":
            dist_to_bottle = np.hypot(pos[0] - 0.28, pos[1] - 0.0)
            if dist_to_bottle > 0.035 and pos[1] < -0.03:
                return {"status": "FAIL_GHOST_GRASP", "tag": "⚠️ [FAIL: GHOST GRASP]", "desc": "Ảo giác điểm ma do khúc xạ quang học (Refraction)", "is_failure": True, "color": (1.0, 0.0, 0.8, 0.95)}

        if self.test_scenario == "flat_object":
            if pos[2] <= table_surface_z + 0.010:
                return {"status": "FAIL_FLAT_OBJECT", "tag": "⛔ [FAIL: FLAT OBJECT]", "desc": "Vật quá dẹt (2.5mm), thiếu khe hở luồn ngón kẹp (No Clearance)", "is_failure": True, "color": (1.0, 0.5, 0.0, 0.95)}

        if finger_tip_z < (table_surface_z - 0.002) or pos[2] < (table_surface_z + 0.004):
            penetration_mm = max(0.0, (table_surface_z - finger_tip_z) * 1000.0)
            return {"status": "FAIL_TABLE_COLLISION", "tag": "💥 [FAIL: TABLE COLLISION]", "desc": f"Ngón kẹp đâm sâu xuống bàn ({penetration_mm:.1f}mm < 225mm)", "is_failure": True, "color": (1.0, 0.1, 0.1, 0.95)}

        if self.test_scenario == "dense_clutter":
            if abs(pos[1]) < 0.03 and width > 0.035:
                return {"status": "FAIL_CLUTTER_COLLISION", "tag": "⚡ [FAIL: CLUTTER COLLISION]", "desc": "Ngón kẹp va chạm vật lân cận (Semantic Blindness)", "is_failure": True, "color": (1.0, 0.2, 0.0, 0.95)}

        return {"status": "OPTIMAL", "tag": "✅ [OPTIMAL]", "desc": "Điểm gắp an toàn hợp lệ", "is_failure": False, "color": (0.1, 0.9, 0.2, 0.95)}

    def _detect_callback(self):
        if self.latest_pc is None:
            return

        t_start = time.perf_counter()
        points_cam = self._pointcloud2_to_xyz(self.latest_pc)
        if points_cam.shape[0] == 0: return

        source_frame = self.latest_pc.header.frame_id if self.latest_pc.header.frame_id else self.camera_frame
        points_base = self._transform_points_to_base(points_cam, source_frame=source_frame)
        points_ws = self._filter_workspace(points_base)
        points_ws = self._apply_scenario_pointcloud_effects(points_ws)

        raw_pts_count = points_ws.shape[0]
        if raw_pts_count < self.min_points:
            return

        self.ws_pc_pub.publish(self._create_pointcloud2(points_ws, self.base_frame))
        table_plane = None

        if self.enable_ransac:
            t_ransac_start = time.perf_counter()
            points_object, points_table, table_plane = self._ransac_plane_segmentation(points_ws)
            t_ransac_ms = (time.perf_counter() - t_ransac_start) * 1000.0
            if points_table.shape[0] > 0: self.table_pc_pub.publish(self._create_pointcloud2(points_table, self.base_frame))
            if points_object.shape[0] > 0: self.object_pc_pub.publish(self._create_pointcloud2(points_object, self.base_frame))
            t_reduct_start = time.perf_counter()
            points_input = self._adaptive_geometric_reduction(points_object, target_k=self.target_point_count)
            t_reduct_ms = (time.perf_counter() - t_reduct_start) * 1000.0
        else:
            t_ransac_ms = 0.0
            t_reduct_ms = 0.0
            if points_ws.shape[0] > 3000:
                v_coords = np.floor(points_ws / 0.007).astype(np.int32)
                _, u_idx = np.unique(v_coords, axis=0, return_index=True)
                points_input = points_ws[u_idx]
            else:
                points_input = points_ws

        filtered_pts_count = points_input.shape[0]
        reduction_pct = max(0.0, (1.0 - filtered_pts_count / float(raw_pts_count)) * 100.0) if raw_pts_count > 0 else 0.0

        t_infer_start = time.perf_counter()
        if self.grasp_mode == "hybrid":
            grasps = self._hybrid_grasp_detection(points_input)
        elif self.use_anygrasp or self.grasp_mode == "anygrasp":
            grasps = self._anygrasp_detection(points_input)
        else:
            grasps = self._heuristic_grasp_detection(points_input)
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        if not grasps: return

        if self.enable_ransac and table_plane is not None:
            grasps = self._enforce_virtual_safety_floor(grasps, table_plane)

        diagnosed_grasps = []
        table_collision_flag = False
        for g in grasps:
            pos, score, rot, width, depth, yaw = g
            diag = self._diagnose_grasp(pos, rot, width, depth, score)
            if diag["status"] == "FAIL_TABLE_COLLISION": table_collision_flag = True
            diagnosed_grasps.append((g, diag))

        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0
        best_yaw_deg = math.degrees(grasps[0][5])

        self._record_benchmark_row(
            raw_pts_count, filtered_pts_count, reduction_pct, t_ransac_ms, t_reduct_ms,
            t_infer_ms, t_total_ms, fps, len(grasps), float(grasps[0][1]), best_yaw_deg, table_collision_flag
        )

        if self.test_scenario != "default": best_candidate = grasps[0]
        else:
            valid_grasps = [item for (item, diag) in diagnosed_grasps if not diag["is_failure"]]
            best_candidate = valid_grasps[0] if valid_grasps else grasps[0]

        best_pos = best_candidate[0]
        best_yaw = best_candidate[5]

        grasp_msg = PoseStamped()
        grasp_msg.header.stamp = self.get_clock().now().to_msg()
        grasp_msg.header.frame_id = self.base_frame
        grasp_msg.pose.position = Point(x=float(best_pos[0]), y=float(best_pos[1]), z=float(best_pos[2]))
        
        if self.dof >= 4:
            grasp_msg.pose.orientation = self._yaw_to_quaternion(best_yaw)
        else:
            # 3 DOF can't control yaw
            grasp_msg.pose.orientation = self._yaw_to_quaternion(0.0)

        self.grasp_poses_pub.publish(grasp_msg)
        self._publish_markers(diagnosed_grasps)

        if self.grasp_mode == "hybrid":
            method_str = "Hybrid (AnyGrasp AI + Geometric Ensemble)"
        elif self.use_anygrasp or self.grasp_mode == "anygrasp":
            method_str = "AnyGrasp AI (Deep Learning)"
        else:
            method_str = "Heuristic (RANSAC + PCA)"
        pipe_str = "RANSAC+Reduction" if self.enable_ransac else "Raw Cloud"
        print(f"\n🔬 [GRASP BENCHMARK] Pipeline: {pipe_str} | Engine: {method_str} | DOF: {self.dof}")
        print(f" 📊 Telemetry: Raw={raw_pts_count} -> Processed={filtered_pts_count} | FPS: {fps:.1f}")
        for idx, (g, diag) in enumerate(diagnosed_grasps[:5]):
            print(f" #{idx+1} Score={g[1]:.2f} Yaw={math.degrees(g[5]):.1f}° | {diag['tag']} {diag['desc']}")

    def _publish_markers(self, diagnosed_grasps: list):
        marker_array = MarkerArray()
        for i, (item, diag) in enumerate(diagnosed_grasps):
            pos = item[0]
            conf = item[1]
            rot = item[2]
            width = item[3] if len(item) > 3 else 0.04
            depth = item[4] if len(item) > 4 else 0.04
            yaw = item[5] if len(item) > 5 else 0.0

            cr, cg, cb, ca = diag["color"]
            color = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=float(ca))

            u_x = rot[:, 0]
            u_y = rot[:, 1]
            norm_x = np.linalg.norm(u_x)
            u_x = u_x / norm_x if norm_x > 1e-4 else np.array([0.0, 0.0, -1.0])
            norm_y = np.linalg.norm(u_y)
            u_y = u_y / norm_y if norm_y > 1e-4 else np.array([0.0, 1.0, 0.0])

            w2 = max(0.015, min(width, 0.065)) / 2.0
            d = max(0.02, min(depth, 0.05))

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
            jaw_marker.scale.x = 0.003
            jaw_marker.color = color
            jaw_marker.lifetime = Duration(sec=2, nanosec=0)

            def to_pt(arr): return Point(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
            jaw_marker.points.extend([to_pt(l_base), to_pt(r_base)])
            jaw_marker.points.extend([to_pt(l_base), to_pt(l_tip)])
            jaw_marker.points.extend([to_pt(r_base), to_pt(r_tip)])
            jaw_marker.points.extend([to_pt(b_center), to_pt(stem)])
            marker_array.markers.append(jaw_marker)

            text_marker = Marker()
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.header.frame_id = self.base_frame
            text_marker.ns = "grasp_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2] + 0.035))
            text_marker.scale.z = 0.013
            yaw_deg = math.degrees(yaw)
            if diag["is_failure"]:
                text_marker.color = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=1.0)
                text_marker.text = f"#{i+1}: {diag['tag']}"
            else:
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.2 if i == 0 else 0.8, a=1.0)
                text_marker.text = f"#{i+1}: S={conf:.3f} W={width*100:.1f}cm Y={yaw_deg:.0f}° {diag['tag']}"
            text_marker.lifetime = Duration(sec=2, nanosec=0)
            marker_array.markers.append(text_marker)

            arrow_marker = Marker()
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.header.frame_id = self.base_frame
            arrow_marker.ns = "approach_arrows"
            arrow_marker.id = i
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.points = [
                Point(x=float(pos[0] - 0.05 * u_x[0]), y=float(pos[1] - 0.05 * u_x[1]), z=float(pos[2] - 0.05 * u_x[2])),
                Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            ]
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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
