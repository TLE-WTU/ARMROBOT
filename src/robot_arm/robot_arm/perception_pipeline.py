#!/usr/bin/env python3
"""
Pure Python Perception Pipeline for 3D Robotic Grasping.
Decoupled from ROS 2 Node logic for maximum testability, modularity, and reusability.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


def filter_workspace(points: np.ndarray, bounds: Dict[str, List[float]]) -> np.ndarray:
    """Filters 3D points inside bounding box workspace."""
    if points.shape[0] == 0:
        return points
    mask = (
        (points[:, 0] >= bounds["x"][0]) & (points[:, 0] <= bounds["x"][1]) &
        (points[:, 1] >= bounds["y"][0]) & (points[:, 1] <= bounds["y"][1]) &
        (points[:, 2] >= bounds["z"][0]) & (points[:, 2] <= bounds["z"][1])
    )
    return points[mask]


def compute_grasp_yaw_pca(cluster_points: np.ndarray) -> float:
    """Computes grasp yaw angle from PCA minor axis of horizontal projection."""
    if cluster_points.shape[0] < 5:
        return 0.0
    xy = cluster_points[:, :2]
    centroid_xy = np.mean(xy, axis=0)
    centered = xy - centroid_xy
    cov = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)
    minor_axis = eigenvectors[:, 0]
    return float(math.atan2(minor_axis[1], minor_axis[0]))


def yaw_to_quaternion_tuple(yaw: float) -> Tuple[float, float, float, float]:
    """
    Converts 2D planar yaw angle (rotation around Z axis) to standard ROS quaternion (x, y, z, w).
    Formula: q = [0, 0, sin(yaw / 2), cos(yaw / 2)]
    """
    half_yaw = yaw / 2.0
    return (0.0, 0.0, float(math.sin(half_yaw)), float(math.cos(half_yaw)))


def ransac_plane_segmentation(
    points: np.ndarray,
    max_iters: int = 150,
    thresh: float = 0.008,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """
    Fast RANSAC tabletop plane segmentation with early stopping and decimation.
    Returns: (points_object, points_table, plane_equation_abcd)
    """
    if points.shape[0] < 50:
        return points, np.empty((0, 3)), (0.0, 0.0, 1.0, -0.25)

    n_pts = points.shape[0]
    best_plane = (0.0, 0.0, 1.0, -0.25)
    max_inlier_count = 0
    rng = np.random.default_rng(seed)

    # Decimate subset for hypothesis testing if point cloud is dense (> 4096 pts)
    if n_pts > 4096:
        step_fit = n_pts // 4096
        fit_pts = points[::step_fit]
    else:
        fit_pts = points
    n_fit = fit_pts.shape[0]

    for _ in range(max_iters):
        idx = rng.choice(n_fit, size=3, replace=False)
        p1, p2, p3 = fit_pts[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        # Ensure plane normal is predominantly vertical (tabletop)
        if abs(normal[2]) < 0.80:
            continue
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, p1))
        distances = np.abs(np.dot(fit_pts, normal) + d)
        inliers_mask = distances < thresh
        inlier_count = int(np.count_nonzero(inliers_mask))
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_plane = (float(normal[0]), float(normal[1]), float(normal[2]), d)
            # Early stopping: if > 70% of points match tabletop plane
            if inlier_count >= int(0.70 * n_fit) and abs(normal[2]) >= 0.90:
                break

    # Vectorized segmentation on full point cloud
    signed_dist = np.dot(points, np.array(best_plane[:3])) + best_plane[3]
    table_mask = np.abs(signed_dist) < thresh
    object_mask = signed_dist >= thresh
    if np.count_nonzero(table_mask) >= 50:
        return points[object_mask], points[table_mask], best_plane

    return points, np.empty((0, 3)), best_plane


def adaptive_geometric_reduction(
    points: np.ndarray,
    voxel_size: float = 0.005,
    target_k: int = 1024,
) -> np.ndarray:
    """Downsamples point cloud using voxel grid + uniform striding to exactly target_k points."""
    if points.shape[0] <= target_k:
        return points

    # Fast pre-stride if cloud is massive (> 10k points) to prevent expensive lexsort
    if points.shape[0] > 10000:
        step_pre = points.shape[0] // 8192
        pts_work = points[::step_pre]
    else:
        pts_work = points

    voxel_coords = np.floor(pts_work / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    downsampled = pts_work[unique_indices]
    if downsampled.shape[0] <= target_k:
        return downsampled

    step = len(downsampled) / float(target_k)
    selected_indices = [int(i * step) for i in range(target_k)]
    return downsampled[selected_indices]


def enforce_virtual_safety_floor(
    grasps: list,
    plane_model: Tuple[float, float, float, float],
    safety_margin: float = 0.005,
) -> list:
    """Adjusts grasp Z elevation so gripper finger tips never collide with the tabletop plane."""
    safe_grasps = []
    normal = np.array(plane_model[:3])
    d = plane_model[3]

    for g in grasps:
        pos = g[0]
        rot = g[2]
        depth = g[4] if len(g) > 4 else 0.04
        u_x = rot[:, 0]
        finger_tip_offset = depth if (u_x[2] < -0.5) else 0.020
        tip_pos = np.array([pos[0], pos[1], pos[2] - finger_tip_offset])
        dist_to_plane = float(np.dot(tip_pos, normal) + d)
        if dist_to_plane < safety_margin:
            pos_adjusted = pos.copy()
            pos_adjusted[2] += safety_margin - dist_to_plane
            safe_grasps.append((pos_adjusted, g[1] * 0.95, g[2], g[3], g[4], g[5]))
        else:
            safe_grasps.append(g)
    return safe_grasps


def is_in_drop_zone(pos: np.ndarray, x_max: float = 0.24, y_max: float = -0.12) -> bool:
    """Returns True if position falls inside the designated placement drop zone."""
    return bool(pos[0] <= x_max and pos[1] <= y_max)

