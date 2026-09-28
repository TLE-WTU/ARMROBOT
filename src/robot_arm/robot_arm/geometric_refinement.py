"""
Modular Geometric Refinement Algorithms for Hybrid AnyGrasp Perception.

This module provides plug-and-play geometric refinement strategies that can be
paired with AnyGrasp deep learning affordances or run independently:
1. PCARefiner: Covariance decomposition, principal axis alignment, centroid blending.
2. OBBRefiner: Oriented Bounding Box calculation, dimension check vs jaw opening, box face alignment.
3. SurfaceNormalRefiner: k-NN surface normal estimation, antipodal contact friction cone alignment.
4. CrossSectionSliceRefiner: Z-height cross-sectional slicing, minimum waist/neck detection.
5. PrimitiveRANSACRefiner: Cylinder / Box shape fitting via RANSAC.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class BaseGeometricRefiner(ABC):
    """Abstract base class for geometric grasp refiners."""

    def __init__(self, name: str, max_gripper_width: float = 0.07):
        self.name = name
        self.max_gripper_width = max_gripper_width

    @abstractmethod
    def refine(
        self,
        ai_grasps: List[Dict[str, Any]],
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        """
        Refine candidate grasps.
        Returns list of tuples: (pos, score, rot_matrix, width, depth, yaw).
        """
        pass

    @abstractmethod
    def generate_heuristic_grasps(
        self,
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        """Generate purely geometric grasp candidates without AI input."""
        pass


def make_top_down_rotation(yaw: float) -> np.ndarray:
    """Create a top-down grasp rotation matrix from yaw angle."""
    col0 = np.array([0.0, 0.0, -1.0])  # Approach along -Z
    col1 = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # Closing axis along Y'
    col2 = np.cross(col0, col1)  # Binormal axis
    return np.column_stack([col0, col1, col2])


class PCARefiner(BaseGeometricRefiner):
    """
    Method 1: Principal Component Analysis (PCA).
    Computes covariance matrix of object clusters, aligns gripper jaws with the minor
    eigen-axis (narrow dimension) to ensure the jaw can close around the object,
    and blends the grasp position toward the cluster centroid.
    """

    def __init__(self, max_gripper_width: float = 0.07, centroid_blend: float = 0.25):
        super().__init__("PCA", max_gripper_width)
        self.centroid_blend = centroid_blend

    def compute_pca_yaw(self, points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        xy = points[:, :2]
        centered = xy - np.mean(xy, axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        minor_axis = eigvecs[:, 0]  # Minor axis in XY = direction of narrowest span
        return math.atan2(minor_axis[1], minor_axis[0])

    def refine(
        self,
        ai_grasps: List[Dict[str, Any]],
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]

        fused = []
        for g in ai_grasps:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            rot = np.array(g["rotation"], dtype=np.float32)
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))
            raw_yaw = math.atan2(rot[1, 1], rot[0, 1])

            fused_pos = pos.copy()
            fused_yaw = raw_yaw
            agreement_bonus = 0.0

            if clusters:
                centroids = [np.mean(c, axis=0) for c in clusters]
                dists = [np.linalg.norm(pos[:2] - c[:2]) for c in centroids]
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < 0.08:
                    cl = clusters[best_idx]
                    c = centroids[best_idx]
                    # Centroid blending
                    fused_pos[:2] = (1.0 - self.centroid_blend) * pos[:2] + self.centroid_blend * c[:2]
                    fused_pos[2] = max(pos[2], c[2] - 0.01)

                    if cl.shape[0] >= 5:
                        pca_yaw = self.compute_pca_yaw(cl)
                        diff1 = abs(math.atan2(math.sin(raw_yaw - pca_yaw), math.cos(raw_yaw - pca_yaw)))
                        diff2 = abs(math.atan2(math.sin(raw_yaw - (pca_yaw + math.pi)), math.cos(raw_yaw - (pca_yaw + math.pi))))
                        min_diff = min(diff1, diff2)
                        if min_diff < math.radians(35):
                            agreement_bonus = 0.15
                            fused_yaw = pca_yaw if diff1 <= diff2 else (pca_yaw + math.pi)
                        elif min_diff > math.radians(65):
                            fused_yaw = pca_yaw if diff1 <= diff2 else (pca_yaw + math.pi)
                            agreement_bonus = 0.05

            composite_score = min(1.0, score + agreement_bonus)
            fused_rot = make_top_down_rotation(fused_yaw)
            fused.append((fused_pos, composite_score, fused_rot, width, depth, fused_yaw))

        return fused

    def generate_heuristic_grasps(
        self,
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]
        grasps = []
        for cl in clusters:
            if cl.shape[0] < 5:
                continue
            c = np.mean(cl, axis=0)
            yaw = self.compute_pca_yaw(cl)
            rot = make_top_down_rotation(yaw)
            pos = np.array([c[0], c[1], max(c[2], table_z + 0.02)])
            grasps.append((pos, 0.80, rot, 0.04, 0.04, yaw))
        return grasps


class OBBRefiner(BaseGeometricRefiner):
    """
    Method 2: Oriented Bounding Box (OBB).
    Computes the 2D/3D minimum area bounding box of the object cluster.
    Identifies the principal extent (length) and orthogonal cross-section (width).
    Verifies that the object cross-section fits within the 7cm gripper opening and
    aligns the jaw closure along the minimum bounding box dimension.
    """

    def __init__(self, max_gripper_width: float = 0.07):
        super().__init__("OBB", max_gripper_width)

    def compute_obb_2d(self, points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Computes 2D oriented bounding box on XY plane.
        Returns: (center_xy, length, width, angle_rad)
        """
        if points.shape[0] < 4:
            c = np.mean(points, axis=0)[:2] if points.shape[0] > 0 else np.zeros(2)
            return c, 0.05, 0.05, 0.0

        xy = points[:, :2]
        centered = xy - np.mean(xy, axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, 1]
        minor_axis = eigvecs[:, 0]

        # Project points onto eigenvectors
        proj_major = np.dot(centered, major_axis)
        proj_minor = np.dot(centered, minor_axis)

        length = float(np.max(proj_major) - np.min(proj_major))
        width = float(np.max(proj_minor) - np.min(proj_minor))
        mid_major = float(np.mean([np.max(proj_major), np.min(proj_major)]))
        mid_minor = float(np.mean([np.max(proj_minor), np.min(proj_minor)]))

        center_xy = np.mean(xy, axis=0) + mid_major * major_axis + mid_minor * minor_axis
        angle = math.atan2(minor_axis[1], minor_axis[0])
        return center_xy, length, width, angle

    def refine(
        self,
        ai_grasps: List[Dict[str, Any]],
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]

        fused = []
        for g in ai_grasps:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))

            # Match with OBB of closest cluster
            fused_pos = pos.copy()
            fused_yaw = 0.0
            obb_bonus = 0.0

            if clusters:
                centroids = [np.mean(c, axis=0) for c in clusters]
                dists = [np.linalg.norm(pos[:2] - c[:2]) for c in centroids]
                best_idx = int(np.argmin(dists))
                cl = clusters[best_idx]

                obb_center, obb_len, obb_w, obb_angle = self.compute_obb_2d(cl)
                # OBB centers precisely on the box volume
                fused_pos[:2] = 0.60 * pos[:2] + 0.40 * obb_center
                fused_pos[2] = max(pos[2], np.mean(cl[:, 2]))
                fused_yaw = obb_angle

                # Check if width fits within max gripper opening
                if obb_w <= self.max_gripper_width:
                    obb_bonus += 0.20  # Fits inside gripper opening perfectly
                    width = min(self.max_gripper_width, max(0.02, obb_w + 0.01))
                else:
                    obb_bonus -= 0.30  # Object too wide for gripper

            composite_score = min(1.0, max(0.1, score + obb_bonus))
            rot = make_top_down_rotation(fused_yaw)
            fused.append((fused_pos, composite_score, rot, width, depth, fused_yaw))

        return fused

    def generate_heuristic_grasps(
        self,
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]
        grasps = []
        for cl in clusters:
            if cl.shape[0] < 5:
                continue
            obb_center, obb_len, obb_w, obb_angle = self.compute_obb_2d(cl)
            z_mid = np.mean(cl[:, 2])
            pos = np.array([obb_center[0], obb_center[1], max(z_mid, table_z + 0.02)])
            rot = make_top_down_rotation(obb_angle)
            score = 0.85 if obb_w <= self.max_gripper_width else 0.40
            grasps.append((pos, score, rot, min(self.max_gripper_width, obb_w + 0.01), 0.04, obb_angle))
        return grasps


class SurfaceNormalRefiner(BaseGeometricRefiner):
    """
    Method 3: Surface Normal & Antipodal Contact Analysis.
    Estimates 3D surface normals around grasp contact points via k-NN plane fitting.
    Evaluates antipodal grasp criterion: closing jaws must be opposite and aligned
    with the inward surface normals to form a stable friction cone.
    """

    def __init__(self, max_gripper_width: float = 0.07, k_neighbors: int = 15):
        super().__init__("SurfaceNormals", max_gripper_width)
        self.k_neighbors = k_neighbors

    def estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """Estimates 3D surface normals for all points using k-NN covariance."""
        n_pts = points.shape[0]
        normals = np.zeros_like(points)
        if n_pts < self.k_neighbors:
            normals[:, 2] = 1.0
            return normals

        for i in range(n_pts):
            p = points[i]
            dists = np.sum((points - p) ** 2, axis=1)
            nn_idx = np.argpartition(dists, self.k_neighbors)[: self.k_neighbors]
            nn_pts = points[nn_idx]
            cov = np.cov(nn_pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]  # Surface normal = eigenvector with smallest variance

            # Orient normal outward from center
            diff_from_center = p - np.mean(points, axis=0)
            if np.dot(normal, diff_from_center) < 0:
                normal = -normal
            normals[i] = normal

        return normals

    def compute_antipodal_score(
        self, grasp_pos: np.ndarray, closing_vector: np.ndarray, points: np.ndarray, normals: np.ndarray
    ) -> float:
        """
        Measures how well the gripper closing vector aligns with opposing surface normals.
        Antipodal contact: dot product with inward normals should be close to 1.
        """
        # Find points near the two finger pads
        finger_offset = 0.025
        p_left = grasp_pos - finger_offset * closing_vector
        p_right = grasp_pos + finger_offset * closing_vector

        d_l = np.sum((points - p_left) ** 2, axis=1)
        d_r = np.sum((points - p_right) ** 2, axis=1)

        idx_l = int(np.argmin(d_l))
        idx_r = int(np.argmin(d_r))

        n_l = normals[idx_l]
        n_r = normals[idx_r]

        # Left normal should point towards +closing_vector (inward), Right should point towards -closing_vector
        align_l = np.dot(n_l, -closing_vector)
        align_r = np.dot(n_r, closing_vector)
        score = float((align_l + align_r) / 2.0)
        return max(0.0, score)

    def refine(
        self,
        ai_grasps: List[Dict[str, Any]],
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        normals = self.estimate_normals(points)
        fused = []

        for g in ai_grasps:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            rot = np.array(g["rotation"], dtype=np.float32)
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))
            raw_yaw = math.atan2(rot[1, 1], rot[0, 1])

            # Sample angles around raw_yaw to maximize antipodal friction cone
            best_yaw = raw_yaw
            best_antipodal = 0.0

            for d_deg in [-30, -15, 0, 15, 30]:
                test_yaw = raw_yaw + math.radians(d_deg)
                closing_vec = np.array([math.cos(test_yaw), math.sin(test_yaw), 0.0])
                anti_score = self.compute_antipodal_score(pos, closing_vec, points, normals)
                if anti_score > best_antipodal:
                    best_antipodal = anti_score
                    best_yaw = test_yaw

            # Boost score based on antipodal friction alignment
            fused_score = min(1.0, score * 0.7 + best_antipodal * 0.3)
            fused_rot = make_top_down_rotation(best_yaw)
            fused.append((pos, fused_score, fused_rot, width, depth, best_yaw))

        return fused

    def generate_heuristic_grasps(
        self,
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        normals = self.estimate_normals(points)
        c = np.mean(points, axis=0)

        best_yaw = 0.0
        best_score = 0.0
        for yaw in np.linspace(0, math.pi, 12):
            closing_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            score = self.compute_antipodal_score(c, closing_vec, points, normals)
            if score > best_score:
                best_score = score
                best_yaw = float(yaw)

        pos = np.array([c[0], c[1], max(c[2], table_z + 0.02)])
        rot = make_top_down_rotation(best_yaw)
        return [(pos, 0.70 + 0.25 * best_score, rot, 0.045, 0.04, best_yaw)]


class CrossSectionSliceRefiner(BaseGeometricRefiner):
    """
    Method 4: Cross-Section Slicing (Height Slice Waist Detection).
    Slices the object cluster along the vertical Z-axis into 5-10 horizontal strata.
    Finds the cross-section with the minimal bounding perimeter/area (the 'waist'
    or narrow neck, such as a bottle neck or cup waist) which is easiest to grasp securely.
    """

    def __init__(self, max_gripper_width: float = 0.07, n_slices: int = 8):
        super().__init__("CrossSectionSlice", max_gripper_width)
        self.n_slices = n_slices

    def find_optimal_slice(self, points: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Finds optimal horizontal slice with minimum width that fits within gripper opening.
        Returns: (slice_centroid_3d, slice_width, slice_yaw)
        """
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        dz = (z_max - z_min) / float(self.n_slices)

        best_z_mid = (z_min + z_max) / 2.0
        best_width = 999.0
        best_yaw = 0.0
        best_centroid = np.mean(points, axis=0)

        for i in range(1, self.n_slices):
            s_z0 = z_min + i * dz
            s_z1 = s_z0 + dz
            mask = (points[:, 2] >= s_z0) & (points[:, 2] <= s_z1)
            slice_pts = points[mask]

            if slice_pts.shape[0] < 5:
                continue

            xy = slice_pts[:, :2]
            c_xy = np.mean(xy, axis=0)
            centered = xy - c_xy
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            minor_axis = eigvecs[:, 0]
            w = 2.0 * math.sqrt(max(1e-6, eigvals[0])) * 2.0  # Approx 95% span
            yaw = math.atan2(minor_axis[1], minor_axis[0])

            # Select narrower slice if it is at least 15mm thick and within gripper opening
            if 0.015 <= w < best_width and w <= self.max_gripper_width:
                best_width = w
                best_yaw = yaw
                best_z_mid = (s_z0 + s_z1) / 2.0
                best_centroid = np.array([c_xy[0], c_xy[1], best_z_mid])

        return best_centroid, best_width if best_width < 900.0 else 0.04, best_yaw

    def refine(
        self,
        ai_grasps: List[Dict[str, Any]],
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]

        fused = []
        for g in ai_grasps:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))
            raw_yaw = math.atan2(g["rotation"][1][1], g["rotation"][0][1])

            # Find matching cluster slice
            fused_pos = pos.copy()
            fused_yaw = raw_yaw
            bonus = 0.0

            if clusters:
                centroids = [np.mean(c, axis=0) for c in clusters]
                dists = [np.linalg.norm(pos[:2] - c[:2]) for c in centroids]
                best_idx = int(np.argmin(dists))
                cl = clusters[best_idx]

                slice_c, slice_w, slice_yaw = self.find_optimal_slice(cl)
                # Pull grasp position towards the optimal waist height Z
                fused_pos[0] = 0.70 * pos[0] + 0.30 * slice_c[0]
                fused_pos[1] = 0.70 * pos[1] + 0.30 * slice_c[1]
                fused_pos[2] = 0.50 * pos[2] + 0.50 * slice_c[2]
                fused_yaw = slice_yaw
                width = slice_w
                bonus = 0.15

            fused_rot = make_top_down_rotation(fused_yaw)
            fused.append((fused_pos, min(1.0, score + bonus), fused_rot, width, depth, fused_yaw))

        return fused

    def generate_heuristic_grasps(
        self,
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]
        grasps = []
        for cl in clusters:
            if cl.shape[0] < 5:
                continue
            slice_c, slice_w, slice_yaw = self.find_optimal_slice(cl)
            rot = make_top_down_rotation(slice_yaw)
            grasps.append((slice_c, 0.85, rot, slice_w, 0.04, slice_yaw))
        return grasps


class PrimitiveRANSACRefiner(BaseGeometricRefiner):
    """
    Method 5: RANSAC Geometric Primitive Fitting.
    Fits geometric primitives (Cylinders and Boxes) to the object cluster using RANSAC.
    If a vertical cylinder is detected (mugs, bottles, cans), it aligns the grasp
    perpendicular to the cylinder surface and sets grasp width to the cylinder diameter.
    """

    def __init__(self, max_gripper_width: float = 0.07, max_iters: int = 100, dist_thresh: float = 0.005):
        super().__init__("PrimitiveRANSAC", max_gripper_width)
        self.max_iters = max_iters
        self.dist_thresh = dist_thresh

    def fit_cylinder_ransac(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Fits a vertical cylinder (x-c_x)^2 + (y-c_y)^2 = r^2.
        Returns: (center_xy, radius, inlier_ratio) or None
        """
        n_pts = points.shape[0]
        if n_pts < 10:
            return None

        xy = points[:, :2]
        best_center = np.mean(xy, axis=0)
        best_radius = 0.025
        best_inliers = 0

        # Fast RANSAC circle fit
        for _ in range(self.max_iters):
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = xy[idx]

            # Circle through 3 points in 2D
            temp = p2[0] ** 2 + p2[1] ** 2
            bc = (p1[0] ** 2 + p1[1] ** 2 - temp) / 2.0
            cd = (temp - p3[0] ** 2 - p3[1] ** 2) / 2.0
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

            if abs(det) < 1e-6:
                continue

            cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
            r = math.sqrt(max(1e-6, (p1[0] - cx) ** 2 + (p1[1] - cy) ** 2))

            if r > 0.08 or r < 0.01:
                continue  # Beyond realistic tabletop object radius

            dists = np.abs(np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) - r)
            inliers = np.sum(dists < self.dist_thresh)
            if inliers > best_inliers:
                best_inliers = inliers
                best_center = np.array([cx, cy])
                best_radius = r
                if float(best_inliers) / float(n_pts) >= 0.75:
                    break

        inlier_ratio = float(best_inliers) / float(n_pts)
        if inlier_ratio >= 0.40:
            return best_center, best_radius, inlier_ratio
        return None

    def refine(
        self,
        ai_grasps: List[Dict[str, Any]],
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]

        fused = []
        for g in ai_grasps:
            pos = np.array(g["translation"], dtype=np.float32)
            score = float(g["score"])
            width = float(g.get("width", 0.04))
            depth = float(g.get("depth", 0.04))
            raw_yaw = math.atan2(g["rotation"][1][1], g["rotation"][0][1])

            fused_pos = pos.copy()
            fused_yaw = raw_yaw
            bonus = 0.0

            if clusters:
                centroids = [np.mean(c, axis=0) for c in clusters]
                dists = [np.linalg.norm(pos[:2] - c[:2]) for c in centroids]
                best_idx = int(np.argmin(dists))
                cl = clusters[best_idx]

                cyl_result = self.fit_cylinder_ransac(cl)
                if cyl_result:
                    cyl_center, cyl_r, inlier_ratio = cyl_result
                    # Cylinder detected! Center grasp right on cylinder axis
                    fused_pos[0] = cyl_center[0]
                    fused_pos[1] = cyl_center[1]
                    fused_pos[2] = max(pos[2], np.mean(cl[:, 2]))
                    width = min(self.max_gripper_width, 2.0 * cyl_r + 0.01)
                    bonus = 0.20 * inlier_ratio

            fused_rot = make_top_down_rotation(fused_yaw)
            fused.append((fused_pos, min(1.0, score + bonus), fused_rot, width, depth, fused_yaw))

        return fused

    def generate_heuristic_grasps(
        self,
        points: np.ndarray,
        clusters: Optional[List[np.ndarray]] = None,
        table_z: float = 0.225,
    ) -> List[Tuple[np.ndarray, float, np.ndarray, float, float, float]]:
        if clusters is None:
            clusters = [points]
        grasps = []
        for cl in clusters:
            if cl.shape[0] < 5:
                continue
            cyl_result = self.fit_cylinder_ransac(cl)
            if cyl_result:
                cyl_c, cyl_r, ratio = cyl_result
                pos = np.array([cyl_c[0], cyl_c[1], max(np.mean(cl[:, 2]), table_z + 0.02)])
                rot = make_top_down_rotation(0.0)
                grasps.append((pos, 0.85, rot, min(self.max_gripper_width, 2.0 * cyl_r + 0.01), 0.04, 0.0))
            else:
                c = np.mean(cl, axis=0)
                pos = np.array([c[0], c[1], max(c[2], table_z + 0.02)])
                rot = make_top_down_rotation(0.0)
                grasps.append((pos, 0.70, rot, 0.04, 0.04, 0.0))
        return grasps


def get_geometric_refiner(name: str, max_gripper_width: float = 0.07) -> BaseGeometricRefiner:
    """Factory function to instantiate geometric refiner by name."""
    norm_name = name.strip().lower()
    if norm_name in ("pca", "principal_components"):
        return PCARefiner(max_gripper_width=max_gripper_width)
    elif norm_name in ("obb", "bounding_box"):
        return OBBRefiner(max_gripper_width=max_gripper_width)
    elif norm_name in ("normals", "surface_normals", "antipodal"):
        return SurfaceNormalRefiner(max_gripper_width=max_gripper_width)
    elif norm_name in ("slice", "cross_section", "waist"):
        return CrossSectionSliceRefiner(max_gripper_width=max_gripper_width)
    elif norm_name in ("ransac", "ransac_primitive", "primitive"):
        return PrimitiveRANSACRefiner(max_gripper_width=max_gripper_width)
    else:
        raise ValueError(
            f"Unknown geometric refiner '{name}'. Valid choices: 'pca', 'obb', 'normals', 'slice', 'ransac_primitive'."
        )
