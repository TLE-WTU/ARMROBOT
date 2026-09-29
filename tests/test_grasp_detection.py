"""Unit tests for the decoupled perception pipeline functions."""
import math
import numpy as np
import pytest

from robot_arm.perception_pipeline import (
    adaptive_geometric_reduction,
    compute_grasp_yaw_pca,
    enforce_virtual_safety_floor,
    filter_workspace,
    ransac_plane_segmentation,
    yaw_to_quaternion_tuple,
    is_in_drop_zone,
)


class TestRANSACPlaneSegmentation:
    """Test real RANSAC tabletop plane segmentation directly."""

    def _make_table_points(self, n: int = 500, z: float = 0.250) -> np.ndarray:
        """Generate synthetic table surface points."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0.10, 0.50, n)
        y = rng.uniform(-0.20, 0.20, n)
        z_pts = np.full(n, z) + rng.normal(0, 0.001, n)
        return np.column_stack([x, y, z_pts])

    def _make_object_points(self, n: int = 100, center=(0.30, 0.0, 0.30)) -> np.ndarray:
        """Generate synthetic object points above table."""
        rng = np.random.default_rng(123)
        offsets = rng.normal(0, 0.015, (n, 3))
        return np.array(center) + offsets

    def test_separates_table_from_objects(self):
        """RANSAC should accurately separate table points from object points."""
        table = self._make_table_points(500, z=0.250)
        obj = self._make_object_points(100, center=(0.30, 0.0, 0.30))
        combined = np.vstack([table, obj])

        pts_obj, pts_table, plane = ransac_plane_segmentation(combined, max_iters=50, thresh=0.008)
        
        # Object points should be ~100 and table points ~500
        assert len(pts_obj) > 70, f"Expected >70 object points, got {len(pts_obj)}"
        assert len(pts_table) > 450, f"Expected >450 table points, got {len(pts_table)}"
        # Plane normal should point up along Z
        assert abs(plane[2]) > 0.95

    def test_empty_points(self):
        """Should handle empty point arrays gracefully."""
        points = np.empty((0, 3))
        pts_obj, pts_table, plane = ransac_plane_segmentation(points)
        assert points.shape[0] == 0
        assert pts_table.shape[0] == 0

    def test_small_pointcloud(self):
        """Should handle very small point clouds (< 50 points)."""
        points = np.random.rand(10, 3)
        pts_obj, pts_table, plane = ransac_plane_segmentation(points)
        assert pts_obj.shape[0] == 10
        assert pts_table.shape[0] == 0


class TestAdaptiveGeometricReduction:
    """Test voxel downsampling and strided reduction."""

    def test_reduction_preserves_shape(self):
        """Output should still be Nx3."""
        points = np.random.rand(5000, 3) * 0.05
        downsampled = adaptive_geometric_reduction(points, voxel_size=0.005, target_k=1024)
        assert downsampled.shape[1] == 3
        assert downsampled.shape[0] <= 1024

    def test_small_cloud_unchanged(self):
        """Clouds smaller than target should pass through unchanged."""
        points = np.random.rand(100, 3)
        result = adaptive_geometric_reduction(points, target_k=1024)
        assert np.array_equal(result, points)


class TestPCAGraspYaw:
    """Test PCA-based grasp yaw computation."""

    def test_elongated_object_yaw(self):
        """For an elongated object along X, minor axis yaw should be ~±90°."""
        rng = np.random.default_rng(42)
        x = rng.normal(0.3, 0.05, 100)
        y = rng.normal(0.0, 0.01, 100)
        z = rng.normal(0.3, 0.01, 100)
        points = np.column_stack([x, y, z])

        yaw = compute_grasp_yaw_pca(points)
        # Minor axis perpendicular to elongation along X => yaw ~ ±90°
        assert abs(abs(yaw) - math.pi / 2) < 0.3, f"Expected yaw ~±90° but got {math.degrees(yaw):.1f}°"


class TestQuaternionConversions:
    """Test correct ROS standard quaternion conversion for planar yaw."""

    def test_yaw_zero_quaternion(self):
        """Yaw=0 should produce identity quaternion (0, 0, 0, 1)."""
        x, y, z, w = yaw_to_quaternion_tuple(0.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z) < 1e-6
        assert abs(w - 1.0) < 1e-6

    def test_yaw_90_deg_quaternion(self):
        """Yaw=90° (pi/2) around Z should have z=sin(45°)=sqrt(2)/2, w=cos(45°)=sqrt(2)/2."""
        x, y, z, w = yaw_to_quaternion_tuple(math.pi / 2.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        expected = math.sqrt(2.0) / 2.0
        assert abs(z - expected) < 1e-5
        assert abs(w - expected) < 1e-5
        assert abs(x**2 + y**2 + z**2 + w**2 - 1.0) < 1e-6


class TestVirtualSafetyFloor:
    """Test that safety floor adjusts grasp Z above tabletop."""

    def test_safety_floor_elevates_low_grasp(self):
        grasps = [
            (np.array([0.25, 0.0, 0.222]), 0.9, np.eye(3), 0.04, 0.04, 0.0)
        ]
        table_plane = (0.0, 0.0, 1.0, -0.225)
        safe = enforce_virtual_safety_floor(grasps, table_plane, safety_margin=0.005)
class TestDropZoneAndWorkspace:
    """Test drop zone exclusion and workspace bounding box filtering."""

    def test_drop_zone_detection(self):
        # Drop zone: x <= 0.24 and y <= -0.12
        in_drop = np.array([0.20, -0.15, 0.25])
        out_drop_x = np.array([0.28, -0.15, 0.25])
        out_drop_y = np.array([0.20, 0.0, 0.25])
        assert is_in_drop_zone(in_drop) is True
        assert is_in_drop_zone(out_drop_x) is False
        assert is_in_drop_zone(out_drop_y) is False

    def test_filter_workspace(self):
        bounds = {
            "x": [0.10, 0.50],
            "y": [-0.20, 0.20],
            "z": [0.22, 0.45],
        }
        points = np.array([
            [0.30, 0.0, 0.30],    # inside
            [0.05, 0.0, 0.30],    # outside x min
            [0.30, 0.30, 0.30],   # outside y max
            [0.30, 0.0, 0.50],    # outside z max
        ])
        filtered = filter_workspace(points, bounds)
        assert filtered.shape[0] == 1
        assert np.allclose(filtered[0], [0.30, 0.0, 0.30])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
