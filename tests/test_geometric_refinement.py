"""
Unit tests for modular geometric refinement algorithms and benchmark dataset.
"""

import math
import numpy as np
import pytest

from robot_arm.geometric_refinement import (
    PCARefiner,
    OBBRefiner,
    SurfaceNormalRefiner,
    CrossSectionSliceRefiner,
    PrimitiveRANSACRefiner,
    get_geometric_refiner,
)
from robot_arm.benchmark_dataset import BenchmarkDataset


class TestPCARefiner:
    def test_pca_yaw_elongated_x(self):
        refiner = PCARefiner()
        rng = np.random.default_rng(42)
        # Elongated along X: minor axis is along Y (yaw ~ ±90 deg)
        pts = np.column_stack([
            rng.normal(0.25, 0.05, 100),
            rng.normal(0.00, 0.01, 100),
            rng.normal(0.26, 0.01, 100),
        ])
        yaw = refiner.compute_pca_yaw(pts)
        assert abs(abs(yaw) - math.pi / 2) < 0.25

    def test_refine_blends_centroid(self):
        refiner = PCARefiner(centroid_blend=0.30)
        pts = np.array([
            [0.25, 0.00, 0.26],
            [0.26, 0.01, 0.26],
            [0.24, -0.01, 0.26],
            [0.25, 0.02, 0.26],
            [0.25, -0.02, 0.26],
        ])
        c = np.mean(pts, axis=0)
        ai_grasps = [{
            "translation": [0.28, 0.05, 0.26],
            "score": 0.85,
            "rotation": np.eye(3).tolist(),
            "width": 0.04,
            "depth": 0.04,
        }]
        fused = refiner.refine(ai_grasps, pts, clusters=[pts])
        assert len(fused) == 1
        pos, score, rot, width, depth, yaw = fused[0]
        # X and Y should be pulled towards centroid (0.25, 0.00)
        assert pos[0] < 0.28
        assert pos[1] < 0.05


class TestOBBRefiner:
    def test_obb_box_dimensions(self):
        refiner = OBBRefiner(max_gripper_width=0.07)
        # Create rectangular point box 0.08m x 0.04m aligned with X and Y
        rng = np.random.default_rng(42)
        x = rng.uniform(-0.04, 0.04, 200)
        y = rng.uniform(-0.02, 0.02, 200)
        pts = np.column_stack([0.25 + x, 0.00 + y, np.full(200, 0.26)])

        center, length, width, angle = refiner.compute_obb_2d(pts)
        assert abs(length - 0.08) < 0.015
        assert abs(width - 0.04) < 0.015
        assert abs(center[0] - 0.25) < 0.01
        assert abs(center[1] - 0.00) < 0.01

    def test_obb_heuristic_grasps(self):
        refiner = OBBRefiner(max_gripper_width=0.07)
        pts = np.random.rand(50, 3) * 0.04 + 0.25
        grasps = refiner.generate_heuristic_grasps(pts)
        assert len(grasps) >= 1
        assert grasps[0][3] <= 0.07  # width within max gripper


class TestSurfaceNormalRefiner:
    def test_estimate_normals_planar(self):
        refiner = SurfaceNormalRefiner(k_neighbors=10)
        # Flat horizontal plane: normals should point vertical (Z close to ±1)
        rng = np.random.default_rng(42)
        x = rng.uniform(0.20, 0.30, 80)
        y = rng.uniform(-0.05, 0.05, 80)
        z = np.full(80, 0.26)
        pts = np.column_stack([x, y, z])

        normals = refiner.estimate_normals(pts)
        assert normals.shape == pts.shape
        # Average Z component should be high
        avg_abs_nz = np.mean(np.abs(normals[:, 2]))
        assert avg_abs_nz > 0.85

    def test_antipodal_score_calculation(self):
        refiner = SurfaceNormalRefiner()
        pts = np.array([
            [0.25, -0.02, 0.26],  # left point
            [0.25, 0.02, 0.26],   # right point
        ])
        normals = np.array([
            [0.0, -1.0, 0.0],     # outward normal pointing left (-Y)
            [0.0, 1.0, 0.0],      # outward normal pointing right (+Y)
        ])
        # Closing vector along Y
        closing_vec = np.array([0.0, 1.0, 0.0])
        score = refiner.compute_antipodal_score(np.array([0.25, 0.0, 0.26]), closing_vec, pts, normals)
        assert score > 0.80


class TestCrossSectionSliceRefiner:
    def test_find_narrowest_waist_on_bottle(self):
        refiner = CrossSectionSliceRefiner(n_slices=6)
        # Wide base (r=0.03, z in [0.22, 0.26]) and narrow neck (r=0.01, z in [0.26, 0.30])
        rng = np.random.default_rng(42)
        th_b = rng.uniform(0, 2 * math.pi, 200)
        base = np.column_stack([0.25 + 0.03 * np.cos(th_b), 0.03 * np.sin(th_b), rng.uniform(0.22, 0.26, 200)])

        th_n = rng.uniform(0, 2 * math.pi, 100)
        neck = np.column_stack([0.25 + 0.01 * np.cos(th_n), 0.01 * np.sin(th_n), rng.uniform(0.26, 0.30, 100)])

        pts = np.vstack([base, neck])
        centroid, width, yaw = refiner.find_optimal_slice(pts)
        # Optimal slice should be in the neck region (Z > 0.26) and width should be small
        assert centroid[2] >= 0.255
        assert width < 0.05


class TestPrimitiveRANSACRefiner:
    def test_fit_cylinder(self):
        refiner = PrimitiveRANSACRefiner(max_iters=150, dist_thresh=0.005)
        # Generate perfect circle points on XY with r=0.035
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * math.pi, 150)
        r = 0.035 + rng.normal(0, 0.001, 150)
        pts = np.column_stack([0.25 + r * np.cos(theta), 0.00 + r * np.sin(theta), rng.uniform(0.23, 0.30, 150)])

        result = refiner.fit_cylinder_ransac(pts)
        assert result is not None
        center, radius, ratio = result
        assert abs(center[0] - 0.25) < 0.01
        assert abs(center[1] - 0.00) < 0.01
        assert abs(radius - 0.035) < 0.005
        assert ratio > 0.70


class TestBenchmarkDataset:
    def test_all_objects_generate_valid_clouds(self):
        dataset = BenchmarkDataset()
        cases = dataset.get_all_test_cases()
        assert len(cases) == 21  # 7 objects x 3 conditions

        for obj_name, cond in cases:
            pts, centroid, label = dataset.load_sample(obj_name, cond)
            assert pts.shape[0] >= 50, f"Object {label} has too few points: {pts.shape[0]}"
            assert pts.shape[1] == 3
            assert len(centroid) == 3
            # With Gaussian jitter sigma=0.003, points can extend slightly down to 0.215
            assert np.min(pts[:, 2]) >= 0.215


class TestRefinerFactory:
    def test_factory_creates_all_refiners(self):
        names = ["pca", "obb", "normals", "slice", "ransac"]
        for n in names:
            ref = get_geometric_refiner(n)
            assert ref is not None
            assert isinstance(ref.name, str)

    def test_factory_invalid_name(self):
        with pytest.raises(ValueError):
            get_geometric_refiner("invalid_method_xyz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
