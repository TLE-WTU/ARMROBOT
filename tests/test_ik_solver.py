"""Unit tests for the unified IK solver."""
import math
import pytest


class TestIKSolverImport:
    """Test that the IK solver module can be imported."""

    def test_import_module(self):
        from robot_arm.ik_solver import IKSolver
        assert IKSolver is not None

    def test_create_solver_3dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=3)
        assert solver.dof == 3

    def test_create_solver_4dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=4)
        assert solver.dof == 4

    def test_create_solver_5dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        assert solver.dof == 5

    def test_create_solver_6dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=6)
        assert solver.dof == 6

    def test_invalid_dof(self):
        from robot_arm.ik_solver import IKSolver
        with pytest.raises(ValueError):
            IKSolver(dof=2)
        with pytest.raises(ValueError):
            IKSolver(dof=7)


class TestForwardKinematics:
    """Test forward kinematics computation."""

    def test_fk_home_position_5dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        x, y, z, yaw = solver.forward_kinematics(0.0, 0.0, 0.0, 0.0, 0.0)
        # At home (all zeros), arm should point straight up
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert z > 0.0  # Should be positive height

    def test_fk_returns_tuple(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        result = solver.forward_kinematics(0.0, 0.0, 0.0, 0.0, 0.0)
        assert len(result) == 4  # (x, y, z, yaw)

    def test_fk_3dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=3)
        result = solver.forward_kinematics(0.0, 0.5, 0.5)
        assert len(result) == 4  # Still returns (x, y, z, yaw)

    def test_fk_6dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=6)
        result = solver.forward_kinematics(0.0, 0.2, 1.0, 0.5, 0.0, 0.0)
        assert len(result) == 4  # (x, y, z, yaw)


class TestInverseKinematics:
    """Test inverse kinematics computation."""

    def test_ik_reachable_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        result = solver.inverse_kinematics(0.25, 0.0, 0.25, yaw=0.0)
        assert result is not None
        assert len(result) == 5

    def test_ik_reachable_target_6dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=6)
        result = solver.inverse_kinematics(0.25, 0.0, 0.25, yaw=0.0)
        assert result is not None
        assert len(result) == 6

    def test_fk_ik_roundtrip_6dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=6)
        targets = [
            (0.25, 0.0, 0.25, 0.0),
            (0.30, 0.1, 0.26, math.pi / 4),
            (0.20, -0.12, 0.30, -math.pi / 3),
        ]
        for x, y, z, yaw in targets:
            joints = solver.inverse_kinematics(x, y, z, yaw)
            assert joints is not None, f"Target unreachable: {x}, {y}, {z}"
            fx, fy, fz, fyaw = solver.forward_kinematics(*joints)
            err = math.sqrt((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2)
            assert err < 0.005, f"6-DOF roundtrip error {err*1000:.2f}mm > 5mm"
            yaw_diff = abs((yaw - fyaw + math.pi) % (2 * math.pi) - math.pi)
            assert yaw_diff < 0.01, f"Yaw difference {yaw_diff} too large"

    def test_ik_with_rotation_matrix_6dof(self):
        from robot_arm.ik_solver import IKSolver
        from robot_arm.geometric_refinement import make_top_down_rotation
        solver = IKSolver(dof=6)
        yaw = 0.35
        rot = make_top_down_rotation(yaw)
        joints = solver.inverse_kinematics(0.28, 0.05, 0.25, yaw=yaw, rotation_matrix=rot)
        assert joints is not None
        assert len(joints) == 6
        fx, fy, fz, fyaw = solver.forward_kinematics(*joints)
        err = math.sqrt((0.28 - fx) ** 2 + (0.05 - fy) ** 2 + (0.25 - fz) ** 2)
        assert err < 0.005

    def test_ik_unreachable_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        # Very far target
        result = solver.inverse_kinematics(10.0, 0.0, 0.25, yaw=0.0)
        assert result is None

    def test_fk_ik_roundtrip_5dof(self):
        """FK -> IK -> FK should return to same position."""
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        targets = [
            (0.25, 0.0, 0.25, 0.0),
            (0.30, 0.1, 0.26, math.pi / 4),
            (0.20, -0.15, 0.35, -math.pi / 2),
        ]
        for x, y, z, yaw in targets:
            joints = solver.inverse_kinematics(x, y, z, yaw)
            if joints is not None:
                fx, fy, fz, _ = solver.forward_kinematics(*joints)
                err = math.sqrt((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2)
                assert err < 0.005, f"Round-trip error {err*1000:.1f}mm > 5mm for target ({x}, {y}, {z})"

    def test_fk_ik_roundtrip_4dof(self):
        """FK -> IK -> FK for 4-DOF."""
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=4)
        targets = [
            (0.25, 0.0, 0.25),
            (0.30, 0.08, 0.28),
            (0.20, -0.12, 0.32),
        ]
        for x, y, z in targets:
            joints = solver.inverse_kinematics(x, y, z)
            if joints is not None:
                fx, fy, fz, _ = solver.forward_kinematics(*joints)
                err = math.sqrt((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2)
                assert err < 0.005, f"4-DOF error {err*1000:.1f}mm > 5mm for target ({x}, {y}, {z})"

    def test_fk_ik_roundtrip_3dof(self):
        """FK -> IK -> FK for 3-DOF."""
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=3)
        joints = solver.inverse_kinematics(0.25, 0.0, 0.25, yaw=0.0)
        if joints is not None:
            fx, fy, fz, _ = solver.forward_kinematics(*joints)
            err = math.sqrt((0.25 - fx) ** 2 + (0.0 - fy) ** 2 + (0.25 - fz) ** 2)
            assert err < 0.010

    def test_ik_negative_y_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        result = solver.inverse_kinematics(0.20, -0.10, 0.30, yaw=0.0)
        assert result is not None

    def test_ik_pitch_limit_rejection(self):
        """Points with extreme pitch requirement (e.g. high Z close to base) should be rejected."""
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        # Point close to base requiring wrist pitch > 2.2 rad
        result = solver.inverse_kinematics(0.08, 0.0, 0.44, yaw=0.0)
        assert result is None, "Expected point with extreme pitch to be rejected as unreachable"


class TestTrajectoryValidation:
    """Test trajectory pre-validation."""

    def test_validate_reachable_trajectory(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        waypoints = [
            (0.25, 0.0, 0.30, 0.0),
            (0.25, 0.0, 0.25, 0.0),
            (0.25, 0.0, 0.30, 0.0),
        ]
        valid, results = solver.validate_trajectory(waypoints)
        assert valid

    def test_validate_unreachable_trajectory(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        waypoints = [
            (0.25, 0.0, 0.30, 0.0),
            (10.0, 0.0, 0.25, 0.0),  # Unreachable
        ]
        valid, results = solver.validate_trajectory(waypoints)
        assert not valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
