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

    def test_invalid_dof(self):
        from robot_arm.ik_solver import IKSolver
        with pytest.raises(ValueError):
            IKSolver(dof=2)


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


class TestInverseKinematics:
    """Test inverse kinematics computation."""

    def test_ik_reachable_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver(dof=5)
        result = solver.inverse_kinematics(0.25, 0.0, 0.25, yaw=0.0)
        assert result is not None
        assert len(result) == 5

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
