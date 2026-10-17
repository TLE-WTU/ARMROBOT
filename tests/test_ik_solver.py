"""Unit tests for the 4-DOF analytical IK solver."""
import math
import pytest


class TestIKSolverImport:
    """Test that the 4-DOF IK solver module can be imported and initialized."""

    def test_import_module(self):
        from robot_arm.ik_solver import IKSolver
        assert IKSolver is not None

    def test_create_solver_4dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        assert solver.DOF == 4
        assert len(solver.JOINT_NAMES) == 4
        assert solver.JOINT_NAMES == ['joint1', 'joint2', 'joint3', 'joint4']


class TestForwardKinematics:
    """Test forward kinematics computation for 4-DOF."""

    def test_fk_home_position_4dof(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        # Home position: [0.0, 0.0, 0.5, 0.5]
        x, y, z, yaw = solver.forward_kinematics(0.0, 0.0, 0.5, 0.5)
        assert abs(x) > 0.0
        assert abs(y) < 1e-6
        assert z > 0.0

    def test_fk_returns_tuple(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        result = solver.forward_kinematics(0.0, 0.0, 0.5, 0.5)
        assert len(result) == 4  # (x, y, z, yaw)


class TestInverseKinematics:
    """Test inverse kinematics computation for 4-DOF."""

    def test_ik_reachable_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        result = solver.inverse_kinematics(0.25, 0.0, 0.25, yaw=0.0)
        assert result is not None
        assert len(result) == 4

    def test_ik_unreachable_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        # Very far target out of workspace bounds
        result = solver.inverse_kinematics(10.0, 0.0, 0.25, yaw=0.0)
        assert result is None

    def test_fk_ik_roundtrip_4dof(self):
        """FK -> IK -> FK should return to same position with sub-millimeter error."""
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        targets = [
            (0.25, 0.0, 0.25, 0.0),
            (0.30, 0.1, 0.26, math.pi / 4),
            (0.20, -0.15, 0.35, -math.pi / 2),
            (0.18, 0.12, 0.23, 0.0),
        ]
        for x, y, z, yaw in targets:
            joints = solver.inverse_kinematics(x, y, z, yaw)
            assert joints is not None, f"Target ({x}, {y}, {z}) should be reachable"
            fx, fy, fz, _ = solver.forward_kinematics(*joints)
            err = math.sqrt((x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2)
            assert err < 0.001, f"Round-trip error {err*1000:.3f}mm > 1mm for target ({x}, {y}, {z})"

    def test_ik_negative_y_target(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        result = solver.inverse_kinematics(0.20, -0.10, 0.30, yaw=0.0)
        assert result is not None
        assert len(result) == 4


class TestTrajectoryValidation:
    """Test trajectory pre-validation."""

    def test_validate_reachable_trajectory(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        waypoints = [
            (0.25, 0.0, 0.30, 0.0),
            (0.25, 0.0, 0.25, 0.0),
            (0.25, 0.0, 0.30, 0.0),
        ]
        valid, results = solver.validate_trajectory(waypoints)
        assert valid
        assert len(results) == 3
        for r in results:
            assert r is not None and len(r) == 4

    def test_validate_unreachable_trajectory(self):
        from robot_arm.ik_solver import IKSolver
        solver = IKSolver()
        waypoints = [
            (0.25, 0.0, 0.30, 0.0),
            (10.0, 0.0, 0.25, 0.0),  # Unreachable
        ]
        valid, results = solver.validate_trajectory(waypoints)
        assert not valid
        assert results[0] is not None
        assert results[1] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
