"""
Unit Tests for Academic Benchmark Metrics and Evaluation Suite.
"""

import math
import numpy as np
import pytest

from benchmark.metrics import (
    compute_antipodal_force_closure,
    check_table_collision,
    check_aperture_compliance,
    check_kinematic_feasibility,
)
from benchmark.evaluator import GraspEvaluator
from robot_arm.ik_solver import IKSolver


def test_antipodal_force_closure_perfect_alignment():
    """Opposing normals along closing axis should pass force closure."""
    p1 = np.array([0.0, -0.02, 0.25])
    p2 = np.array([0.0, 0.02, 0.25])
    n1 = np.array([0.0, 1.0, 0.0])   # pointing inward along +Y
    n2 = np.array([0.0, -1.0, 0.0])  # pointing inward along -Y

    is_fc, a1, a2 = compute_antipodal_force_closure(p1, p2, n1, n2, friction_coeff=0.4)
    assert is_fc is True
    assert a1 < 1.0
    assert a2 < 1.0


def test_antipodal_force_closure_rejected_shear():
    """Normals perpendicular to closing axis must fail force closure."""
    p1 = np.array([0.0, -0.02, 0.25])
    p2 = np.array([0.0, 0.02, 0.25])
    n1 = np.array([1.0, 0.0, 0.0])   # perpendicular (+X)
    n2 = np.array([-1.0, 0.0, 0.0])  # perpendicular (-X)

    is_fc, a1, a2 = compute_antipodal_force_closure(p1, p2, n1, n2, friction_coeff=0.4)
    assert is_fc is False


def test_table_collision_detection():
    """Poses whose fingertips penetrate table_z must trigger collision flag."""
    table_z = 0.225
    rot = np.eye(3)

    # Safe pose (well above table)
    safe_pos = np.array([0.3, 0.0, 0.30])
    assert check_table_collision(safe_pos, rot, table_z=table_z) is False

    # Penetrating pose (fingertip extends 5cm down into table)
    colliding_pos = np.array([0.3, 0.0, 0.23])
    assert check_table_collision(colliding_pos, rot, table_z=table_z) is True


def test_aperture_compliance():
    """Gripper aperture must be within physical stroke limits (2mm - 80mm)."""
    assert check_aperture_compliance(0.040, max_gripper_width=0.08) is True
    assert check_aperture_compliance(0.080, max_gripper_width=0.08) is True
    assert check_aperture_compliance(0.095, max_gripper_width=0.08) is False
    assert check_aperture_compliance(0.001, max_gripper_width=0.08) is False


def test_kinematic_feasibility_with_ik():
    """Valid reachable pose should return feasible; out-of-reach should return infeasible."""
    solver = IKSolver(dof=5)

    # Reachable pose within arm workspace
    reachable_pos = np.array([0.30, 0.0, 0.25])
    is_reach, angles = check_kinematic_feasibility(reachable_pos, 0.0, solver)
    assert is_reach is True
    assert angles is not None

    # Unreachable pose (too far away)
    unreachable_pos = np.array([1.50, 0.0, 0.25])
    is_reach, angles = check_kinematic_feasibility(unreachable_pos, 0.0, solver)
    assert is_reach is False
    assert angles is None


def test_evaluator_analytical_pipeline():
    """GraspEvaluator should correctly compute analytical metrics for geometric grasps."""
    solver = IKSolver(dof=5)
    evaluator = GraspEvaluator(table_z=0.225, ik_solver=solver)

    # Generate synthetic planar cluster
    rng = np.random.default_rng(42)
    pts = rng.uniform([0.28, -0.02, 0.25], [0.32, 0.02, 0.28], (200, 3))

    grasps, latency = evaluator.run_geometric_pipeline(pts, method="pca")
    assert len(grasps) > 0
    assert latency >= 0.0

    metrics = evaluator.evaluate_analytical(grasps, pts)
    assert metrics["valid_grasp_found"] is True
    assert isinstance(metrics["ik_feasible"], bool)
    assert isinstance(metrics["table_collision"], bool)
