import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

j = inverse_kinematics(0.300, 0.076, 0.355, 0.0)
print(f"Pre-grasp joints: {j}")
