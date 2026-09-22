import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

px, py, pz = 0.0, -0.28, 0.26
pre_place_z = pz + 0.10

j_pre_place = inverse_kinematics(px, py, pre_place_z, 0.0)
j_place = inverse_kinematics(px, py, pz, 0.0)

print(f"pre_place: {j_pre_place is not None}")
print(f"place: {j_place is not None}")
