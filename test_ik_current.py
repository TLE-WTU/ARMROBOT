import math
import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

print("GRASP:")
try:
    j = inverse_kinematics(0.300, 0.076, 0.275, -1.2)
    print([math.degrees(x) for x in j])
except Exception as e:
    print(e)

print("PRE-GRASP:")
try:
    j = inverse_kinematics(0.300, 0.076, 0.355, -1.2)
    print([math.degrees(x) for x in j])
except Exception as e:
    print(e)
