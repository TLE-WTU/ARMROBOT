import math
import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

try:
    j = inverse_kinematics(0.20, -0.15, 0.34, 0)
    print([math.degrees(x) for x in j])
except Exception as e:
    print(e)
