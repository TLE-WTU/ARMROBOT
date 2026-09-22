import math
import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

try:
    j = inverse_kinematics(0.30, 0.08, 0.26, 0)
    print([math.degrees(x) for x in j])
except Exception as e:
    print(e)
