import math
import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

j = inverse_kinematics(0.400, 0.000, 0.150, 0.0)
print([math.degrees(x) for x in j])
