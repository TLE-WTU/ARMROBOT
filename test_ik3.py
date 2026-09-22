import math
import sys
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics

j = inverse_kinematics(0.300, 0.076, 0.275, 0.0)
print([math.degrees(x) for x in j])
