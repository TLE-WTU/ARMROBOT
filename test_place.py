import sys
import numpy as np
import math
sys.path.append('src/robot_5dof/robot_5dof')
from ik_solver import inverse_kinematics
j = inverse_kinematics(0.200, -0.150, 0.380, -math.radians(68.7))
print(f"Joints for place: {j}")
if j:
    print(f"Degrees: {np.degrees(j)}")
