import math
import numpy as np
def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
j2 = 0.207
j3 = 1.401
j4 = -1.532
T = np.eye(4) @ rot_y(j2) @ rot_y(j3) @ rot_y(j4)
print("Z-axis:", T @ np.array([0,0,1,0]))
