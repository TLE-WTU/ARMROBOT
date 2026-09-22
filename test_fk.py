import math
import numpy as np

def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])

def trans_z(d):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]])

j2 = 0.207
j3 = 1.401
j4 = 1.532

T = np.eye(4)
# joint2
T = T @ rot_y(j2)
# link2
T = T @ trans_z(0.25)
# joint3
T = T @ rot_y(j3)
# link3
T = T @ trans_z(0.20)
# joint4
T = T @ rot_y(j4)

print("Final Z vector (direction of link4):")
print(T @ np.array([0, 0, 1, 0]))

T = np.eye(4)
print("Base:", T @ np.array([0,0,0,1]))
T = T @ rot_y(j2)
T = T @ trans_z(0.25)
print("Joint3 pos:", T @ np.array([0,0,0,1]))
T = T @ rot_y(j3)
T = T @ trans_z(0.20)
print("Joint4 pos:", T @ np.array([0,0,0,1]))
T = T @ rot_y(j4)
T = T @ trans_z(0.08)
print("End effector pos:", T @ np.array([0,0,0,1]))
T2 = np.eye(4) @ rot_y(j2) @ trans_z(0.25) @ rot_y(j3)
print("Z-axis of Frame 2:", T2 @ np.array([0,0,1,0]))
print("X-axis of Frame 2:", T2 @ np.array([1,0,0,0]))
