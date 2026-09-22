import re

with open('src/robot_5dof/robot_5dof/pick_place_node.py', 'r') as f:
    code = f.read()

code = re.sub(
    r"t2_home, t3_home = -1\.0, 1\.0\n\s+t4_home = math\.pi - \(t2_home \+ t3_home\)",
    "t2_home = 0.0\n        t3_home = 0.5\n        t4_home = 0.5",
    code
)

with open('src/robot_5dof/robot_5dof/pick_place_node.py', 'w') as f:
    f.write(code)
print("Patched!")
