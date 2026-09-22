with open("src/robot_5dof/urdf/robot_5dof.urdf.xacro", "r") as f:
    lines = f.readlines()

in_end_effector = False
for i, line in enumerate(lines):
    if 'name="end_effector_joint"' in line:
        in_end_effector = True
    if in_end_effector and '<parent link="link4"/>' in line:
        lines[i] = line.replace('link4', 'link5')
    if in_end_effector and '<origin' in line:
        lines[i] = line.replace('link4_height', 'link5_height')
        in_end_effector = False

with open("src/robot_5dof/urdf/robot_5dof.urdf.xacro", "w") as f:
    f.writelines(lines)
