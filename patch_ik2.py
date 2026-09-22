with open("src/robot_5dof/robot_5dof/ik_solver.py", "r") as f:
    content = f.read()

new_content = content.replace(
    "if t4 > math.pi/2:\n        t4 = math.pi/2",
    "if t4 > math.pi/2:\n        print(f'[IK WARNING] Wrist pitch {math.degrees(t4):.1f}° clamped to 90° to prevent self-collision!')\n        t4 = math.pi/2"
)

with open("src/robot_5dof/robot_5dof/ik_solver.py", "w") as f:
    f.write(new_content)
