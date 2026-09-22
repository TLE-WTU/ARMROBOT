with open("src/robot_5dof/robot_5dof/ik_solver.py", "r") as f:
    content = f.read()

new_content = content.replace(
    "t4 = math.pi - (t2 + t3)",
    "t4 = math.pi - (t2 + t3)\n    # Prevent wrist from folding backwards into forearm (>90 deg)\n    if t4 > math.pi/2:\n        t4 = math.pi/2\n    elif t4 < -math.pi/2:\n        t4 = -math.pi/2"
)

with open("src/robot_5dof/robot_5dof/ik_solver.py", "w") as f:
    f.write(new_content)
