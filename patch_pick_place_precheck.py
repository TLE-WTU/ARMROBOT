import re

with open('src/robot_5dof/robot_5dof/pick_place_node.py', 'r') as f:
    content = f.read()

# Just replace the offending error string lines with multiple standard lines
old_err = """                self.get_logger().error(f"  Lift:      {j_lift is not None} (z={lift_z:.3f})
  Pre-place: {j_pre_place is not None} (z={pre_place_z:.3f})
  Place:     {j_place is not None} (z={pz:.3f})")"""

new_err = """                self.get_logger().error(f"  Lift:      {j_lift is not None} (z={lift_z:.3f})")
                self.get_logger().error(f"  Pre-place: {j_pre_place is not None} (z={pre_place_z:.3f})")
                self.get_logger().error(f"  Place:     {j_place is not None} (z={pz:.3f})")"""

content = content.replace(old_err, new_err)

with open('src/robot_5dof/robot_5dof/pick_place_node.py', 'w') as f:
    f.write(content)
