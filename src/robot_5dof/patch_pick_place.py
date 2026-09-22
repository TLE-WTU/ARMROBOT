with open('robot_5dof/pick_place_node.py', 'r') as f:
    data = f.read()

# 1. Action goal arrays from 4 to 5
data = data.replace(
    'self.joint_names = ["joint1", "joint2", "joint3", "joint4"]',
    'self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]'
)

data = data.replace(
    'home_joints = [0.0, t2_home, t3_home, t4_home]  # [base, shoulder, elbow, wrist_pitch]',
    'home_joints = [0.0, t2_home, t3_home, t4_home, 0.0]  # [base, shoulder, elbow, wrist_pitch, wrist_roll]'
)

# 2. Update IK solver call to use target_yaw!
old_ik_call = '''        joints = inverse_kinematics(x, y, z)
        if joints is None:
            self.get_logger().error(f"IK Failed for ({x:.3f}, {y:.3f}, {z:.3f})")
            return False'''

new_ik_call = '''        # For 5-DOF, we pass target_yaw as well
        joints = inverse_kinematics(x, y, z, target_yaw)
        if joints is None:
            self.get_logger().error(f"IK Failed for ({x:.3f}, {y:.3f}, {z:.3f}, yaw={math.degrees(target_yaw):.1f})")
            return False'''

data = data.replace(old_ik_call, new_ik_call)

# In move_to_pose we need to handle target_yaw
old_move_to_pose = '''    def move_to_pose(self, x, y, z, duration=2.0):
        """Move end effector to (x,y,z) with pitch=180 deg (straight down)."""
        joints = inverse_kinematics(x, y, z)'''

new_move_to_pose = '''    def move_to_pose(self, x, y, z, target_yaw=0.0, duration=2.0):
        """Move end effector to (x,y,z) with pitch=180 deg (straight down) and specified yaw."""
        joints = inverse_kinematics(x, y, z, target_yaw)'''
data = data.replace(old_move_to_pose, new_move_to_pose)

# Update move_to_pose calls inside execute_pick_place
# [2/8] Moving smoothly to pre-grasp
old_pre = 'success = self.move_to_pose(target_x, target_y, pre_grasp_z, duration=3.0)'
new_pre = 'success = self.move_to_pose(target_x, target_y, pre_grasp_z, target_yaw=target_yaw, duration=3.0)'
data = data.replace(old_pre, new_pre)

# [3/8] Lowering down to grasp
old_grasp = 'success = self.move_to_pose(target_x, target_y, grasp_z, duration=2.0)'
new_grasp = 'success = self.move_to_pose(target_x, target_y, grasp_z, target_yaw=target_yaw, duration=2.0)'
data = data.replace(old_grasp, new_grasp)

# [5/8] Lifting object smoothly
old_lift = 'success = self.move_to_pose(target_x, target_y, lift_z, duration=2.0)'
new_lift = 'success = self.move_to_pose(target_x, target_y, lift_z, target_yaw=target_yaw, duration=2.0)'
data = data.replace(old_lift, new_lift)

# [6/8] Moving to place position
old_place = 'success = self.move_to_pose(place_pos.x, place_pos.y, place_pos.z, duration=3.0)'
new_place = 'success = self.move_to_pose(place_pos.x, place_pos.y, place_pos.z, target_yaw=0.0, duration=3.0)'
data = data.replace(old_place, new_place)

with open('robot_5dof/pick_place_node.py', 'w') as f:
    f.write(data)
