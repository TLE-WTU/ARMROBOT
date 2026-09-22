import re

with open('urdf/robot_5dof.urdf.xacro', 'r') as f:
    urdf = f.read()

# 1. Add link5 properties
link4_prop = '<xacro:property name="link4_mass" value="0.15"/>'
link5_prop = '''<xacro:property name="link4_mass" value="0.15"/>

  <!-- Link5 - Wrist Roll (new 5th DOF: wrist roll around Z) -->
  <xacro:property name="link5_radius" value="0.025"/>
  <xacro:property name="link5_height" value="0.02"/>
  <xacro:property name="link5_mass" value="0.1"/>'''
urdf = urdf.replace(link4_prop, link5_prop)

# 2. Add link5 and joint5
link4_def = '''  <link name="link4">
    <xacro:cylinder_inertia m="${link4_mass}" r="${link4_radius}" h="${link4_height}"/>
    <visual>
      <origin xyz="0 0 ${link4_height/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${link4_radius}" length="${link4_height}"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${link4_height/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${link4_radius}" length="${link4_height}"/>
      </geometry>
    </collision>
  </link>'''

link5_def = '''  <link name="link4">
    <xacro:cylinder_inertia m="${link4_mass}" r="${link4_radius}" h="${link4_height}"/>
    <visual>
      <origin xyz="0 0 ${link4_height/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${link4_radius}" length="${link4_height}"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${link4_height/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${link4_radius}" length="${link4_height}"/>
      </geometry>
    </collision>
  </link>

  <joint name="joint5" type="revolute">
    <parent link="link4"/>
    <child link="link5"/>
    <origin xyz="0 0 ${link4_height}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-pi}" upper="${pi}" velocity="3.0" effort="20.0"/>
    <dynamics damping="0.1" friction="0.05"/>
  </joint>

  <link name="link5">
    <xacro:cylinder_inertia m="${link5_mass}" r="${link5_radius}" h="${link5_height}"/>
    <visual>
      <origin xyz="0 0 ${link5_height/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${link5_radius}" length="${link5_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.4 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${link5_height/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${link5_radius}" length="${link5_height}"/>
      </geometry>
    </collision>
  </link>'''

urdf = urdf.replace(link4_def, link5_def)

# 3. Change gripper parent to link5
gripper_joint = '''  <joint name="gripper_base_joint" type="fixed">
    <parent link="link4"/>
    <child link="gripper_base"/>
    <origin xyz="0 0 ${link4_height}" rpy="0 0 0"/>
  </joint>'''

gripper_joint_new = '''  <joint name="gripper_base_joint" type="fixed">
    <parent link="link5"/>
    <child link="gripper_base"/>
    <origin xyz="0 0 ${link5_height}" rpy="0 0 0"/>
  </joint>'''

urdf = urdf.replace(gripper_joint, gripper_joint_new)

with open('urdf/robot_5dof.urdf.xacro', 'w') as f:
    f.write(urdf)
