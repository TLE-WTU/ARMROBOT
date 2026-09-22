with open('urdf/robot_5dof_ros2_control.xacro', 'r') as f:
    data = f.read()

j4 = '''    <joint name="joint4">
      <command_interface name="position">
        <param name="min">-3.14</param>
        <param name="max">3.14</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>'''

j5 = j4 + '''

    <joint name="joint5">
      <command_interface name="position">
        <param name="min">-3.14</param>
        <param name="max">3.14</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>'''

data = data.replace(j4, j5)
with open('urdf/robot_5dof_ros2_control.xacro', 'w') as f:
    f.write(data)
