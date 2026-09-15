"""
Launch Gazebo Harmonic with 3 DoF robot, controllers, and camera bridge.

Usage:
    ros2 launch robot_3dof gazebo.launch.py
    ros2 launch robot_3dof gazebo.launch.py world:=empty
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("robot_3dof")

    # ── Launch arguments ──
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=PathJoinSubstitution([pkg_share, "worlds", "pick_and_place.sdf"]),
        description="Path to Gazebo world file",
    )

    # ── Process URDF via xacro ──
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([pkg_share, "urdf", "robot_3dof.urdf.xacro"]),
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # ── Nodes ──

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # Gazebo Harmonic
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "gz_args": ["-r -v 4 ", LaunchConfiguration("world")],
        }.items(),
    )

    # Spawn robot into Gazebo from topic
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "robot_3dof",
            "-topic", "robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.0",
        ],
        output="screen",
    )

    # ros_gz_bridge for camera topics
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen",
    )

    # ── Controller spawners ──
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    # Chain controller spawning: JSB first, then JTC and gripper after JSB is done
    delay_jtc = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[
                joint_trajectory_controller_spawner,
                gripper_controller_spawner,
            ],
        )
    )

    return LaunchDescription([
        world_arg,
        robot_state_publisher,
        gazebo,
        spawn_robot,
        gz_bridge,
        joint_state_broadcaster_spawner,
        delay_jtc,
    ])
