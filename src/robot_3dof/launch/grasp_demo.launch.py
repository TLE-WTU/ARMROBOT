"""
Launch full grasp demo: Gazebo + AnyGrasp detection + Pick-and-Place.

Usage:
    # Fallback mode (no AnyGrasp license needed):
    ros2 launch robot_3dof grasp_demo.launch.py

    # With AnyGrasp SDK:
    ros2 launch robot_3dof grasp_demo.launch.py use_anygrasp:=true
"""

import os
import launch
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("robot_3dof")

    # ── Launch arguments ──
    use_anygrasp_arg = DeclareLaunchArgument(
        "use_anygrasp",
        default_value="false",
        description="Use AnyGrasp SDK (true) or heuristic fallback (false)",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2 for visualization",
    )

    test_scenario_arg = DeclareLaunchArgument(
        "test_scenario",
        default_value="default",
        description="Benchmark scenario: 'default', 'transparent_bottle', 'flat_object', 'dense_clutter'",
    )

    # ── Include base gazebo launch ──
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_share, "launch", "gazebo.launch.py"])
        ]),
    )

    # ── AnyGrasp detection node ──
    anygrasp_params_file = PathJoinSubstitution([
        pkg_share, "config", "anygrasp_params.yaml"
    ])

    grasp_detection_node = Node(
        package="robot_3dof",
        executable="grasp_detection_node.py",
        name="grasp_detection_node",
        output="screen",
        parameters=[
            anygrasp_params_file,
            {
                "use_anygrasp": LaunchConfiguration("use_anygrasp"),
                "test_scenario": LaunchConfiguration("test_scenario"),
            },
        ],
    )

    # ── Pick-and-Place orchestrator node ──
    pick_place_node = Node(
        package="robot_3dof",
        executable="pick_place_node.py",
        name="pick_place_node",
        output="screen",
        parameters=[anygrasp_params_file],
    )

    # ── RViz2 ──
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "config.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    # ── AnyGrasp Deep Learning Inference Service ──
    # Runs in Python 3.10 Conda environment 'robot_env' with GPU CUDA & MinkowskiEngine
    conda_python = "/home/tienle/miniconda3/envs/robot_env/bin/python"
    anygrasp_service_script = PathJoinSubstitution([
        FindPackagePrefix("robot_3dof"), "lib", "robot_3dof", "anygrasp_service.py"
    ])
    anygrasp_service_proc = ExecuteProcess(
        cmd=[
            conda_python,
            anygrasp_service_script,
            "--checkpoint_path", "/home/tienle/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
            "--socket_path", "/tmp/anygrasp_ipc.sock",
            "--max_gripper_width", "0.06",
            "--gripper_height", "0.04",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_anygrasp")),
    )

    return LaunchDescription([
        use_anygrasp_arg,
        use_rviz_arg,
        test_scenario_arg,
        anygrasp_service_proc,
        gazebo_launch,
        grasp_detection_node,
        pick_place_node,
        rviz_node,
    ])
