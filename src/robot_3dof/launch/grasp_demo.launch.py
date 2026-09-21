"""
Launch full grasp demo: Gazebo + AnyGrasp detection + Pick-and-Place.

Usage:
    # Default scenario (Mug, Duck, Torus):
    ros2 launch robot_3dof grasp_demo.launch.py

    # Benchmark failure scenarios:
    ros2 launch robot_3dof grasp_demo.launch.py test_scenario:=transparent_bottle
    ros2 launch robot_3dof grasp_demo.launch.py test_scenario:=flat_object
    ros2 launch robot_3dof grasp_demo.launch.py test_scenario:=dense_clutter
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare("robot_3dof")
    scenario = LaunchConfiguration("test_scenario").perform(context)
    use_anygrasp = LaunchConfiguration("use_anygrasp")
    use_rviz = LaunchConfiguration("use_rviz")

    # Map scenario to dedicated Gazebo SDF world
    world_map = {
        "transparent_bottle": "pick_and_place_bottle.sdf",
        "flat_object": "pick_and_place_flat.sdf",
        "dense_clutter": "pick_and_place_clutter.sdf",
        "default": "pick_and_place.sdf",
    }
    world_filename = world_map.get(scenario, "pick_and_place.sdf")
    world_path = PathJoinSubstitution([pkg_share, "worlds", world_filename])

    # Include base gazebo launch with the dynamically selected world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_share, "launch", "gazebo.launch.py"])
        ]),
        launch_arguments={"world": world_path}.items(),
    )

    # AnyGrasp detection node
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
                "use_anygrasp": use_anygrasp,
                "test_scenario": scenario,
            },
        ],
    )

    # Pick-and-Place orchestrator node
    pick_place_node = Node(
        package="robot_3dof",
        executable="pick_place_node.py",
        name="pick_place_node",
        output="screen",
        parameters=[anygrasp_params_file],
    )

    # RViz2
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "config.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    # AnyGrasp Deep Learning Inference Service
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
        condition=IfCondition(use_anygrasp),
    )

    return [
        anygrasp_service_proc,
        gazebo_launch,
        grasp_detection_node,
        pick_place_node,
        rviz_node,
    ]


def generate_launch_description():
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

    return LaunchDescription([
        use_anygrasp_arg,
        use_rviz_arg,
        test_scenario_arg,
        OpaqueFunction(function=launch_setup),
    ])
