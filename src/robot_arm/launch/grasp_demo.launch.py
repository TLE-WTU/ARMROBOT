"""
Launch full grasp demo: Gazebo + AnyGrasp detection + Pick-and-Place (parameterized by DoF).

Usage:
    # Default scenario (Mug, Duck, Torus) with 5 DoF:
    ros2 launch robot_arm grasp_demo.launch.py dof:=5

    # Benchmark failure scenarios:
    ros2 launch robot_arm grasp_demo.launch.py test_scenario:=transparent_bottle dof:=4
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
    pkg_share = FindPackageShare("robot_arm")
    scenario = LaunchConfiguration("test_scenario").perform(context)
    use_anygrasp = LaunchConfiguration("use_anygrasp")
    grasp_mode = LaunchConfiguration("grasp_mode")
    use_rviz = LaunchConfiguration("use_rviz")
    enable_ransac = LaunchConfiguration("enable_ransac")
    dof = LaunchConfiguration("dof")

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
        launch_arguments={
            "world": world_path,
            "dof": dof,
        }.items(),
    )

    # Parameter files
    anygrasp_params_file = PathJoinSubstitution([
        pkg_share, "config", "anygrasp_params.yaml"
    ])
    robot_params_file = PathJoinSubstitution([
        pkg_share, "config", "robot_params.yaml"
    ])

    grasp_detection_node = Node(
        package="robot_arm",
        executable="grasp_detection_node.py",
        name="grasp_detection_node",
        output="screen",
        parameters=[
            anygrasp_params_file,
            {
                "dof": dof,
                "use_anygrasp": use_anygrasp,
                "grasp_mode": grasp_mode,
                "test_scenario": scenario,
                "enable_ransac": enable_ransac,
                "use_sim_time": True,
            },
        ],
    )

    # Pick-and-Place orchestrator node
    pick_place_node = Node(
        package="robot_arm",
        executable="pick_place_node.py",
        name="pick_place_node",
        output="screen",
        parameters=[
            anygrasp_params_file,
            {
                "dof": dof,
                "use_sim_time": True,
            },
        ],
    )

    # RViz2
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "config.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    # AnyGrasp Deep Learning Inference Service
    conda_python = LaunchConfiguration("conda_python").perform(context)
    checkpoint_path = LaunchConfiguration("checkpoint_path").perform(context)
    anygrasp_service_script = PathJoinSubstitution([
        FindPackagePrefix("robot_arm"), "lib", "robot_arm", "anygrasp_service.py"
    ])
    anygrasp_service_proc = ExecuteProcess(
        cmd=[
            conda_python,
            anygrasp_service_script,
            "--checkpoint_path", checkpoint_path,
            "--socket_path", "/tmp/anygrasp_ipc.sock",
            "--max_gripper_width", "0.07",
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
    dof_arg = DeclareLaunchArgument(
        "dof",
        default_value="5",
        description="Degrees of freedom of the arm (3, 4, or 5)",
    )

    grasp_mode_arg = DeclareLaunchArgument(
        "grasp_mode",
        default_value="heuristic",
        description="Grasp synthesis mode: 'heuristic', 'hybrid' (AI + Geometric Ensemble), 'anygrasp'",
    )

    use_anygrasp_arg = DeclareLaunchArgument(
        "use_anygrasp",
        default_value="false",
        description="Launch AnyGrasp background inference service",
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

    enable_ransac_arg = DeclareLaunchArgument(
        "enable_ransac",
        default_value="true",
        description="Enable RANSAC plane segmentation and geometric reduction",
    )

    default_conda_py = os.environ.get(
        "CONDA_PYTHON",
        os.path.expanduser("~/miniconda3/envs/robot_env/bin/python")
    )
    conda_python_arg = DeclareLaunchArgument(
        "conda_python",
        default_value=default_conda_py,
        description="Path to Python interpreter in AnyGrasp conda environment",
    )

    default_ckpt = os.environ.get(
        "ANYGRASP_CHECKPOINT",
        os.path.expanduser("~/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar")
    )
    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value=default_ckpt,
        description="Path to AnyGrasp model checkpoint file",
    )

    return LaunchDescription([
        dof_arg,
        grasp_mode_arg,
        use_anygrasp_arg,
        use_rviz_arg,
        test_scenario_arg,
        enable_ransac_arg,
        conda_python_arg,
        checkpoint_path_arg,
        OpaqueFunction(function=launch_setup),
    ])
