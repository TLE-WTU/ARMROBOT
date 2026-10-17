# ARMROBOT

[![CI](https://github.com/TLE-WTU/ARMROBOT/actions/workflows/ci.yaml/badge.svg)](https://github.com/TLE-WTU/ARMROBOT/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ROS 2 package for a 4-DOF articulated robotic manipulator featuring analytical inverse kinematics, Franka Hand parallel-jaw gripper integration, and a hybrid grasp synthesis pipeline combining planar RANSAC segmentation with geometric refiners and deep affordance prediction.

---

## Overview

ARMROBOT provides a modular, resource-efficient manipulation framework built on ROS 2 Jazzy and Gazebo Harmonic. It is targeted at tabletop pick-and-place tasks with structured top-down approach geometries.

### Features
- **Analytical Kinematics**: Closed-form forward and inverse kinematics solvers with sub-millimeter precision, running in under 10 microseconds without numerical convergence issues.
- **Top-Down Grasp Alignment**: Automatic wrist pitch compensation maintains the end-effector perpendicular to the support surface throughout the reachable workspace.
- **Franka Hand Gripper**: Prismatic parallel-jaw configuration with 70 mm maximum aperture and compliant contact dynamics.
- **Hybrid Grasp Synthesis**: Integrates 5 geometric refinement algorithms (PCA, OBB, Surface Normals, Cross-Section Slicing, Primitive RANSAC) with optional AnyGrasp deep-learning affordance.
- **Simulation and Benchmarking**: Full dynamics in Gazebo Harmonic with ros2_control, plus headless PyBullet rollouts for batch evaluation.

---

## System Architecture

```
                    +------------------------------------+
                    |        RGB-D Point Cloud           |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   RANSAC Table Plane Segmentation  |
                    |   & Workspace Voxel Downsampling   |
                    +-----------------+------------------+
                                      |
                   +------------------+------------------+
                   |                                     |
                   v                                     v
       +-----------------------+             +-----------------------+
       |   AnyGrasp Model      |             |  Geometric Refiners   |
       |  (PyTorch / IPC)      |             |  (PCA, OBB, Normals)  |
       +-----------+-----------+             +-----------+-----------+
                   |                                     |
                   +------------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   Virtual Safety Floor & Ensemble  |
                    +-----------------+------------------+
                                      | Target Pose (x, y, z, yaw)
                                      v
                    +------------------------------------+
                    |    Analytical 4-DOF IK Solver      |
                    +-----------------+------------------+
                                      | Joint Trajectory
                                      v
                    +------------------------------------+
                    |    ros2_control Joint Controller   |
                    +------------------------------------+
```

---

## Kinematics Specification

### Manipulator Parameters

| Joint | Type | Axis | Description | Range |
| :--- | :--- | :---: | :--- | :---: |
| `joint1` | Revolute | $Z$ | Base rotation (Yaw) | $[-\pi, +\pi]$ rad |
| `joint2` | Revolute | $Y$ | Shoulder pitch | $[-2.5, +2.5]$ rad |
| `joint3` | Revolute | $Y$ | Elbow pitch | $[-2.5, +2.5]$ rad |
| `joint4` | Revolute | $Y$ | Wrist pitch (approach angle) | $[-\pi, +\pi]$ rad |
| `gripper_left_joint` | Prismatic | $Y$ | Franka left finger | $[0.0, 0.035]$ m |
| `gripper_right_joint` | Prismatic | $Y$ | Franka right finger (mimic) | $[0.0, 0.035]$ m |

### Link Dimensions
- Base height ($h_{\text{base}}$): $0.175\ \text{m}$ (from `base_link` origin to `joint2` shoulder axis)
- Upper arm length ($L_1$): $0.250\ \text{m}$ (`joint2` to `joint3`)
- Forearm length ($L_2$): $0.200\ \text{m}$ (`joint3` to `joint4`)
- Tool center point offset ($L_{\text{hand}}$): $0.1334\ \text{m}$ (`joint4` to grasp midpoint)
- **Reachable Workspace**: $X \in [0.10, 0.50]\ \text{m}$, $Y \in [-0.20, 0.20]\ \text{m}$, $Z \in [0.22, 0.45]\ \text{m}$

### Inverse Kinematics Formulation

Given target Cartesian coordinates $\mathbf{p}^* = (x^*, y^*, z^*)^T$:

1. **Base Yaw ($\theta_1$)**:
   $$\theta_1 = \text{atan2}(y^*, x^*)$$

2. **Wrist Position**:
   Enforcing the vertical descent condition ($\sum_{i=2}^4 \theta_i = \pi$), the wrist center is projected:
   $$r_{\text{wrist}} = \sqrt{(x^*)^2 + (y^*)^2}$$
   $$z_{\text{wrist}} = z^* + L_{\text{hand}}$$

3. **Shoulder ($\theta_2$) and Elbow ($\theta_3$) Angles**:
   Let $\Delta z = z_{\text{wrist}} - h_{\text{base}}$ and $D = \sqrt{r_{\text{wrist}}^2 + \Delta z^2}$.
   A solution exists if $|L_1 - L_2| \le D \le L_1 + L_2$.
   $$\cos\theta_3 = \frac{D^2 - L_1^2 - L_2^2}{2 L_1 L_2} \implies \theta_3 = \arccos(\text{clip}(\cos\theta_3, -1, 1))$$
   $$\alpha = \text{atan2}(r_{\text{wrist}}, \Delta z), \quad \beta = \text{atan2}(L_2 \sin\theta_3, L_1 + L_2 \cos\theta_3)$$
   $$\theta_2 = \alpha - \beta$$

4. **Wrist Pitch Compensation ($\theta_4$)**:
   $$\theta_4 = \pi - (\theta_2 + \theta_3)$$

---

## Perception and Grasp Synthesis

The perception node processes raw `sensor_msgs/PointCloud2` input through the following stages:

1. **Table Plane Segmentation**: RANSAC plane fitting isolates the dominant horizontal support surface and estimates table height $z_{\text{table}}$.
2. **Cluster Extraction & Geometric Refinement**:
   - **PCA**: Decomposes spatial covariance to align gripper orientation with the object's minimum inertia axis.
   - **OBB**: Computes minimum bounding box to determine required finger stroke.
   - **Surface Normals**: Estimates antipodal contact pairs using $k$-nearest neighbors normal estimation.
   - **Cross-Section Slicing**: Evaluates horizontal slices to locate narrow grip regions (e.g., bottle necks).
   - **Primitive RANSAC**: Fits cylinders and bounding boxes to regular geometric primitives.
3. **Safety Verification**: Grasp candidates with fingertip penetration into the table surface are rejected or elevated via the virtual safety floor module.

---

## Benchmark Results

Empirical performance measured across 99 physical trials on standardized benchmark objects (Mug, Bottle, Duck, Torus, Disc, Box, Clutter) in simulation:

| Method | Architecture | Success Rate | IK Feasible | Table Collisions | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA)** | CPU Algebraic | **77.8%** | 100.0% | 0.0% | 0.3 ms |
| **Pure Geometric (Normals)** | CPU Algebraic | **77.8%** | 100.0% | 0.0% | 110.7 ms |
| **Pure Geometric (OBB)** | CPU Algebraic | **66.7%** | 100.0% | 0.0% | 0.4 ms |
| **Pure Geometric (Slice)** | CPU Algebraic | **55.6%** | 100.0% | 0.0% | 0.5 ms |
| **Hybrid (AnyGrasp + OBB)** | GPU AI + CPU Geometry | **55.6%** | 100.0% | 0.0% | 127.5 ms |
| **Hybrid (AnyGrasp + PCA)** | GPU AI + CPU Geometry | **44.4%** | 100.0% | 0.0% | 98.4 ms |
| **Pure AnyGrasp (Baseline)** | Deep Network Only | 0.0% | 0.0% | 66.7% | 132.6 ms |

For detailed breakdowns, see [ACADEMIC_BENCHMARK_REPORT.md](benchmark/ACADEMIC_BENCHMARK_REPORT.md).

---

## Installation

### Dependencies
- Ubuntu 24.04 LTS
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic

Install ROS 2 dependencies:
```bash
sudo apt update && sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-controller-manager \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-position-controllers \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  python3-colcon-common-extensions \
  python3-pytest
```

Install Python requirements:
```bash
pip install -r requirements.txt
```

### Build Workspace
```bash
cd ~/ARMROBOT
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select robot_arm --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

---

## Usage

### 1. Launch Simulation Environment
Starts Gazebo Harmonic with the 4-DOF arm and ros2_control interface:
```bash
ros2 launch robot_arm gazebo.launch.py
```

### 2. Run Autonomous Pick-and-Place
```bash
# Heuristic mode (CPU only)
ros2 launch robot_arm grasp_demo.launch.py use_anygrasp:=false grasp_mode:=heuristic

# Hybrid mode (AI + Geometric refinement)
ros2 launch robot_arm grasp_demo.launch.py use_anygrasp:=true grasp_mode:=hybrid
```

Specific test scenarios:
```bash
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=transparent_bottle
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=dense_clutter
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=flat_object
```

### 3. Run Benchmark Suite
```bash
# Headless physical grasp evaluation in PyBullet
python3 benchmark/run_academic_benchmark.py --headless --output_dir ./benchmark

# Geometric refiner comparison
python3 scripts/run_geometric_benchmark.py
```

---

## Testing

Run unit tests and benchmark evaluation tests:
```bash
PYTHONPATH=./:src/robot_arm pytest -v tests/
```

Test coverage includes:
- Analytical forward and inverse kinematics consistency
- Trajectory pre-validation and workspace limits
- Planar RANSAC and geometric refiner operations
- GraspNet Average Precision ($AP_{0.4}$, $AP_{0.8}$) metrics
- Coulomb friction cone and force-closure criteria

---

## Repository Structure

```
ARMROBOT/
├── .github/workflows/ci.yaml       # Continuous integration workflow
├── benchmark/                      # Grasp evaluation metrics and rollout harness
│   ├── ACADEMIC_BENCHMARK_REPORT.md
│   ├── dataset_loader.py
│   ├── eval_ap.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── physics_environment.py
│   └── run_academic_benchmark.py
├── scripts/
│   └── run_geometric_benchmark.py  # Standalone benchmark runner
├── src/robot_arm/                  # ROS 2 package
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/                     # Controller and parameter definitions
│   │   ├── anygrasp_params.yaml
│   │   ├── controllers_4dof.yaml
│   │   └── robot_params.yaml
│   ├── launch/                     # Launch scripts
│   │   ├── gazebo.launch.py
│   │   └── grasp_demo.launch.py
│   ├── meshes/                     # Arm and object STL/DAE models
│   ├── robot_arm/                  # Python node implementations
│   │   ├── anygrasp_service.py
│   │   ├── benchmark_dataset.py
│   │   ├── geometric_refinement.py
│   │   ├── grasp_detection_node.py
│   │   ├── ik_solver.py
│   │   └── pick_place_node.py
│   ├── urdf/                       # URDF and ros2_control Xacro definitions
│   └── worlds/                     # Gazebo simulation worlds
├── tests/                          # Automated unit test suite
├── requirements.txt
└── RUN_GUIDE.md
```

---

## References

1. **Ferrari, C., & Canny, J.** (1992). *Planning optimal grasps*. IEEE International Conference on Robotics and Automation (ICRA).
2. **Nguyen, V.-D.** (1988). *Constructing force-closure grasps*. The International Journal of Robotics Research, 7(3), 3–16.
3. **Fang, H.-S., et al.** (2020). *GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

---

## License

This project is licensed under the [MIT License](LICENSE).
