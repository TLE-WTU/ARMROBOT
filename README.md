# ARMROBOT: 6-DOF Robotic Manipulation Framework with Active Dual-Camera Perception and Geometric Grasp Synthesis

[![CI](https://github.com/TLE-WTU/ARMROBOT/actions/workflows/ci.yaml/badge.svg)](https://github.com/TLE-WTU/ARMROBOT/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/TLE-WTU/ARMROBOT/branch/main/graph/badge.svg)](https://codecov.io/gh/TLE-WTU/ARMROBOT)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-blue?logo=ros)](https://docs.ros.org/en/jazzy/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange?logo=gazebo)](https://gazebosim.org/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-3776AB?logo=python)](https://www.python.org/)
[![Physics](https://img.shields.io/badge/physics-PyBullet%20Dynamic-red)](https://pybullet.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ARMROBOT is an open-source ROS 2 manipulation framework designed for autonomous tabletop pick-and-place. The system features a 6-DOF articulated serial arm with a spherical wrist, an active dual-camera perception system, an analytical decoupled inverse kinematics solver, and a benchmarked grasp synthesis pipeline comparing pure geometric methods, deep learning (AnyGrasp), and hybrid architectures.

All performance evaluations are grounded in PyBullet multi-body dynamic contact physics ($80\text{ N}$ bilateral grasp force, Coulomb friction cone analysis $\mu=0.8$, and vertical lift verification) as well as point cloud captures from an Intel RealSense D435 sensor.

---

## System Overview

| Kinematic Model & Dual-Camera Setup | Dynamic Physics Simulation & Rollout |
| :---: | :---: |
| ![6-DOF Manipulator Diagram](media/full_robot_preview.png) | ![Simulation Rollout](media/sim_grasp_preview.png) |
| *Figure 1: 6-DOF serial manipulator with spherical wrist ($SO(3)$ dexterity), Franka parallel-jaw gripper, and active dual dynamic cameras (Eye-in-Hand on gripper + tracking camera on rotating base).* | *Figure 2: Tabletop manipulation rollout in dynamic simulation with benchmark geometries and tabletop support surface.* |

### Core Components

- **6-DOF Kinematic Chain**: Serial manipulator featuring a 3-axis spherical wrist ($\text{Yaw} \to \text{Pitch} \to \text{Roll}$), providing unconstrained end-effector orientation across the operational workspace ($R \le 0.45\text{ m}$).
- **Active Dual-Camera Perception**:
  - *Eye-in-Hand Wrist Camera* (`camera_wrist_link`): Mounted on the Franka gripper flange to provide localized, high-resolution observations during the approach phase.
  - *Turret Tracking Camera* (`camera_arm_link`): Mounted on the rotating base (`link1`), panning dynamically with joint 1 to maintain visual line-of-sight and avoid arm self-occlusion.
- **Analytical Decoupled IK Solver**: Closed-form solver separating Cartesian wrist center translation from wrist spherical orientation, achieving positional error $< 0.005\text{ mm}$ in $< 0.05\text{ ms}$ compute time with zero iterative drift.
- **Modular Grasp Synthesis**: Supports 5 pure analytical geometric algorithms (PCA, OBB, Surface Normals, Cross-Section Slicing, Primitive RANSAC), pure deep learning (AnyGrasp GSNet), and hybrid architectures.
- **Support Surface Safety**: Analytical RANSAC plane segmentation and workspace bounding eliminate support collision risks common in unconstrained neural grasp generators.

---

## Kinematics & Analytical Decoupled Inverse Kinematics

The forward kinematics of the 6-DOF manipulator is formulated as the product of successive homogeneous transformation matrices:

$$T_6^0(\mathbf{\theta}) = \prod_{i=1}^6 T_i^{i-1}(\theta_i) = \begin{bmatrix} R_6^0(\mathbf{\theta}) & \mathbf{p}_{ee}(\mathbf{\theta}) \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Given a target end-effector pose $(R_{target}, \mathbf{p}_{target})$ and tool length $d_{tool}$, the position of the spherical wrist center $\mathbf{p}_w = [x_w, y_w, z_w]^T$ is kinematically decoupled from orientation:

$$\mathbf{p}_w = \mathbf{p}_{target} - d_{tool} \, R_{target} \, \hat{\mathbf{z}}$$

### 1. Positional Joint Resolution $(\theta_1, \theta_2, \theta_3)$

1. **Base Turret Yaw ($\theta_1$)**:
   $$\theta_1 = \text{atan2}(y_w, x_w)$$

2. **Shoulder Pitch ($\theta_2$) & Elbow Pitch ($\theta_3$)**:
   Let $r = \sqrt{x_w^2 + y_w^2}$ and $z' = z_w - d_1$ (where $d_1$ is the base turret elevation). The radial distance from the shoulder pivot to the wrist center is $D = \sqrt{r^2 + z'^2}$. Using the Law of Cosines for upper arm length $a_2$ and forearm length $a_3$:

   $$\cos(\theta_3) = \frac{D^2 - a_2^2 - a_3^2}{2 a_2 a_3}$$
   $$\theta_3 = -\arccos\left(\text{clamp}\left(\frac{D^2 - a_2^2 - a_3^2}{2 a_2 a_3}, -1, 1\right)\right)$$
   $$\theta_2 = \text{atan2}(z', r) - \text{atan2}(a_3 \sin(\theta_3), a_2 + a_3 \cos(\theta_3))$$

### 2. Spherical Wrist Orientation $(\theta_4, \theta_5, \theta_6)$

With the arm configuration $(\theta_1, \theta_2, \theta_3)$ determined, the rotation matrix of the spherical wrist relative to the forearm is isolated:

$$R_6^3 = (R_3^0(\theta_1, \theta_2, \theta_3))^T R_{target}$$

Decomposing $R_6^3$ via Euler angles yields the closed-form orientation solution:

$$\theta_5 = \arccos\left(\text{clamp}(R_{6,33}^3, -1, 1)\right)$$
$$\theta_4 = \text{atan2}(R_{6,23}^3, R_{6,13}^3), \quad \theta_6 = \text{atan2}(R_{6,32}^3, -R_{6,31}^3)$$

This closed-form formulation eliminates singularity hunting and provides deterministic execution times suitable for real-time control loops.

---

## Perception & Grasp Planning Pipeline

### 1. Support Plane Segmentation (RANSAC)

Support fixtures (e.g., tabletops) are modeled as a plane $\Pi: \mathbf{n}^T \mathbf{p} + d = 0$ with constraints $\|\mathbf{n}\| = 1$ and normal alignment $|\mathbf{n} \cdot \hat{\mathbf{z}}| \ge \cos(15^\circ)$.

Candidate seed points are iteratively sampled to identify the maximum inlier consensus set:

$$\mathcal{I}^* = \arg\max_{\Pi_k} \sum_{i=1}^N \mathbb{I}\left(|\mathbf{n}_k^T \mathbf{p}_i + d_k| < \epsilon_{th}\right)$$

Points situated above the support plane plus a calibrated safety margin ($\delta_{safety} = 8\text{ mm}$) define the segmented target object cloud:

$$\mathcal{P}_{object} = \left\{ \mathbf{p}_i \in \mathcal{P} \setminus \mathcal{I}^* \;\middle|\; z_{table} + \delta_{safety} \le \mathbf{p}_{i, z} \le z_{max} \right\}$$

### 2. Principal Component Analysis (PCA) Refiner

For target cloud $\mathcal{P}_{object} = \{\mathbf{p}_i\}_{i=1}^M$, the sample covariance matrix $\mathbf{C}$ is calculated:

$$\mathbf{C} = \frac{1}{M} \sum_{i=1}^M (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{p}_i - \bar{\mathbf{p}})^T, \quad \bar{\mathbf{p}} = \frac{1}{M} \sum_{i=1}^M \mathbf{p}_i$$

Eigendecomposition yields ordered axes $\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k$ with $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$:
- $\mathbf{v}_1$: Primary axis of elongation.
- $\mathbf{v}_2$: Intermediate axis, defining the optimal closing aperture direction for parallel fingers.
- $\mathbf{v}_3$: Surface normal / approach direction.

The antipodal grasp yaw is aligned with the minimum cross-sectional variance:
$$\theta_{yaw} = \text{atan2}(\mathbf{v}_{2, y}, \mathbf{v}_{2, x})$$

### 3. Coulomb Friction Cone & Contact Evaluation

For contact points $\mathbf{c}_1, \mathbf{c}_2$ with inward surface normals $\mathbf{n}_1, \mathbf{n}_2$, force closure under Coulomb friction requires the contact line $\mathbf{c}_{12} = \frac{\mathbf{c}_2 - \mathbf{c}_1}{\|\mathbf{c}_2 - \mathbf{c}_1\|}$ to fall inside both friction cones:

$$\arccos\left(-\mathbf{n}_1^T \mathbf{c}_{12}\right) \le \arctan(\mu), \quad \arccos\left(\mathbf{n}_2^T \mathbf{c}_{12}\right) \le \arctan(\mu)$$

In physical rollouts, a grasp is recorded as successful only when the arm completes a full trajectory: descent $\to$ finger closure with $80\text{ N}$ force $\to$ vertical lift $\ge 5\text{ cm}$ without slip under gravitational acceleration ($g = 9.81\text{ m/s}^2$).

---

## Experimental Evaluation

We evaluated 11 grasp synthesis methods under identical physical conditions across 99 individual rollouts on standard benchmark CAD models (`duck`, `lego`, `block`, `clutter`) in PyBullet, combined with real-world Intel RealSense D435 sensor scans.

### Table 1: Comprehensive Method Leaderboard

Grasp success rate (GSR) and standard GraspNet-style Average Precision (AP) metrics are reported. AP is computed as the area under the Precision-Recall curve across all grasp candidates, with a candidate labelled positive when it passes Antipodal Force-Closure at the specified Coulomb friction coefficient $\mu$ (Fang et al., CVPR 2020).

| Method | Paradigm | GSR | AP₀.₄ | AP₀.₈ | AP | IK Feasibility | Table Collision | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA)** | Analytical Covariance | **77.8%** | **68.4%** | **81.2%** | **74.8%** | **100.0%** | **0.0%** | **0.3 ± 0.1 ms** |
| **Pure Geometric (SurfaceNormals)** | $k$-NN Antipodal Contact | **77.8%** | **67.1%** | **79.8%** | **73.5%** | **100.0%** | **0.0%** | 110.7 ± 292.9 ms |
| **Pure Geometric (OBB)** | Oriented Bounding Box | 66.7% | 58.3% | 71.4% | 64.9% | **100.0%** | **0.0%** | **0.4 ± 0.1 ms** |
| **Pure Geometric (CrossSectionSlice)** | Height Waist Slicing | 55.6% | 47.2% | 59.8% | 53.5% | **100.0%** | **0.0%** | **0.5 ± 0.1 ms** |
| **Hybrid (AnyGrasp + OBB)** | Neural Affordance + OBB Filter | 55.6% | 46.9% | 58.3% | 52.6% | **100.0%** | **0.0%** | 127.5 ± 126.3 ms |
| **Hybrid (AnyGrasp + PCA)** | Neural Affordance + PCA Filter | 44.4% | 37.1% | 47.6% | 42.4% | **100.0%** | **0.0%** | 98.4 ± 73.4 ms |
| **Hybrid (AnyGrasp + CrossSectionSlice)** | Neural Affordance + Slice Filter | 44.4% | 36.8% | 46.2% | 41.5% | **100.0%** | **0.0%** | 163.8 ± 190.9 ms |
| **Hybrid (AnyGrasp + SurfaceNormals)** | Neural Affordance + Normal Filter | 44.4% | 35.9% | 45.1% | 40.5% | **100.0%** | **0.0%** | 304.1 ± 629.5 ms |
| **Pure Geometric (PrimitiveRANSAC)** | Analytic Model Fitting | 33.3% | 28.4% | 36.1% | 32.3% | **100.0%** | **0.0%** | 2.3 ± 1.5 ms |
| **Hybrid (AnyGrasp + PrimitiveRANSAC)** | Neural Affordance + RANSAC Filter | 22.2% | 18.7% | 24.3% | 21.5% | **100.0%** | **0.0%** | 993.5 ± 1774.2 ms |
| **Pure AnyGrasp (Baseline)** | End-to-End GSNet | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 66.7% | 132.6 ± 114.9 ms |

### Table 2: Performance Breakdown by Target Object

| Method | Block (`box`) | Lego (`cube_cluster`) | Duck (`freeform_mesh`) | Clutter (`multi_body`) | RealSense Scan | Mean GSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA)** | 100.0% (2/2) | 100.0% (2/2) | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/1) | **77.8%** |
| **Pure Geometric (SurfaceNormals)** | 100.0% (2/2) | 100.0% (2/2) | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/1) | **77.8%** |
| **Pure Geometric (OBB)** | 100.0% (2/2) | 100.0% (2/2) | 50.0% (1/2) | 50.0% (1/2) | 0.0% (0/1) | **66.7%** |
| **Pure Geometric (CrossSectionSlice)** | 100.0% (2/2) | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/2) | 0.0% (0/1) | **55.6%** |
| **Hybrid (AnyGrasp + OBB)** | 100.0% (2/2) | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/2) | 0.0% (0/1) | **55.6%** |
| **Hybrid (AnyGrasp + PCA)** | 100.0% (2/2) | 100.0% (2/2) | 0.0% (0/2) | 0.0% (0/2) | 0.0% (0/1) | **44.4%** |
| **Hybrid (AnyGrasp + SurfaceNormals)** | 100.0% (2/2) | 100.0% (2/2) | 0.0% (0/2) | 0.0% (0/2) | 0.0% (0/1) | **44.4%** |
| **Pure AnyGrasp (Baseline)** | 0.0% (0/2) | 0.0% (0/2) | 0.0% (0/2) | 0.0% (0/2) | 0.0% (0/1) | **0.0%** |

### Table 3: Comparison with Existing Frameworks

| Characteristic | Dex-Net 2.0 / 4.0 | Contact-GraspNet | AnyGrasp Baseline | ARMROBOT (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **Manipulator Articulation** | 4-DOF planar overhead | Unconstrained 6-DOF | Unconstrained 6-DOF | **6-DOF with Spherical Wrist** |
| **Camera Configuration** | Static overhead sensor | Static single sensor | Static single view | **Active Dual Dynamic (Wrist + Turret)** |
| **IK Formulation** | Numerical optimization | Numerical / OpenRAVE | External / Unintegrated | **Analytical Decoupled ($< 0.005\text{ mm}$)** |
| **Support Boundary Handling** | Heuristic depth cutoff | None | None | **RANSAC Analytic Plane + Margin** |
| **Tabletop Collision Rate** | Low (Planar only) | High in clutter | 66.7% | **0.0% across all geometric/hybrid** |
| **Edge Compute (CPU)** | Moderate (CNN forward pass) | No (GPU PyTorch) | No (GPU CUDA/Minkowski) | **Yes (0.3 ms single-thread CPU)** |
| **Physical Validation** | Quasi-static wrench | Static wrench metrics | Synthetic labels | **PyBullet Multi-Body Contact Dynamics** |
| **Middleware Integration** | ROS 1 Indigo/Kinetic | Standalone Python | Standalone Python | **Native ROS 2 Jazzy & Gazebo Harmonic** |

### Key Observations

1. **Root Cause of Pure Neural Failure**: End-to-end 6-DOF grasp networks predict candidates directly in point cloud coordinate space without geometric support constraints. As a result, $66.7\%$ of raw AnyGrasp candidates penetrated the tabletop surface or required subterranean approaches, producing collision aborts.
2. **Impact of Geometric Post-Filtering**: Passing neural candidates through analytical geometric bounds (RANSAC table clamping and OBB aperture validation) reduced collisions to $0.0\%$ and raised the grasp success rate to $55.6\%$.
3. **Execution Latency on Embedded Hardware**: Pure geometric PCA achieves $0.3\text{ ms}$ latency on a standard CPU thread—over $400\times$ faster than neural network inference on GPU ($132.6\text{ ms}$)—while offering deterministic behavior and higher physical success ($77.8\%$ vs. $0.0\%$).

---

## Architecture

```
                                  +---------------------------------------------+
                                  |         Gazebo Harmonic Simulation          |
                                  |  - 6-DOF Manipulator with Spherical Wrist   |
                                  |  - Franka Emika Parallel Gripper            |
                                  |  - Active Dual Dynamic RGB-D Cameras        |
                                  |    • /camera/wrist/points (Eye-in-Hand)     |
                                  |    • /camera/arm/points   (Turret Tracking) |
                                  |  - Tabletop Workspace with CAD Objects      |
                                  +----------------------+----------------------+
                                                         |
                                 Synchronized Point Clouds & Joint States (/clock synced)
                                                         v
+-----------------------------+        +-----------------------------------------------+
| AnyGrasp Server             |        | ROS 2 Jazzy Manipulation Pipeline             |
| (Python 3.10 / Conda)       |<------>|                                               |
| - MinkowskiEngine 3D        |  JSON  | - grasp_detection_node                        |
| - 6-DOF Grasp Synthesis     | Socket |   • Voxel Downsampling (5mm)                  |
| - Score & Width Inference   |  IPC   |   • RANSAC Support Plane Segmentation         |
+-----------------------------+        |   • Geometric Refiners (PCA, OBB, Normals)    |
                                       |   • Safety Boundary Clamping                  |
                                       |                                               |
                                       | - pick_place_node (State Machine)             |
                                       |   • Analytical 6-DOF Decoupled IK             |
                                       |   • S-Curve Trajectory Interpolation          |
                                       |   • Action Execution (Approach, Grasp, Lift)  |
                                       +-----------------------+-----------------------+
                                                               |
                                                 RViz2 Markers & JointTrajectory
                                                               v
                                       +-----------------------------------------------+
                                       | ros2_control Hardware Interface               |
                                       | - joint_trajectory_controller (6-DOF Arm)     |
                                       | - gripper_controller (Franka Gripper)         |
                                       +-----------------------------------------------+
```

---

## Installation & Setup

### Prerequisites

- **Ubuntu 24.04 LTS**
- **ROS 2 Jazzy Jalisco**
- **Gazebo Harmonic (`gz-sim8`)**
- Python 3.12 (system) & Python 3.10 (optional Conda environment for AnyGrasp)

```bash
# 1. Install ROS 2 Jazzy and Gazebo Harmonic packages
sudo apt update && sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher

# 2. Install Python dependencies
pip install numpy scipy open3d pybullet pytest
```

### Build Workspace

```bash
# Clone repository
git clone https://github.com/TLE-WTU/ARMROBOT.git
cd ARMROBOT

# Source ROS 2 environment and build
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_arm --symlink-install
source install/setup.bash
```

---

## Usage

### 1. Run Automated Unit Tests (78/78 Passing)

Verify kinematics, perception filters, and physics metrics:
```bash
PYTHONPATH=src/robot_arm:. pytest tests/ -v
```

### 2. Launch Simulation

```bash
# Launch Gazebo Harmonic with 6-DOF arm and dual cameras
ros2 launch robot_arm gazebo.launch.py dof:=6

# Inspect dual dynamic point cloud topics in separate terminals:
ros2 topic hz /camera/wrist/points    # Eye-in-Hand camera
ros2 topic hz /camera/arm/points      # Turret tracking camera
```

### 3. Run Autonomous Pick-and-Place

```bash
# Pure Geometric Pipeline (100% CPU, Real-Time < 1 ms)
ros2 launch robot_arm grasp_demo.launch.py dof:=6

# Hybrid Deep Learning Pipeline (requires AnyGrasp server running)
ros2 launch robot_arm grasp_demo.launch.py dof:=6 use_anygrasp:=true

# Benchmark scenarios:
ros2 launch robot_arm grasp_demo.launch.py dof:=6 test_scenario:=dense_clutter
ros2 launch robot_arm grasp_demo.launch.py dof:=6 test_scenario:=transparent_bottle
ros2 launch robot_arm grasp_demo.launch.py dof:=6 test_scenario:=flat_object
```

### 4. Run Academic Benchmark Suite

```bash
# Quick validation run (2 trials across 5 benchmark objects)
python3 benchmark/run_academic_benchmark.py --trials 2

# Full evaluation with 3D GUI visualization
python3 benchmark/run_academic_benchmark.py --trials 5 --gui

# Compute GraspNet-style AP metrics (AP_0.4, AP_0.8, AP_seen, AP_novel)
python3 -m benchmark.eval_ap --csv benchmark/academic_benchmark_results.csv
```

Results are saved to `benchmark/academic_benchmark_results.csv` and summarized in `benchmark/ACADEMIC_BENCHMARK_REPORT.md`.

---

## Configuration Reference

Key geometric, physical, and perception parameters are centralized in [`src/robot_arm/config/robot_params.yaml`](src/robot_arm/config/robot_params.yaml):

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `dof` | int | `6` | Kinematic degrees of freedom (`3`, `4`, `5`, or `6`) |
| `link_lengths.base_height` | float | `0.150 m` | Base turret height above ground |
| `link_lengths.link2_length` | float | `0.250 m` | Upper arm length |
| `link_lengths.link3_length` | float | `0.200 m` | Forearm length |
| `link_lengths.hand_length` | float | `0.140 m` | Spherical wrist offset to Franka gripper contact center |
| `table.height_world` | float | `0.250 m` | Table surface elevation in world frame |
| `table.safety_margin` | float | `0.008 m` | Collision clearance floor offset above table surface |
| `gripper.max_width` | float | `0.070 m` | Maximum Franka parallel-jaw aperture |
| `gripper.max_effort` | float | `80.0 N` | Dynamic bilateral grip force |
| `perception.voxel_size` | float | `0.005 m` | Voxel downsampling grid size |
| `perception.ransac_distance_threshold` | float | `0.008 m` | RANSAC residual threshold for table support inliers |
| `perception.ransac_max_iterations` | int | `150` | Maximum RANSAC trials |
| `pick_place.pre_grasp_offset_z` | float | `0.060 m` | Approach height offset |
| `pick_place.post_grasp_lift_z` | float | `0.080 m` | Vertical lift clearance under gravity |

---

## Repository Structure

```
ARMROBOT/
├── .github/workflows/ci.yaml       # GitHub Actions CI (Linting, Tests, Coverage, ROS 2 Build)
├── benchmark/                      # Grasp benchmarking suite
│   ├── metrics.py                  # Force-closure, collision, aperture metrics + AP bridge
│   ├── eval_ap.py                  # GraspNet-style AP evaluation (AP_0.4, AP_0.8, AP_seen, AP_novel)
│   ├── physics_environment.py      # PyBullet dynamic rollout environment
│   ├── dataset_loader.py           # Intel RealSense D435 RGB-D loader
│   ├── evaluator.py                # 11-method head-to-head evaluation engine
│   ├── run_academic_benchmark.py   # Benchmark runner
│   └── academic_benchmark_results.csv
├── media/                          # Technical diagrams and simulation figures
│   ├── full_robot_preview.png      # 6-DOF kinematic assembly and annotations
│   ├── sim_grasp_preview.png       # PyBullet simulation rollout view
│   ├── franka_gripper_preview.png  # Franka gripper CAD render
│   └── gazebo_window.png           # Tabletop workspace in Gazebo
├── src/robot_arm/                  # Unified ROS 2 manipulation package
│   ├── config/                     # Controller and robot YAML configurations
│   ├── launch/                     # Gazebo and pick-and-place launch scripts
│   ├── meshes/                     # STL and DAE mesh models
│   ├── robot_arm/                  # Core Python nodes and algorithmic modules
│   │   ├── ik_solver.py            # Analytical decoupled 6-DOF IK solver
│   │   ├── perception_pipeline.py  # RANSAC plane segmentation and safety checks
│   │   ├── geometric_refinement.py # 5 modular geometric refiners
│   │   ├── grasp_detection_node.py # Perception ROS 2 wrapper
│   │   ├── pick_place_node.py      # Motion orchestrator and state machine
│   │   └── anygrasp_service.py     # JSON socket IPC bridge for AnyGrasp
│   ├── urdf/                       # Parameterized URDF Xacro descriptions
│   └── worlds/                     # Gazebo simulation world SDFs
├── tests/                          # Automated unit test suite (78 tests)
├── RUN_GUIDE.md                    # Developer operational manual
└── LICENSE                         # MIT License
```

---

## Citation

```bibtex
@misc{armrobot2026,
  author       = {Tien Le and Contributors},
  title        = {{ARMROBOT: 6-DOF Robotic Manipulation Framework with Active Dual-Camera Perception and Geometric Grasp Synthesis}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/TLE-WTU/ARMROBOT}},
  version      = {3.0.0}
}
```

## License

This project is released under the [MIT License](LICENSE).
