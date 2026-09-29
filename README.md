# 🤖 ARMROBOT — 5-DoF Robotic Arm with AI Grasp Synthesis & Perception Pipeline

An autonomous Pick-and-Place robotics framework featuring a **3/4/5-DoF Articulated Robotic Arm** with a parallel-jaw gripper, simulated in **ROS 2 Jazzy** and **Gazebo Harmonic**, integrated with **AnyGrasp** (Deep Learning 6-DoF grasp synthesis) and an analytical **RANSAC Tabletop Segmentation & Adaptive Geometric Reduction** perception pipeline.

> **v2.0** — Unified architecture with parameterized DOF support, SO-ARM100 mesh integration, and improved code quality.

https://github.com/user-attachments/assets/demo_5dof_grasp (or see [`media/demo_5dof_grasp.webm`](media/demo_5dof_grasp.webm))

---

## ✨ Key Highlights

- **Unified Multi-DOF Architecture**: Single codebase supporting 3, 4, and 5 DOF via `dof` parameter
- **3D Mesh Visualization**: Integrated [SO-ARM100](https://github.com/JafarAbdi/ros2_so_arm100) high-fidelity STL meshes (Apache 2.0 License)
- **Analytical IK Solver**: Class-based solver with trajectory pre-validation and configurable link lengths
- **RANSAC Perception Pipeline**: Tabletop plane segmentation + PCA-based grasp yaw extraction
- **AnyGrasp AI Integration**: Optional zero-shot 6-DoF grasp detection via secure JSON-based IPC
- **S-Curve Trajectory Interpolation**: Smooth cosine profiles eliminating jerk
- **Benchmark System**: CSV-based ablation study framework with diagnostic failure analysis
- **CI/CD Pipeline**: GitHub Actions with linting, unit tests, and ROS 2 build verification

## 🏗️ System Architecture

```
                                  +---------------------------+
                                  |  Gazebo Harmonic (Sim)    |
                                  |  - Overhead RGB-D Camera  |
                                  |  - N-DoF Arm + Gripper    |
                                  |  - 3D Benchmark Objects   |
                                  +-------------+-------------+
                                                |
                     Camera Depth/RGB & Joint State Topics (/clock synced)
                                                v
+------------------------+        +---------------------------+
| AnyGrasp Server        |        | ROS 2 Jazzy Core Nodes    |
| (Conda: Python 3.10)  |<------>| (Python 3.12)             |
| - MinkowskiEngine      |  JSON  | - grasp_detection_node    |
| - 6-DoF Grasp Network  |  IPC   |   • RANSAC Plane Seg      |
| - Score/Width/Depth    | Socket |   • Adaptive Reduction    |
+------------------------+        |   • PCA Yaw Orientation   |
                                  | - pick_place_node (FSM)   |
                                  |   • Analytical N-DoF IK   |
                                  |   • S-Curve Interpolation |
                                  |   • Trajectory Validation |
                                  +-------------+-------------+
                                                |
                                        RViz2 Markers &
                                  ros2_control Joint Commands
```

---

## 📁 Repository Structure

```
ARMROBOT/
├── .github/workflows/ci.yaml    # GitHub Actions CI pipeline
├── README.md
├── RUN_GUIDE.md
├── src/
│   └── robot_arm/               # Unified package (replaces robot_3dof/4dof/5dof)
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── config/
│       │   ├── robot_params.yaml        # Centralized parameters
│       │   ├── anygrasp_params.yaml     # AnyGrasp config
│       │   ├── controllers_3dof.yaml
│       │   ├── controllers_4dof.yaml
│       │   └── controllers_5dof.yaml
│       ├── launch/
│       │   ├── gazebo.launch.py         # Parameterized: dof:=3|4|5
│       │   └── grasp_demo.launch.py     # Full demo launch
│       ├── meshes/
│       │   ├── arm/                     # SO-ARM100 STL meshes
│       │   └── objects/                 # Benchmark objects (OBJ)
│       ├── robot_arm/
│       │   ├── ik_solver.py                 # Multi-DOF IK solver
│       │   ├── perception_pipeline.py       # Pure Python perception (RANSAC, PCA, safety floor)
│       │   ├── geometric_refinement.py      # Modular refiners (PCA, OBB, normals, slice, cylinder)
│       │   ├── grasp_detection_node.py      # ROS 2 Perception wrapper node
│       │   ├── pick_place_node.py           # FSM orchestrator
│       │   └── anygrasp_service.py          # Secure JSON IPC bridge
│       ├── urdf/
│       │   ├── robot_arm.urdf.xacro         # Parameterized by DOF
│       │   ├── robot_arm_gazebo.xacro
│       │   └── robot_arm_ros2_control.xacro
│       └── worlds/                          # Gazebo SDF worlds
└── tests/
    ├── test_ik_solver.py
    ├── test_grasp_detection.py
    └── test_geometric_refinement.py
```

---

## 🚀 Quick Start

### Prerequisites

- **Ubuntu 24.04** with ROS 2 Jazzy
- **Gazebo Harmonic** (gz-sim8)
- Python packages: `numpy`, `scipy`
- Optional: AnyGrasp SDK (requires NVIDIA GPU + CUDA)

### Build

```bash
# Clone repository
git clone https://github.com/TLE-WTU/ARMROBOT.git
cd ARMROBOT

# Build with colcon
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_arm
source install/setup.bash
```

### Run Simulation

```bash
# Launch Gazebo with 5-DOF arm (default)
ros2 launch robot_arm gazebo.launch.py

# Or specify DOF
ros2 launch robot_arm gazebo.launch.py dof:=3
ros2 launch robot_arm gazebo.launch.py dof:=4
ros2 launch robot_arm gazebo.launch.py dof:=5
```

### Run Grasp Demo

```bash
# Full pick-and-place demo with RANSAC perception
ros2 launch robot_arm grasp_demo.launch.py dof:=5

# With AnyGrasp AI
ros2 launch robot_arm grasp_demo.launch.py dof:=5 use_anygrasp:=true

# Benchmark scenarios
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=transparent_bottle
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=flat_object
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=dense_clutter
```

---

## 🧪 Testing

```bash
# Run automated test suite (37 tests)
python3 -m pytest tests/ -v

# Lint check
flake8 src/robot_arm/robot_arm/ --max-line-length=120
```

---

## 🔧 Configuration

All robot parameters are centralized in [`config/robot_params.yaml`](src/robot_arm/config/robot_params.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dof` | 5 | Degrees of freedom (3, 4, or 5) |
| `table.height_world` | 0.250 m | Table surface height in world frame |
| `gripper.max_width` | 0.07 m | Maximum Franka gripper opening |
| `perception.voxel_size` | 0.005 m | Voxel grid size for downsampling |
| `perception.ransac_max_iterations` | 150 | RANSAC iteration count (with early stopping) |

---

## 🏆 Geometric Benchmark & Algorithmic Comparison

Evaluated on **147 test cases** across 7 object geometries (`mug`, `duck`, `torus`, `glass_bottle`, `box_package`, `flat_disc`, `dense_clutter`) and 3 point-cloud noise/sparsity conditions:

| Rank | Method | Success Rate | Table Collision | Centering Error | Normal Alignment | Latency | Composite Score (CPI) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Pure Heuristic** *(Table RANSAC + PCA)* | **100.0%** | **0.0%** | **9.1 mm** | **0.705** | **0.18 ms** | **89.5 / 100** |
| 🥈 | **AnyGrasp + PrimitiveRANSAC** | **76.2%** | **14.3%** | **13.4 mm** | **0.573** | **25.59 ms** | **69.0 / 100** |
| 🥉 | **AnyGrasp + OBB** *(Oriented BBox)* | **76.2%** | **14.3%** | **30.2 mm** | **0.574** | **1.79 ms** | **65.3 / 100** |
| 4 | **AnyGrasp + PCA** | 76.2% | 14.3% | 28.1 mm | 0.414 | 1.29 ms | 63.2 / 100 |
| 5 | **AnyGrasp + CrossSectionSlice** | 76.2% | 14.3% | 32.3 mm | 0.501 | 3.64 ms | 62.4 / 100 |
| 6 | **AnyGrasp + SurfaceNormals** | 66.7% | 23.8% | 31.8 mm | 0.431 | 125.35 ms | 47.7 / 100 |
| 7 | **Pure AnyGrasp** *(Baseline DL)* | 66.7% | 23.8% | 31.8 mm | 0.214 | 52.27 ms | **43.4 / 100** |

Run the automated benchmark suite:
```bash
python3 scripts/run_geometric_benchmark.py
```

Detailed report: [`benchmark_report.md`](benchmark_report.md) | Full dataset: [`benchmark_results_geometric_comparison.csv`](benchmark_results_geometric_comparison.csv)

---

## 🙏 Acknowledgments

- **SO-ARM100 Meshes**: [ros-physical-ai/ros2_so_arm](https://github.com/JafarAbdi/ros2_so_arm100) — Apache 2.0 License
- **Franka Emika Hand**: Modern industrial parallel-jaw gripper mesh integration
- **AnyGrasp**: [graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)
- Built with **ROS 2 Jazzy**, **Gazebo Harmonic**, and **ros2_control**
- Special thanks to AnyGrasp, Gazebo Harmonic, and RViz2 open-source communities

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
