# Operation Guide: ARMROBOT 6-DOF Manipulation & Perception Pipeline

Operational and execution manual for the 6-DOF robotic arm with dual active cameras (Eye-in-Hand + Turret Tracking) and grasp synthesis pipeline.

---

## 1. System Requirements & Environment

### Operating System & Core Frameworks
- **OS**: Ubuntu 24.04 LTS (Noble Numbat)
- **Middleware**: ROS 2 Jazzy Jalisco (Desktop Install)
- **Physics Simulator**: Gazebo Harmonic (`gz-sim8`) & PyBullet Physics Engine
- **Languages**: Python 3.12 (ROS 2 nodes) & Python 3.10 (Conda environment for AnyGrasp / MinkowskiEngine)

### System Packages Installation
```bash
# 1. Core ROS 2 Jazzy & Gazebo Harmonic integration
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher

# 2. Python spatial math dependencies
pip install numpy scipy open3d pybullet
```

---

## 2. Workspace Build

```bash
cd ~/ARMROBOT

# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Build unified robot_arm package with symlink install
colcon build --packages-select robot_arm --symlink-install

# Source package environment
source install/setup.bash
```

> **Note:** If building with an active Conda environment returns `catkin_pkg` errors, force the system Python executable:
> ```bash
> colcon build --packages-select robot_arm --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
> ```

---

## 3. Simulation & Execution Commands

### 3.1 Gazebo Harmonic 3D Simulation
Launch the simulated 6-DOF robot arm with Franka Emika parallel-jaw gripper and dual dynamic cameras:

```bash
# Standard 6-DOF arm (Default)
ros2 launch robot_arm gazebo.launch.py dof:=6

# Backward-compatible multi-DOF configurations:
ros2 launch robot_arm gazebo.launch.py dof:=3
ros2 launch robot_arm gazebo.launch.py dof:=4
ros2 launch robot_arm gazebo.launch.py dof:=5

# Specific benchmark environments:
ros2 launch robot_arm gazebo.launch.py dof:=6 world:=pick_and_place_clutter
ros2 launch robot_arm gazebo.launch.py dof:=6 world:=pick_and_place_bottle
ros2 launch robot_arm gazebo.launch.py dof:=6 world:=pick_and_place_flat
```

### 3.2 Dual Dynamic Camera Topic Inspection
Verify that both dynamic perception feeds are actively broadcasting point clouds and RGB-D images:

```bash
# Camera 1: Eye-in-Hand camera on gripper
ros2 topic hz /camera/wrist/points
ros2 topic echo /camera/wrist/points --once

# Camera 2: Turret tracking camera on rotating base
ros2 topic hz /camera/arm/points
ros2 topic echo /camera/arm/points --once
```

### 3.3 Autonomous Pick-and-Place Demo
Execute the full perception-to-motion control loop:

```bash
# 1. Geometric Pipeline (100% CPU, Real-Time < 1 ms)
ros2 launch robot_arm grasp_demo.launch.py dof:=6

# 2. Hybrid AI Pipeline with AnyGrasp (requires AnyGrasp service running)
ros2 launch robot_arm grasp_demo.launch.py dof:=6 use_anygrasp:=true

# 3. Benchmark failure scenario injection:
ros2 launch robot_arm grasp_demo.launch.py dof:=6 test_scenario:=dense_clutter
ros2 launch robot_arm grasp_demo.launch.py dof:=6 test_scenario:=transparent_bottle
ros2 launch robot_arm grasp_demo.launch.py dof:=6 test_scenario:=flat_object
```

---

## 4. AnyGrasp Deep Learning Service (Optional GPU Pipeline)

AnyGrasp requires Python 3.10 and MinkowskiEngine on an NVIDIA GPU environment:

```bash
# Activate conda environment
conda activate robot_env

# Start AnyGrasp IPC Server
python src/robot_arm/robot_arm/anygrasp_service.py
```

The service establishes a Unix Domain Socket at `/tmp/anygrasp_ipc.sock` using JSON + base64 serialization for zero-copy point cloud transfer.

---

## 5. Automated 11-Method Academic Benchmark

Run the standardized, peer-reviewed benchmark comparing all 5 Geometric algorithms, Pure AnyGrasp AI, and 5 Hybrid combinations under identical physical conditions:

```bash
# Quick validation run (2 trials across all 4 benchmark objects)
python3 benchmark/run_academic_benchmark.py --trials 2

# Full rigorous evaluation (5 trials with GUI visualization)
python3 benchmark/run_academic_benchmark.py --trials 5 --gui
```

### Generated Artifacts
- **Leaderboard & Report**: [`benchmark/ACADEMIC_BENCHMARK_REPORT.md`](benchmark/ACADEMIC_BENCHMARK_REPORT.md) (or [`benchmark_report.md`](benchmark_report.md))
- **Raw Telemetry Dataset**: [`benchmark/academic_benchmark_results.csv`](benchmark/academic_benchmark_results.csv)

---

## 6. Automated Unit Tests

Run the full regression test suite (55 unit tests covering kinematics, perception, safety filters, and benchmark metrics):

```bash
PYTHONPATH=src/robot_arm:. pytest tests/ -v
```

---

## 7. Troubleshooting Matrix

| Symptom / Error | Root Cause | Verified Solution |
| :--- | :--- | :--- |
| `Gazebo Harmonic crashes on start` | Incomplete installation of `gz-sim8` | Run `sudo apt install gz-harmonic ros-jazzy-ros-gz` |
| `Controllers not active / unconfigured` | Controller manager spawned before Gazebo bridge | Run `source install/setup.bash` and ensure `ros2 control list_controllers` shows active |
| `AnyGrasp license feature id mismatch` | Virtual network adapter (e.g. USB tethering) altered hardware MAC | `run_academic_benchmark.py` and `anygrasp_service.py` synchronize hardware MACs automatically |
| `Point cloud topic empty` | Sensor plugin frame mismatch in Gazebo | Verify `/camera/wrist/points` with `ros2 topic info /camera/wrist/points` |
| `Target unreachable D out of bounds` | Approach vector inversion in 6-DOF IK | Updated `IKSolver(dof=6)` uses analytical wrist decoupling; verify coordinates within workspace $R \le 0.45\text{ m}$ |
| `colcon build catkin_pkg error` | Active Conda environment hijacking Python path | Add `-DPython3_EXECUTABLE=/usr/bin/python3` to `colcon build` |
