# ARMROBOT Execution Guide

Step-by-step instructions for building, running simulations, and executing benchmark evaluations.

## 1. Prerequisites

### System Requirements
- Ubuntu 24.04 LTS
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic (`ros_gz_sim`)
- Python 3.12

### Install ROS 2 Packages
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

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Build Workspace

```bash
cd ~/ARMROBOT

# Source ROS 2 underlay
source /opt/ros/jazzy/setup.bash

# Build package
colcon build --symlink-install --packages-select robot_arm --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

# Source overlay
source install/setup.bash
```

---

## 3. Running Simulations

### 3.1 Launch Gazebo Simulation
Starts Gazebo Harmonic with the 4-DOF robot arm, ros2_control hardware interface, and overhead RGB-D camera sensor:
```bash
ros2 launch robot_arm gazebo.launch.py
```

### 3.2 Autonomous Pick-and-Place Demo

**Option A: Heuristic Geometric Mode (CPU only, no GPU needed)**
```bash
ros2 launch robot_arm grasp_demo.launch.py use_anygrasp:=false grasp_mode:=heuristic
```

**Option B: Hybrid Mode (AnyGrasp + Geometric Ensemble)**
Requires AnyGrasp checkpoint and PyTorch environment:
```bash
ros2 launch robot_arm grasp_demo.launch.py use_anygrasp:=true grasp_mode:=hybrid
```

**Testing Clutter & Adversarial Scenarios:**
```bash
# Translucent bottle scenario
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=transparent_bottle

# Dense clutter scenario
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=dense_clutter

# Flat object scenario
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=flat_object
```

---

## 4. Running Benchmarks

### 4.1 Headless Physics Grasp Evaluation (PyBullet)
Runs physical rollouts on standard CAD shapes (cube, cylinder, mug, duck, clutter) using PyBullet:
```bash
python3 benchmark/run_academic_benchmark.py --headless --output_dir ./benchmark
```

### 4.2 Analytical Geometric Benchmark
Evaluates the 5 geometric refinement algorithms against the dataset:
```bash
python3 scripts/run_geometric_benchmark.py
```

---

## 5. Running Tests

```bash
PYTHONPATH=./:src/robot_arm pytest -v tests/
```
