# 🚀 Run Guide — ARMROBOT v2.0

## Prerequisites

### System Requirements
- **Ubuntu 24.04 LTS** (Noble Numbat)
- **ROS 2 Jazzy** (desktop install)
- **Gazebo Harmonic** (gz-sim8)
- Python 3.12+

### Install ROS 2 Jazzy
```bash
# Follow official instructions: https://docs.ros.org/en/jazzy/Installation.html
sudo apt install ros-jazzy-desktop
```

### Install Gazebo Harmonic & ROS 2 Bridge
```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-ros2-control ros-jazzy-ros2-controllers
sudo apt install ros-jazzy-xacro ros-jazzy-robot-state-publisher
```

### Install Python Dependencies
```bash
pip install numpy scipy
```

---

## Build

```bash
cd ARMROBOT
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_arm --symlink-install
source install/setup.bash
```

---

## Launch Commands

### Basic Gazebo Simulation
```bash
# 5-DOF (default)
ros2 launch robot_arm gazebo.launch.py

# Specify DOF
ros2 launch robot_arm gazebo.launch.py dof:=3
ros2 launch robot_arm gazebo.launch.py dof:=5

# Specify world
ros2 launch robot_arm gazebo.launch.py world:=pick_and_place_bottle
```

### Full Grasp Demo (Perception + Pick-and-Place)
```bash
# Default mode with RANSAC perception
ros2 launch robot_arm grasp_demo.launch.py dof:=5

# With AnyGrasp AI (requires separate AnyGrasp service)
ros2 launch robot_arm grasp_demo.launch.py dof:=5 use_anygrasp:=true

# Disable RANSAC for baseline comparison
ros2 launch robot_arm grasp_demo.launch.py dof:=5 enable_ransac:=false
```

### Benchmark Scenarios
```bash
# Transparent bottle (optical refraction ghost grasps)
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=transparent_bottle

# Ultra-thin flat object (low affordance)
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=flat_object

# Dense clutter (adjacent object collision)
ros2 launch robot_arm grasp_demo.launch.py test_scenario:=dense_clutter
```

---

## AnyGrasp AI Setup (Optional)

AnyGrasp requires a separate Conda environment with Python 3.10:

```bash
# Create conda environment
conda create -n anygrasp python=3.10
conda activate anygrasp
pip install torch torchvision MinkowskiEngine

# Start the AnyGrasp IPC service
python src/robot_arm/robot_arm/anygrasp_service.py
```

The service communicates with the ROS 2 nodes via a JSON-encoded Unix domain socket at `/tmp/anygrasp_ipc.sock`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Gazebo crashes on start | Ensure `gz-sim8` is installed: `sudo apt install gz-harmonic` |
| Controllers not found | Run `source install/setup.bash` after building |
| Mesh not loading in Gazebo | Verify `GZ_SIM_RESOURCE_PATH` includes the install directory |
| AnyGrasp socket not found | Start `anygrasp_service.py` in conda environment first |
| Point cloud empty | Check camera topic: `ros2 topic echo /camera/points --once` |
| `ModuleNotFoundError: No module named 'catkin_pkg'` | If Conda is activated, build using: `colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3` |

