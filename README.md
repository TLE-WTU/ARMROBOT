# 🤖 3-DoF Robotic Arm with AnyGrasp Deep Learning AI Grasp Synthesis

A complete autonomous Pick-and-Place robotics framework featuring a **3-DoF Articulated Robotic Arm** with a parallel-jaw gripper, simulated in **ROS 2 Jazzy** and **Gazebo Harmonic**, integrated with **AnyGrasp** (Deep Learning 6-DoF grasp synthesis with MinkowskiEngine & PyTorch) via high-speed IPC bridge.

---

## 🌟 Key Highlights

- **Inverse Kinematics & Smooth Motion**: Custom analytical Inverse Kinematics (IK) with cubic S-curve trajectory interpolation to eliminate joint velocity spikes and ensure smooth acceleration/deceleration.
- **AnyGrasp Deep Learning Integration**: Seamless zero-shot 6-DoF grasp detection directly from dense RGB-D point clouds using 3D sparse convolutions (MinkowskiEngine).
- **Cross-Environment IPC Bridge**: UNIX domain socket bridge enabling ROS 2 Jazzy (running modern Python 3.12) to communicate transparently with AnyGrasp's PyTorch/CUDA environment (Python 3.10).
- **Realistic 3D Complex Meshes**: Includes 3D models with irregular geometry (Coffee Mug with handle, Rubber Duck, Torus ring, cylinders, and blocks) for robust grasp benchmarking.
- **Rich 3D RViz & Terminal Visualization**:
  - Real-time 3D gripper wireframe visualization (`visualization_msgs/Marker` `LINE_LIST`) showing grasp jaw width, opening, and orientation.
  - 3D Floating HUD text labels displaying grasp confidence scores.
  - Segmented workspace point clouds (`sensor_msgs/PointCloud2`).
  - Terminal ASCII dashboard summarizing candidate grasp metrics and execution status.
- **Industrial Mechanical Design Specifications**: Complete link lengths, mass distribution, static/dynamic torque calculations, and hardware actuator recommendations included.

---

## 🏗️ System Architecture

```
                                +---------------------------+
                                |  Gazebo Harmonic (Sim)    |
                                |  - RGB-D Depth Camera     |
                                |  - 3-DoF Arm + Gripper    |
                                |  - 3D Objects in World    |
                                +-------------+-------------+
                                              |
                   Camera Depth/RGB & Joint State Topics
                                              v
+------------------------+      +---------------------------+
| AnyGrasp Server        |      | ROS 2 Jazzy Core Nodes    |
| (Conda: Python 3.10)   |<---->| (Python 3.12)             |
| - MinkowskiEngine      | IPC  | - grasp_detection_node    |
| - Score/Width/Depth    |Socket| - pick_place_node (FSM)   |
| - Collision Detection  |      | - ik_solver (S-Curve)     |
+------------------------+      +-------------+-------------+
                                              |
                                      RViz2 3D Markers &
                                      Trajectory Commands
```

---

## 📁 Repository Structure

```
robot_3dof_ws/
├── .gitignore                      # Excludes build artifacts, caches, and weights > 100MB
├── README.md                       # Documentation and startup guide
└── src/
    └── robot_3dof/
        ├── CMakeLists.txt
        ├── package.xml
        ├── config/
        │   ├── anygrasp_params.yaml    # Grasp thresholds, camera intrinsics, workspace limits
        │   └── controllers.yaml        # ros2_control joint trajectory configuration
        ├── launch/
        │   ├── gazebo.launch.py        # World, robot spawning, ros2_control managers
        │   └── grasp_demo.launch.py    # Main launch entrypoint (Gazebo + Nodes + RViz)
        ├── meshes/
        │   └── objects/
        │       ├── duck.obj            # 3D Rubber Duck mesh
        │       ├── mug.obj             # 3D Coffee Mug with handle
        │       └── torus.obj           # 3D Torus ring
        ├── robot_3dof/
        │   ├── __init__.py
        │   ├── anygrasp_service.py     # IPC Bridge Service client & server
        │   ├── grasp_detection_node.py # PointCloud processor & AnyGrasp communicator
        │   ├── ik_solver.py            # Analytical IK & S-curve trajectory generator
        │   └── pick_place_node.py      # Autonomous Pick-and-Place State Machine
        ├── rviz/
        │   └── config.rviz             # Pre-configured RViz display layout
        ├── urdf/
        │   ├── robot_3dof.urdf.xacro   # Robot arm & gripper kinematics/inertia
        │   ├── robot_3dof_gazebo.xacro # Gazebo Sim plugins & sensor attachments
        │   └── robot_3dof_ros2_control.xacro # Hardware interface for ros2_control
        └── worlds/
            └── pick_and_place.sdf      # Gazebo world with ground, table, lighting & 3D objects
```

---

## ⚙️ Mechanical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Link 0 (Base height)** | $175\text{ mm}$ | Distance from ground mount to shoulder axis |
| **Link 1 (Upper Arm $L_1$)** | $250\text{ mm}$ | Shoulder joint to elbow joint |
| **Link 2 (Forearm $L_2$)** | $230\text{ mm}$ | Elbow joint to wrist/tool center point |
| **Total Reach Radius** | $\approx 480\text{ mm}$ | Horizontal workspace coverage |
| **Payload Capacity** | $0.25 - 0.50\text{ kg}$ | Suitable for cups, fruits, small parcels |
| **Elbow Torque ($\tau_3$)** | $\ge 3.0\text{ N}\cdot\text{m}$ | Planetary NEMA 17 or RobStride 01 actuator |
| **Shoulder Torque ($\tau_2$)**| $\ge 9.0\text{ N}\cdot\text{m}$ | Planetary NEMA 23 (1:10) or CyberGear BLDC |
| **Base Yaw Torque ($\tau_1$)**| $\ge 3.0\text{ N}\cdot\text{m}$ | Direct / 1:5 reduction NEMA 17/23 |

---

## 🚀 Getting Started

### 1. Prerequisites
- **Ubuntu 24.04 LTS** (or 22.04 LTS)
- **ROS 2 Jazzy Jalisco** (Desktop Install)
- **Gazebo Harmonic (Gz Sim)** with `ros_gz` bridges:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-joint-trajectory-controller
  ```

### 2. AnyGrasp Environment Setup (Optional for AI Grasping)
AnyGrasp requires PyTorch with CUDA and MinkowskiEngine (tested on Python 3.10):
```bash
# 1. Create Conda environment
conda create -n robot_env python=3.10 -y
conda activate robot_env

# 2. Install PyTorch matching your CUDA version (e.g. CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install MinkowskiEngine & AnyGrasp dependencies
pip install open3d scipy opencv-python "numpy<2.0"
# Follow AnyGrasp SDK instructions to build MinkowskiEngine and register license
```

> **Note on Model Checkpoint & License:**
> Download `checkpoint_detection.tar` from the official AnyGrasp repository and place it in your AnyGrasp SDK directory. Ensure your valid license file (`*.lic`) is placed in the designated folder as instructed by AnyGrasp.

---

## 🛠️ Build & Installation

Clone this repository and compile with colcon:
```bash
# Navigate to the workspace
cd ~/robot_3dof_ws

# Build package
colcon build --symlink-install

# Source the ROS 2 workspace
source install/setup.bash
```

---

## 🎮 Running the Simulation

### Option A: Standard Autonomous Demo (Heuristic Mode)
Runs Gazebo Harmonic simulation, spawns the 3-DoF arm and objects, and executes pick-and-place with built-in point cloud cluster detection:
```bash
ros2 launch robot_3dof grasp_demo.launch.py
```

### Option B: AI-Powered AnyGrasp Mode
In terminal 1, start the AnyGrasp IPC Bridge service in your conda environment:
```bash
conda activate robot_env
python3 src/robot_3dof/robot_3dof/anygrasp_service.py
```

In terminal 2, launch the ROS 2 simulation with AnyGrasp enabled:
```bash
source install/setup.bash
ros2 launch robot_3dof grasp_demo.launch.py use_anygrasp:=true
```

---

## 📊 Visualizations

- **RViz2**: Displays camera RGB-D stream, target workspace bounding boxes, 3D candidate gripper wireframes, and confidence labels.
- **Terminal Dashboard**: Outputs real-time ASCII table with grasp candidate pose $(x, y, z, \text{yaw}, \text{pitch})$, width, and score ranking.

---

## 📜 License & Acknowledgments

- **Project Core**: Released under the MIT License.
- **AnyGrasp**: AnyGrasp library and model weights are subject to the original authors' academic/commercial licensing terms.
- Special thanks to the ROS 2 and Gazebo communities.
