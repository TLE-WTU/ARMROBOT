# 🤖 5-DoF Robotic Arm with AI Grasp Synthesis & Perception Pipeline

An autonomous Pick-and-Place robotics framework featuring a **5-DoF Articulated Robotic Arm** with a parallel-jaw gripper, simulated in **ROS 2 Jazzy** and **Gazebo Harmonic**, integrated with **AnyGrasp** (Deep Learning 6-DoF grasp synthesis with MinkowskiEngine & PyTorch) and an analytical **RANSAC Tabletop Segmentation & Adaptive Geometric Reduction** perception pipeline.

---

## 🌟 Key Highlights

- **5-DoF Kinematic Structure**:
  - **Joint 1 (Base Turret)**: Continuous/Revolute yaw rotation around $Z$.
  - **Joint 2 (Shoulder)**: Pitch rotation around $Y$.
  - **Joint 3 (Elbow)**: Pitch rotation around $Y$.
  - **Joint 4 (Wrist Pitch)**: Pitch rotation around $Y$ (enforces true top-down $180^\circ$ approach with anti-collision folding protection).
  - **Joint 5 (Wrist Roll)**: Axial roll rotation around $Z$ (aligns gripper jaws precisely with object principal axes).
  - **Gripper**: Prismatic parallel-jaw mechanism with high-friction silicone pads.
- **Analytical Inverse Kinematics (IK) & S-Curve Trajectories**:
  - Analytical closed-form IK solver supporting 5-DoF top-down grasp synthesis with exact yaw alignment.
  - Multi-point cosine S-curve trajectory interpolation eliminating acceleration spikes and joint jerk.
  - Pre-flight kinematic validation checking all trajectory waypoints (pre-grasp, grasp, lift, pre-place, place) before initiating movement.
- **Perception Pipeline**:
  - **RANSAC Tabletop Plane Segmentation**: Mathematically separates the support table plane from objects.
  - **Adaptive Geometric Reduction**: Voxel downsampling + uniform strided reduction down to target point counts for real-time edge processing.
  - **PCA-Based Grasp Yaw Extraction**: Computes object minor principal axis for optimal antipodal grasp alignment.
  - **Analytical Virtual Safety Floor**: Enforces minimum finger tip clearance above the table surface ($Z=0.225\,\text{m}$ in base frame).
- **AnyGrasp Deep Learning AI Integration**:
  - Optional zero-shot 6-DoF grasp detection via high-speed UNIX domain socket IPC bridge.
  - Transparent inter-environment communication between ROS 2 Jazzy (Python 3.12) and PyTorch/MinkowskiEngine (Python 3.10).
- **Benchmark & Edge Stress Scenarios**:
  - Evaluates standard 3D meshes (Coffee Mug, Rubber Duck, Torus Toy).
  - Benchmark scenarios for transparent objects (optical refraction/ghost grasps), ultra-thin flat objects (low affordance), and dense clutter.
- **Rich 3D RViz2 & Terminal HUD**:
  - 3D gripper wireframe visualization showing jaw width, opening, and orientation.
  - Separated point cloud topics for table and object points.
  - Live ASCII telemetry dashboard printing throughput (FPS), point cloud reduction percentage, and grasp metrics.

## 🎥 Demo Video

> **ARM_ROBOT 5-DoF (RANSAC + PCA + AnyGrasp):**
> 
> File video in repository: [`media/demo_5dof_grasp.webm`](media/demo_5dof_grasp.webm)

---

## 🏗️ System Architecture

```
                                  +---------------------------+
                                  |  Gazebo Harmonic (Sim)    |
                                  |  - Overhead RGB-D Camera  |
                                  |  - 5-DoF Arm + Gripper    |
                                  |  - 3D Benchmark Objects   |
                                  +-------------+-------------+
                                                |
                     Camera Depth/RGB & Joint State Topics (/clock synced)
                                                v
+------------------------+        +---------------------------+
| AnyGrasp Server        |        | ROS 2 Jazzy Core Nodes    |
| (Conda: Python 3.10)   |<------>| (Python 3.12)             |
| - MinkowskiEngine      |  IPC   | - grasp_detection_node    |
| - 6-DoF Grasp Network  | Socket |   • RANSAC Plane Seg      |
| - Score/Width/Depth    |        |   • Adaptive Reduction    |
+------------------------+        |   • PCA Yaw Orientation   |
                                  | - pick_place_node (FSM)   |
                                  |   • Analytical 5-DoF IK   |
                                  |   • S-Curve Interpolation |
                                  |   • Pre-Trajectory Check  |
                                  +-------------+-------------+
                                                |
                                        RViz2 Markers &
                                  ros2_control Joint Commands
```

---

## 📁 Repository Structure

```
robot_3dof_ws/
├── README.md                           # Documentation and startup guide
├── RUN_GUIDE.md                        # Quick command reference
├── benchmark_results.csv               # Automated perception & grasp benchmark logs
└── src/
    ├── robot_5dof/                     # Primary 5-DoF robot package
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── config/
    │   │   ├── anygrasp_params.yaml    # Grasp thresholds, camera parameters, workspace bounds
    │   │   └── controllers.yaml        # ros2_control configuration (JTC + Gripper Action)
    │   ├── launch/
    │   │   ├── gazebo.launch.py        # World, robot spawner, controllers, bridges
    │   │   └── grasp_demo.launch.py    # Main launch entrypoint (Gazebo + Nodes + RViz)
    │   ├── robot_5dof/
    │   │   ├── __init__.py
    │   │   ├── anygrasp_service.py     # IPC Bridge Service client & server
    │   │   ├── grasp_detection_node.py # Point cloud processor, RANSAC, PCA, diagnostics
    │   │   ├── ik_solver.py            # Analytical 5-DoF IK & S-curve trajectory generator
    │   │   └── pick_place_node.py      # Autonomous Pick-and-Place State Machine
    │   ├── rviz/
    │   │   └── config.rviz             # RViz2 display layout
    │   ├── urdf/
    │   │   ├── robot_5dof.urdf.xacro   # 5-DoF arm & gripper kinematic chain
    │   │   ├── robot_5dof_gazebo.xacro # Gazebo Sim plugins, sensors, materials
    │   │   └── robot_5dof_ros2_control.xacro # Hardware interface for ros2_control
    │   └── worlds/
    │       └── pick_and_place.sdf      # Gazebo world with table, lighting & 3D objects
    ├── robot_4dof/                     # 4-DoF reference package
    └── robot_3dof/                     # 3-DoF legacy package
```

---

## ⚙️ Kinematic & Mechanical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Height ($h_0$)** | $50\text{ mm}$ | Ground mount cylinder ($Z_{\text{base}} = 25\text{ mm}$) |
| **Turret Link 1 ($h_1$)** | $150\text{ mm}$ | Base to shoulder pitch axis ($Z_{\text{shoulder}} = 175\text{ mm}$) |
| **Upper Arm Link 2 ($L_1$)** | $250\text{ mm}$ | Shoulder joint to elbow joint |
| **Forearm Link 3 ($L_2$)** | $200\text{ mm}$ | Elbow joint to wrist pitch joint |
| **Wrist Pitch Link 4 ($h_4$)** | $30\text{ mm}$ | Wrist pitch joint to wrist roll joint |
| **Wrist Roll Link 5 ($h_5$)** | $20\text{ mm}$ | Wrist roll joint to gripper base |
| **Gripper + Finger Pad ($L_{\text{hand}}$)** | $30\text{ mm}$ | Gripper base to grasp center ($L_{\text{total\_wrist}} = 80\text{ mm}$) |
| **Total Max Reach** | $\approx 450\text{ mm}$ | Planar reach $L_1 + L_2$ from shoulder |
| **Table Top ($Z_{\text{base}}$)** | $225\text{ mm}$ | Table surface in robot base frame ($250\text{ mm}$ in world) |
| **Nominal Grasp Height ($Z_{\text{base}}$)** | $248 - 253\text{ mm}$ | Object center on tabletop |
| **Joint Limits** | | |
| • `joint1` (Base Yaw) | $[-180^\circ, +180^\circ]$ | Horizontal turret rotation |
| • `joint2` (Shoulder Pitch) | $[-143^\circ, +143^\circ]$ | Main arm lift |
| • `joint3` (Elbow Pitch) | $[-143^\circ, +143^\circ]$ | Forearm reach |
| • `joint4` (Wrist Pitch) | $[-90^\circ, +90^\circ]$ | Top-down orientation with anti-folding clamp |
| • `joint5` (Wrist Roll) | $[-90^\circ, +90^\circ]$ | Axial rotation for jaw alignment |
| • `gripper_left_joint` | $[0.0, 30.0\text{ mm}]$ | Parallel jaw opening stroke ($0 - 60\text{ mm}$ opening) |

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

---

## 🛠️ Build & Installation

Always build inside the workspace directory (`robot_3dof_ws`):

```bash
# 1. Navigate to the workspace
cd ~/ARMROBOT/robot_3dof_ws

# 2. Build the robot_5dof package
colcon build --packages-select robot_5dof --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

# 3. Source the environment
source install/setup.bash
```

---

## 🎮 Running the Simulation

### Option A: Autonomous Pick-and-Place (Heuristic Mode - Default)
Launches Gazebo Harmonic, spawns the 5-DoF robot, loads the table with 3D benchmark objects, runs the RANSAC perception node, and executes autonomous pick-and-place with top-down antipodal grasping:

```bash
source install/setup.bash
ros2 launch robot_5dof grasp_demo.launch.py
```

### Option B: Deep Learning AI AnyGrasp Mode
If you have an active AnyGrasp environment:

**Terminal 1 — Start AnyGrasp IPC Bridge:**
```bash
conda activate robot_env
python3 src/robot_5dof/robot_5dof/anygrasp_service.py
```

**Terminal 2 — Launch ROS 2 Framework:**
```bash
source install/setup.bash
ros2 launch robot_5dof grasp_demo.launch.py use_anygrasp:=true
```

### Option C: Benchmark & Ablation Scenarios
Test the perception and grasping system against specific edge scenarios:

```bash
# Transparent bottle (evaluates optical refraction & ghost grasps)
ros2 launch robot_5dof grasp_demo.launch.py scenario:=transparent_bottle

# Ultra-thin flat object (evaluates table clearance affordance)
ros2 launch robot_5dof grasp_demo.launch.py scenario:=flat_object

# Dense clutter (evaluates semantic blindness & adjacent collision)
ros2 launch robot_5dof grasp_demo.launch.py scenario:=dense_clutter
```

---

## 📊 Perception & Motion Telemetry

During execution, the terminal displays real-time telemetry:

```
════════════════════════════════════════════════════════════════════════════════════════════════
🔬 [EDGE GRASP BENCHMARK] Pipeline: RANSAC+Reduction (Proposed) │ Engine: Heuristic Fallback │ DOF: 5
────────────────────────────────────────────────────────────────────────────────────────────────
 📊 Perception Telemetry:
    • Points: Raw=2540 → Processed=642 (74.7% reduced)
    • Latency: RANSAC=3.2ms │ Reduction=1.8ms │ Inference=2.1ms
    • Performance: Total=7.1ms │ Throughput=140.8 FPS │ Table Collision: ✅ ZERO
────────────────────────────────────────────────────────────────────────────────────────────────
 🤖 Detected 3 Grasps on 3D Objects (5 DoF with Wrist Pitch & Roll):
 Rank  │ Score   │ Position (X, Y, Z)       │ Yaw°    │ Width   │ Diagnostic / Analysis
────────────────────────────────────────────────────────────────────────────────────────────────
 ★ #1  │ 0.9500  │ [ 0.30,  0.08,  0.25]    │  23.5°  │  4.0cm  │ ✅ [OPTIMAL] Điểm gắp an toàn hợp lệ
   #2  │ 0.9200  │ [ 0.25,  0.03,  0.25]    │ -12.1°  │  4.0cm  │ ✅ [OPTIMAL] Điểm gắp an toàn hợp lệ
   #3  │ 0.8800  │ [ 0.34, -0.06,  0.25]    │  54.2°  │  4.0cm  │ ✅ [OPTIMAL] Điểm gắp an toàn hợp lệ
════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 📜 License & Acknowledgments

- **Core Framework**: Released under the MIT License.
- **AnyGrasp**: Subject to the original authors' licensing terms.
- Built with **ROS 2 Jazzy**, **Gazebo Harmonic**, and **ros2_control**.
