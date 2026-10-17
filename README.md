# ARMROBOT: Kinematically Constrained 4-DOF Tabletop Grasp Synthesis

[![CI](https://github.com/TLE-WTU/ARMROBOT/actions/workflows/ci.yaml/badge.svg)](https://github.com/TLE-WTU/ARMROBOT/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ROS 2: Jazzy](https://img.shields.io/badge/ROS%202-Jazzy%20Jalisco-orange.svg)](https://docs.ros.org/en/jazzy/)
[![Gazebo: Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-lightblue.svg)](https://gazebosim.org/)

ROS 2 package for a 4-DOF articulated robotic manipulator featuring closed-form dual-branch inverse kinematics, Franka Hand parallel-jaw gripper integration, and an analytical geometric grasp synthesis pipeline designed for kinematically constrained planar-vertical architectures.

**Author:** Lê Mạnh Tiến

---

## 1. Abstract & Scientific Problem Formulation

Deep learning grasp synthesis models (e.g., AnyGrasp, Contact-GraspNet, GraspNet-1Billion) formulate grasp proposal generation in unconstrained $SE(3)$, assuming manipulators with $\ge 6$ degrees of freedom capable of executing arbitrary position and 3D orientation $(x, y, z, \phi, \theta, \psi)$. 

However, embedded and cost-sensitive robotic platforms frequently deploy **4-DOF serial manipulators** (azimuth $\theta_1$, shoulder pitch $\theta_2$, elbow pitch $\theta_3$, wrist pitch $\theta_4$) with top-down parallel-jaw grippers. Such platforms face two fundamental challenges:

1. **The 4-DOF Kinematic Yaw Lock:** Under the vertical approach condition ($\sum_{i=2}^4 \theta_i = \pi$), the gripper tool axis is fixed downwards along $-Z$. Because only the base joint $\theta_1$ rotates about the $Z$-axis, the end-effector closing direction in the world frame is **tangentially locked** to the base azimuth:
   $$\psi_{\text{closing}} \equiv \theta_1 + \frac{\pi}{2} \pmod{\pi}$$
   Without a fifth joint (wrist roll), the robot cannot rotate the gripper independently of its position $(x, y)$. Unconstrained 6-DOF models predict grasp angles that are physically infeasible, resulting in severe orientation mismatch ($\Delta\psi \sim 60^\circ - 90^\circ$) and grasp failure.

2. **Computational Overhead:** Deep neural models require dedicated GPU hardware, introducing $70 - 500\ \text{ms}$ inference latencies and substantial power consumption.

### Core Scientific Thesis
*On kinematically constrained 4-DOF manipulators, closed-form geometric grasp synthesis (exploiting cross-sectional slicing, covariance decomposition, and primitive shape fitting) matches or exceeds deep neural affordance models in dynamic pick-and-place success rate, while reducing end-to-end compute latency by over $100\times$ on standard CPU hardware and eliminating GPU dependencies.*

---

## 2. Manipulator Kinematics & Workspace Specification

### Manipulator Joint Parameters

| Joint | Type | Axis | Description | Range |
| :--- | :--- | :---: | :--- | :---: |
| `joint1` | Revolute | $Z$ | Base rotation (Azimuth / Yaw) | $[-\pi, +\pi]$ rad |
| `joint2` | Revolute | $Y$ | Shoulder pitch | $[-2.5, +2.5]$ rad |
| `joint3` | Revolute | $Y$ | Elbow pitch | $[-2.5, +2.5]$ rad |
| `joint4` | Revolute | $Y$ | Wrist pitch (Top-down compensation) | $[-\pi, +\pi]$ rad |
| `gripper_left_joint` | Prismatic | $Y$ | Franka left finger | $[0.0, 0.035]$ m |
| `gripper_right_joint` | Prismatic | $Y$ | Franka right finger (mimic) | $[0.0, 0.035]$ m |

### Link Dimensions
- Base height ($h_{\text{base}}$): $0.175\ \text{m}$ (from `base_link` origin to `joint2` shoulder axis)
- Upper arm length ($L_1$): $0.250\ \text{m}$ (`joint2` to `joint3`)
- Forearm length ($L_2$): $0.200\ \text{m}$ (`joint3` to `joint4`)
- Tool center point offset ($L_{\text{hand}}$): $0.1334\ \text{m}$ (`joint4` to grasp contact center)

### Curvilinear Reachable Workspace Envelope
In earlier configurations, the workspace was approximated as a rectangular cuboid $X \in [0.10, 0.50]\ \text{m}, Z \in [0.22, 0.45]\ \text{m}$. However, under the top-down vertical grasp condition, the wrist center must be positioned at $z_{\text{wrist}} = z + L_{\text{hand}}$. 

The vertical offset between the shoulder axis and wrist center is:
$$\Delta z(z) = z + L_{\text{hand}} - h_{\text{base}} = z + 0.1334 - 0.175 = z - 0.0416\ \text{m}$$

The maximum reach in the horizontal ground plane is bounded by the triangle inequality of links $L_1$ and $L_2$:
$$r_{\max}(z) = \sqrt{(L_1 + L_2)^2 - \Delta z(z)^2} = \sqrt{0.45^2 - (z - 0.0416)^2}$$

The minimum reach is non-zero when $|\Delta z| \le |L_1 - L_2| = 0.05\ \text{m}$:
$$r_{\min}(z) = \begin{cases} \sqrt{(L_1 - L_2)^2 - \Delta z(z)^2} & \text{if } |\Delta z| \le 0.05\ \text{m} \\ 0 & \text{otherwise} \end{cases}$$

#### True Workspace Reach vs. Target Height

| Target Height $z$ | Vertical Offset $\Delta z$ | True Maximum Radial Reach $r_{\max}(z)$ | Kinematic Status |
| :---: | :---: | :---: | :--- |
| $0.225\ \text{m}$ (Table surface) | $0.1834\ \text{m}$ | **$0.411\ \text{m}$** (NOT $0.50\ \text{m}$) | Bounded by link hypotenuse |
| $0.300\ \text{m}$ | $0.2584\ \text{m}$ | **$0.368\ \text{m}$** | Intermediate reach |
| $0.400\ \text{m}$ | $0.3584\ \text{m}$ | **$0.272\ \text{m}$** | Significantly reduced |
| $0.450\ \text{m}$ (Ceiling) | $0.4084\ \text{m}$ | **$0.189\ \text{m}$** (NOT $0.50\ \text{m}$) | Bounded |
| $0.4916\ \text{m}$ | $0.4500\ \text{m}$ | **$0.000\ \text{m}$** | Absolute kinematic ceiling ($r=0$) |

Target coordinates outside this curvilinear volume cannot be reached by any joint configuration. ARMROBOT filters all point clouds and grasp candidates through this exact algebraic boundary prior to evaluation.

---

## 3. Mathematical Proof: The 4-DOF Yaw Lock

Consider a 4-DOF manipulator where:
- $\theta_1$ rotates the entire superstructure about the world $Z$-axis.
- $\theta_2, \theta_3, \theta_4$ rotate about parallel axes perpendicular to the plane formed by the arm links ($Y_{\text{local}}$).

To grasp tabletop objects from above, the approach vector must be vertical:
$$\hat{\mathbf{z}}_{\text{tcp}} = \begin{bmatrix} 0 \\ 0 \\ -1 \end{bmatrix}$$
This enforces the kinematic constraint:
$$\theta_2 + \theta_3 + \theta_4 = \pi$$

The Franka Hand fingers actuate along the end-effector local $Y$-axis. In the base reference frame, the orientation of the end-effector is obtained by rotating around $Z$ by $\theta_1$ and around $Y$ by $\theta_2 + \theta_3 + \theta_4 = \pi$:
$$\mathbf{R}_{\text{ee}} = \mathbf{R}_z(\theta_1) \mathbf{R}_y(\pi) = \begin{bmatrix} \cos\theta_1 & -\sin\theta_1 & 0 \\ \sin\theta_1 & \cos\theta_1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{bmatrix} = \begin{bmatrix} -\cos\theta_1 & -\sin\theta_1 & 0 \\ -\sin\theta_1 & \cos\theta_1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$

The closing vector of the gripper fingers corresponds to the second column of $\mathbf{R}_{\text{ee}}$:
$$\hat{\mathbf{y}}_{\text{ee}} = \begin{bmatrix} -\sin\theta_1 \\ \cos\theta_1 \\ 0 \end{bmatrix} = \begin{bmatrix} -y / \sqrt{x^2 + y^2} \\ x / \sqrt{x^2 + y^2} \\ 0 \end{bmatrix}$$

The angle of this closing vector in the horizontal plane is:
$$\psi_{\text{closing}} = \text{atan2}(\cos\theta_1, -\sin\theta_1) = \theta_1 + \frac{\pi}{2} \pmod{\pi}$$

### Analytical Consequence
The finger closing axis is **strictly perpendicular to the radial vector** $\mathbf{r} = (x, y)^T$. The robot has **zero freedom** to adjust its grasp yaw independently of target position $(x, y)$. 

When unconstrained 6-DOF models propose a grasp with desired orientation $\psi^*$, the physical discrepancy is:
$$\Delta\psi = \min\left(|\psi^* - \psi_{\text{closing}}|, \pi - |\psi^* - \psi_{\text{closing}}|\right)$$
On asymmetric objects, if $\Delta\psi > 25^\circ$, the fingers collide with the object boundary during closing, preventing force-closure. 

ARMROBOT resolves this by:
1. Automatically projecting grasp candidate orientation to $\psi_{\text{closing}}(x, y)$.
2. Scoring candidates by an orientability penalty $P_{\text{orient}} = \exp\left(-\frac{\Delta\psi^2}{2\sigma_\psi^2}\right)$, rejecting candidates that cannot be gripped along the available closing direction.

---

## 4. Complete Analytical Dual-Branch Inverse Kinematics

Given target position $\mathbf{p}^* = (x^*, y^*, z^*)^T$:

1. **Base Azimuth ($\theta_1$):**
   $$\theta_1 = \text{atan2}(y^*, x^*)$$

2. **Wrist Center Projection:**
   $$r_{\text{wrist}} = \sqrt{(x^*)^2 + (y^*)^2}, \quad z_{\text{wrist}} = z^* + L_{\text{hand}}$$
   $$\Delta z = z_{\text{wrist}} - h_{\text{base}}, \quad D = \sqrt{r_{\text{wrist}}^2 + \Delta z^2}$$

3. **Dual Elbow Branches ($s \in \{+1, -1\}$):**
   $$\cos\theta_3 = \frac{D^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$
   If $|L_1 - L_2| \le D \le L_1 + L_2$:
   $$\theta_3 = s \cdot \arccos\left(\text{clip}(\cos\theta_3, -1, 1)\right)$$
   $$\alpha = \text{atan2}(r_{\text{wrist}}, \Delta z), \quad \beta = \text{atan2}(L_2 \sin\theta_3, L_1 + L_2 \cos\theta_3)$$
   $$\theta_2 = \alpha - \beta$$
   $$\theta_4 = \left( (\pi - (\theta_2 + \theta_3) + \pi) \pmod{2\pi} \right) - \pi$$

4. **Joint Limits & Clearance Verification:**
   - Limits: $|\theta_1| \le \pi$, $|\theta_2| \le 2.5\ \text{rad}$, $|\theta_3| \le 2.5\ \text{rad}$, $|\theta_4| \le \pi$.
   - Elbow table clearance: $z_{\text{elbow}} = h_{\text{base}} + L_1 \cos\theta_2 \ge z_{\text{table}} + 0.02\ \text{m}$.
   - Branch selection: If both $s = +1$ (elbow-up) and $s = -1$ (elbow-down) are kinematically feasible and collision-free, the solver selects the branch minimizing joint displacement $\|\mathbf{q} - \mathbf{q}_{\text{current}}\|_2$, with default preference for the elbow-up configuration.

### Latency Measurement
Benchmarked over $100,000$ consecutive evaluations on an Intel Core i7 x86_64 CPU:
- **Mean Latency:** $1.29\ \mu\text{s}$ per query (standard deviation: $0.18\ \mu\text{s}$).
- Sub-millimeter spatial error ($< 10^{-6}\ \text{m}$) with zero numerical optimization failures.

---

## 5. Perception & Grasp Synthesis Pipeline

```
                    +------------------------------------+
                    |        RGB-D Point Cloud           |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   RANSAC Table Plane Segmentation  |
                    |   & Curvilinear Workspace Filter   |
                    +-----------------+------------------+
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
        +-----------------------+            +-----------------------+
        |   AnyGrasp Model      |            |  Geometric Refiners   |
        |  (PyTorch / IPC)      |            |  (Slice, RANSAC, PCA) |
        +-----------+-----------+            +-----------+-----------+
                    |                                    |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   4-DOF Yaw Lock Projection        |
                    |   & Virtual Safety Floor (z_table) |
                    +-----------------+------------------+
                                      | Valid Pose (x, y, z, psi_closing)
                                      v
                    +------------------------------------+
                    |  Dual-Branch Analytical IK Solver  |
                    +-----------------+------------------+
                                      | Joint Trajectory
                                      v
                    +------------------------------------+
                    |    ros2_control Joint Controller   |
                    +------------------------------------+
```

### Geometric Refiner Modules
1. **Cross-Section Slicing:** Evaluates horizontal point cloud slices at varying $z$ to locate narrow waistlines (e.g., bottle necks, handles).
2. **Primitive RANSAC:** Fits cylinders, spheres, and rectangular bounding boxes to segment regular geometries.
3. **PCA (Principal Component Analysis):** Computes spatial covariance eigenvectors to determine principal inertia axes.
4. **Surface Normals:** Estimates antipodal contact point pairs using $k$-NN surface normal estimation.
5. **OBB (Oriented Bounding Box):** Computes minimum volume enclosing boxes for parallel-jaw stroke alignment.

---

## 6. Empirical Benchmark Results

Evaluated across $N = 40$ simulated rollouts per method ($N_{\text{total}} = 480$ trials) in PyBullet with dynamic contact physics, simulated sensor degradation (Gaussian depth noise $\sigma = 2\ \text{mm}$, 10% point cloud dropout), and diverse object geometries (`duck`, `lego`, `block`, `dense_clutter`).

### Overall Performance Leaderboard ($N = 40$)

All success rates report **Wilson 95% Confidence Intervals**:

| Rank | Method | Architecture | Success Rate (95% Wilson CI) | IK Feasibility | Table Collision | Mean Latency (ms) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **Pure Geometric (CrossSectionSlice)** | CPU Analytical | **45.0%** [30.7%, 60.2%] | 97.5% | 0.0% | **0.43 ± 0.07** |
| 2 | **Pure Geometric (PrimitiveRANSAC)** | CPU Analytical | **37.5%** [24.2%, 53.0%] | 100.0% | 0.0% | **1.66 ± 0.27** |
| 3 | **Pure Geometric (PCA)** | CPU Analytical | **30.0%** [18.1%, 45.4%] | 100.0% | 0.0% | **0.31 ± 0.03** |
| 4 | **Pure Geometric (SurfaceNormals)** | CPU Analytical | **30.0%** [18.1%, 45.4%] | 100.0% | 0.0% | **7.91 ± 4.25** |
| 5 | **Hybrid (AnyGrasp + CrossSectionSlice)** | GPU AI + CPU Geometry | **17.5%** [8.7%, 31.9%] | 100.0% | 0.0% | **297.48 ± 199.12** |
| 6 | **Pure Geometric (OBB)** | CPU Analytical | **15.0%** [7.1%, 29.1%] | 100.0% | 0.0% | **0.29 ± 0.02** |
| 7 | **Hybrid (AnyGrasp + PrimitiveRANSAC)** | GPU AI + CPU Geometry | **12.5%** [5.5%, 26.1%] | 100.0% | 0.0% | **1438.84 ± 1159.93** |
| 8 | **Hybrid (AnyGrasp + PCA)** | GPU AI + CPU Geometry | **10.0%** [4.0%, 23.1%] | 100.0% | 0.0% | **108.64 ± 55.44** |
| 9 | **Hybrid (AnyGrasp + SurfaceNormals)** | GPU AI + CPU Geometry | **10.0%** [4.0%, 23.1%] | 100.0% | 0.0% | **193.66 ± 112.24** |
| 10 | **Hybrid (AnyGrasp + OBB)** | GPU AI + CPU Geometry | **7.5%** [2.6%, 19.9%] | 100.0% | 0.0% | **196.33 ± 125.42** |
| 11 | **AnyGrasp (4-DOF Adapted Baseline)** | Deep Learning Baseline | **0.0%** [0.0%, 8.8%] | 75.0% | 0.0% | **77.16 ± 5.92** |
| 12 | **AnyGrasp (Raw 6-DOF Baseline)** | Deep Learning Baseline | **0.0%** [0.0%, 8.8%] | 57.5% | 70.0% | **91.22 ± 51.73** |

*For complete per-object breakdowns and raw rollout CSV logs, see [ACADEMIC_BENCHMARK_REPORT.md](benchmark/ACADEMIC_BENCHMARK_REPORT.md) and [academic_benchmark_trial_log.csv](benchmark/academic_benchmark_trial_log.csv).*

### Candidate Filtration Funnel
To eliminate survivorship bias, candidate counts are audited across each filtering stage:

| Method | Generated Candidates | Rejected (Workspace) | Rejected (Table) | Rejected (4-DOF Yaw) | Final Executed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (CrossSectionSlice)** | 40 | 1 | 0 | 0 | 39 |
| **Pure Geometric (PrimitiveRANSAC)** | 40 | 0 | 0 | 0 | 40 |
| **Pure Geometric (PCA)** | 40 | 0 | 0 | 0 | 40 |
| **AnyGrasp (Raw 6-DOF Baseline)** | 1,620 | 17 | 28 | 33 | 31 |
| **AnyGrasp (4-DOF Adapted Baseline)** | 1,498 | 10 | 0 | 10 | 30 |

### Statistical Hypothesis Testing (McNemar's Paired Test)
Tested on matched random seeds ($N = 40$) with Edwards' continuity correction:
- **Pure PCA vs. AnyGrasp (4-DOF Adapted Baseline):** Discordant pairs $b/c = 12/0$, $\chi^2 = 10.083$, $p = 0.0015$ ($p < 0.05$). **Statistically significant superiority of geometric synthesis.**
- **Pure CrossSectionSlice vs. AnyGrasp (4-DOF Adapted):** Discordant pairs $b/c = 18/0$, $\chi^2 = 16.056$, $p < 0.0001$. **Statistically significant.**
- **Pure PCA vs. Pure OBB:** Discordant pairs $b/c = 10/4$, $\chi^2 = 1.786$, $p = 0.1814$ (no statistically significant difference).

---

## 7. Installation & Setup

### Prerequisites
- Ubuntu 24.04 LTS (or Ubuntu 22.04 LTS)
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic
- Python 3.10+

### Option A: Standalone CPU Setup (Recommended)
No GPU, CUDA, or AnyGrasp license is required. Runs all geometric refiners, IK solvers, and PyBullet benchmarks out of the box:

```bash
# 1. Install ROS 2 Jazzy dependencies
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

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build ROS 2 workspace
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select robot_arm --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

### Option B: Docker Container
```bash
# Build the container
docker build -t armrobot:latest .

# Run benchmark inside container
docker run --rm armrobot:latest pytest -v tests/
```

### Option C: Optional AnyGrasp Deep Affordance Setup
AnyGrasp requires an academic license file and registered network interface:
```bash
conda activate robot_env
export PYTHONPATH="${PYTHONPATH}:./:src/robot_arm:/path/to/anygrasp_sdk/grasp_detection"
python3 benchmark/run_academic_benchmark.py --checkpoint /path/to/checkpoint_detection.tar --trials 10
```

---

## 8. Usage Guide

### 1. Launch Gazebo Harmonic Simulation
Starts the simulated 4-DOF manipulator with Franka Hand and `ros2_control`:
```bash
source install/setup.bash
ros2 launch robot_arm gazebo.launch.py
```

### 2. Run Autonomous Grasp Detection & Manipulation
```bash
# Analytical Geometric Mode (Zero-GPU, fast CPU execution)
ros2 launch robot_arm grasp_demo.launch.py use_anygrasp:=false grasp_mode:=heuristic

# Hybrid Mode (Deep Neural Proposals + Geometric Refinement)
ros2 launch robot_arm grasp_demo.launch.py use_anygrasp:=true grasp_mode:=hybrid
```

### 3. Run Academic Benchmark Suite
```bash
# Run simulated PyBullet rollouts with depth noise and statistical reporting
python3 benchmark/run_academic_benchmark.py --trials 10

# Run automated unit test suite (79 tests)
PYTHONPATH=./:src/robot_arm pytest -v tests/
```

---

## 9. Repository Structure

```
ARMROBOT/
├── .github/workflows/ci.yaml       # Continuous integration pipeline
├── Dockerfile                      # Containerized reproduction environment
├── LICENSE                         # MIT License
├── README.md                       # Academic documentation & kinematics proofs
├── requirements.txt                # Pinned Python package dependencies
├── benchmark/                      # Evaluation suite and statistical analysis
│   ├── ACADEMIC_BENCHMARK_REPORT.md # Comprehensive experimental report
│   ├── academic_benchmark_results.csv # Aggregated statistical metrics
│   ├── academic_benchmark_trial_log.csv # Raw per-trial rollout traces
│   ├── dataset_loader.py           # Real sensor RGB-D point cloud loader
│   ├── eval_ap.py                  # Average Precision (AP_0.4, AP_0.8) metrics
│   ├── evaluator.py                # Grasp candidate evaluation harness
│   ├── metrics.py                  # Wilson CIs, McNemar test, Force-Closure
│   ├── physics_environment.py      # PyBullet simulation with depth sensor noise
│   └── run_academic_benchmark.py   # Automated 480-rollout benchmark runner
├── src/robot_arm/                  # ROS 2 package
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/                     # Controller and hardware configurations
│   │   ├── controllers_4dof.yaml
│   │   └── robot_params.yaml
│   ├── launch/                     # ROS 2 launch definitions
│   │   ├── gazebo.launch.py
│   │   └── grasp_demo.launch.py
│   ├── meshes/                     # Arm links and Franka Hand meshes
│   ├── robot_arm/                  # Modular Python implementations
│   │   ├── anygrasp_service.py     # IPC bridge for AnyGrasp model
│   │   ├── benchmark_dataset.py    # Canonical benchmark object generator
│   │   ├── geometric_refinement.py # 5 Geometric refiners with 4-DOF alignment
│   │   ├── grasp_detection_node.py # Perception node with RANSAC table fitting
│   │   ├── ik_solver.py            # Dual-branch analytical 4-DOF IK solver
│   │   └── pick_place_node.py      # State machine for trajectory execution
│   ├── urdf/                       # 4-DOF Xacro and ros2_control descriptions
│   └── worlds/                     # Tabletop Gazebo simulation worlds
└── tests/                          # Automated unit and integration test suite
    ├── test_benchmark_suite.py
    ├── test_eval_ap.py
    ├── test_geometric_refinement.py
    ├── test_grasp_detection.py
    └── test_ik_solver.py
```

---

## 10. References

1. **Ferrari, C., & Canny, J.** (1992). *Planning optimal grasps*. IEEE International Conference on Robotics and Automation (ICRA), pp. 2290–2295.
2. **Nguyen, V.-D.** (1988). *Constructing force-closure grasps*. The International Journal of Robotics Research, 7(3), 3–16.
3. **Fang, H.-S., Wang, C., Gou, M., & Lu, C.** (2020). *GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11444–11453.
4. **Wilson, E. B.** (1927). *Probable inference, the law of population, and the nature of a binomial series*. Journal of the American Statistical Association, 22(158), pp. 209–212.
5. **McNemar, Q.** (1947). *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika, 12(2), pp. 153–157.

---

## 11. Citation & License

```bibtex
@misc{tien2026armrobot,
  author = {Lê Mạnh Tiến},
  title = {ARMROBOT: Kinematically Constrained 4-DOF Tabletop Grasp Synthesis},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/TLE-WTU/ARMROBOT}}
}
```

This project is licensed under the [MIT License](LICENSE).
