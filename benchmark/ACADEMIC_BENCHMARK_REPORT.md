# Benchmark Report: 4-DOF Tabletop Grasping and Manipulation

## 1. Overview and Methodology

This benchmark evaluates geometric and learning-based grasp synthesis methods on a 4-DOF articulated manipulator equipped with a parallel-jaw Franka Hand gripper.

### Evaluation Criteria
1. **Grasp Success Rate (GSR)**: Percentage of physical rollouts where the object is grasped, lifted $\ge 5\ \text{cm}$ off the tabletop, and held stably for $\ge 3\ \text{seconds}$ in simulation.
2. **Force-Closure Verification**: Coulomb friction cone analysis ($\mu = 0.4$ and $\mu = 0.8$, Ferrari & Canny 1992, Nguyen 1988) on antipodal contact points.
3. **Kinematic Feasibility**: Analytical inverse kinematics feasibility within joint displacement and velocity limits.
4. **Collision Avoidance**: Table collision detection during pre-grasp approach and finger closing phases.
5. **Inference Latency**: Average computation time per grasp pose synthesis.

---

## 2. Benchmark Results

Evaluated across 99 grasp trials on standardized tabletop objects (Mug, Bottle, Duck, Torus, Disc, Box, Clutter):

| Method | Architecture | Success Rate (GSR) | Force-Closure ($\mu=0.8$) | IK Feasibility | Table Collisions | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA)** | CPU Algebraic | **77.8%** | 0.0% | 100.0% | 0.0% | **0.3 ± 0.1 ms** |
| **Pure Geometric (Normals)** | CPU Algebraic | **77.8%** | 0.0% | 100.0% | 0.0% | **110.7 ± 292.9 ms** |
| **Pure Geometric (OBB)** | CPU Algebraic | **66.7%** | 0.0% | 100.0% | 0.0% | **0.4 ± 0.1 ms** |
| **Pure Geometric (Slice)** | CPU Algebraic | **55.6%** | 0.0% | 100.0% | 0.0% | **0.5 ± 0.1 ms** |
| **Hybrid (AnyGrasp + OBB)** | GPU AI + CPU Geometry | **55.6%** | 0.0% | 100.0% | 0.0% | **127.5 ± 126.3 ms** |
| **Hybrid (AnyGrasp + PCA)** | GPU AI + CPU Geometry | **44.4%** | 0.0% | 100.0% | 0.0% | **98.4 ± 73.4 ms** |
| **Hybrid (AnyGrasp + Slice)** | GPU AI + CPU Geometry | **44.4%** | 0.0% | 100.0% | 0.0% | **163.8 ± 190.9 ms** |
| **Hybrid (AnyGrasp + Normals)** | GPU AI + CPU Geometry | **44.4%** | 0.0% | 100.0% | 0.0% | **304.1 ± 629.5 ms** |
| **Pure Geometric (RANSAC)** | CPU Algebraic | **33.3%** | 0.0% | 100.0% | 0.0% | **2.3 ± 1.5 ms** |
| **Hybrid (AnyGrasp + RANSAC)** | GPU AI + CPU Geometry | **22.2%** | 0.0% | 100.0% | 0.0% | **993.5 ± 1774.2 ms** |
| **Pure AnyGrasp (Baseline)** | Deep Network Only | **0.0%** | 0.0% | 0.0% | 66.7% | **132.6 ± 114.9 ms** |

---

## 3. Geometric Refinement Analysis

### Pure Geometric Methods (CPU Only)
Methods operating directly on 3D point cloud geometry achieve millisecond-level execution times without requiring GPU memory or model weights:
- **PCA (Principal Component Analysis)**: Computes covariance eigendecomposition of the object cluster. The grasp axis aligns perpendicular to the principal axis of elongation. Suitable for elongated and symmetric tools.
- **OBB (Oriented Bounding Box)**: Estimates minimum-volume bounding box via rotating calipers. Determines optimal grasp width and orientation. Suitable for rectangular packages and prismatic shapes.
- **Surface Normals**: Estimates local surface normals via $k$-NN covariance. Selects contact pairs maximizing antipodal alignment ($\mathbf{n}_1 \cdot \mathbf{n}_2 \approx -1$).
- **CrossSectionSlice**: Discretizes point cloud along the vertical axis to find cross-sectional necks or waists. Effective for bottles, cups, and glassware.
- **Primitive RANSAC**: Fits cylinders and planar patches to reduce sensor noise and identify primary grasp axes.

### Hybrid Integration (Deep Affordance + Geometric Filters)
Raw deep learning models (such as AnyGrasp) often predict contact affordances below or intersecting the supporting plane. Integrating geometric safety filters:
- Completely eliminated table collisions ($66.7\% \to 0.0\%$).
- Restored $100\%$ kinematic feasibility by projecting grasp poses into reachable Cartesian coordinates.
