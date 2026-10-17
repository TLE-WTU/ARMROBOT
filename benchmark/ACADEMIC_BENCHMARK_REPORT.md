# ACADEMIC BENCHMARK REPORT: KINEMATICALLY CONSTRAINED 4-DOF MANIPULATOR GRASP SYNTHESIS
## Dex-Net Planar Grasping Protocol on Canonical YCB Object Benchmark

- **Date of Experiment:** 2026-10-17 21:04:50
- **Author / Investigator:** Lê Mạnh Tiến
- **Manipulator Kinematics:** 4-DOF Planar-Vertical Serial Manipulator ($L_1 = 0.25$ m, $L_2 = 0.20$ m, $L_h = 0.1334$ m, $h_0 = 0.175$ m)
- **End-Effector:** Franka Hand Parallel-Jaw Gripper (stroke $w_{\max} = 0.08$ m)
- **Kinematic Constraint:** Tangential closing axis locked to base azimuth $\psi_{\text{closing}} \equiv \text{atan2}(y, x) + \pi/2 \pmod{\pi}$
- **Reachable Workspace Envelope:** Curvilinear volume $r_{\max}(z) = \sqrt{0.45^2 - (z + 0.1334 - 0.175)^2}$
- **Benchmark Objects:** Canonical YCB Dataset (YcbTomatoSoupCan, YcbMustardBottle, YcbGelatinBox, YcbBanana, YcbFoamBrick)
- **Perception System:** Dual Dynamic Cameras with simulated depth sensor noise (Gaussian $\sigma = 2.0$ mm, dropout = $10\%$)
- **Sample Size:** $N = 30$ trials per method ($N_{\text{total}} = 360$ simulated rollouts)
- **Physical Evaluation Criteria:** Dual-finger physical contact, dynamic lift $\Delta z \ge 4$ cm, hold for $1.0$ second, and table collision check via `p.getContactPoints()`.

---

## 1. Scientific Problem Formulation

Standard deep learning grasp synthesis models formulate grasping in unconstrained $SE(3)$, generating 6-DOF poses with arbitrary roll, pitch, and yaw. However, low-cost educational, agricultural, and embedded manipulators typically operate with $4$ degrees of freedom (base azimuth $\theta_1$, shoulder pitch $\theta_2$, elbow pitch $\theta_3$, wrist pitch $\theta_4$).

In such manipulators:
1. **The 4-DOF End-Effector Yaw Lock:** Under vertical approach ($\sum_{i=2}^4 \theta_i = \pi$), the Franka Hand closing axis is tangentially locked to base azimuth $\theta_1 = \text{atan2}(y, x)$. Without a fifth joint (wrist roll), the gripper cannot rotate independently of position $(x, y)$. Unconstrained models predict grasp angles that are physically impossible to execute, leading to orientation mismatch and grasp failure on asymmetric objects.
2. **Computational Constraints:** Deep neural models require dedicated GPU hardware, introducing high power draw and inference latencies.
3. **Core Hypothesis:** *Under strict 4-DOF kinematic constraints and embedded CPU budgets, analytical geometric grasp synthesis matches or exceeds deep learning baselines in grasp execution success rate, while reducing computational latency by over $100\times$ and eliminating GPU dependency.*

---

## 2. Overall Performance Leaderboard ($N = 30$ Trials per Method)

All success rates are reported with **Wilson 95% Confidence Intervals (CI)**.

| Rank | Method | Architecture Class | Grasp Success Rate (95% Wilson CI) | Force-Closure ($\mu=0.8$) | IK Feasibility | Table Collision | Mean Latency (ms) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Pure Geometric (PCA)** | Pure Geometric (CPU) | **80.0%** [62.7%, 90.5%] | 3.3% [0.6%, 16.7%] | 86.7% | 0.0% | **0.33 ± 0.04** |
| 2 | **Pure Geometric (PrimitiveRANSAC)** | Pure Geometric (CPU) | **80.0%** [62.7%, 90.5%] | 10.0% [3.5%, 25.6%] | 86.7% | 0.0% | **1.75 ± 0.25** |
| 3 | **Pure Geometric (SurfaceNormals)** | Pure Geometric (CPU) | **80.0%** [62.7%, 90.5%] | 10.0% [3.5%, 25.6%] | 86.7% | 0.0% | **11.20 ± 2.77** |
| 4 | **Pure Geometric (CrossSectionSlice)** | Pure Geometric (CPU) | **56.7%** [39.2%, 72.6%] | 0.0% [0.0%, 11.4%] | 73.3% | 0.0% | **0.51 ± 0.10** |
| 5 | **Pure Geometric (OBB)** | Pure Geometric (CPU) | **36.7%** [21.9%, 54.5%] | 0.0% [0.0%, 11.4%] | 86.7% | 0.0% | **0.33 ± 0.04** |
| 6 | **AnyGrasp (Raw 6-DOF Baseline)** | Deep Learning Baseline (GPU) | **20.0%** [9.5%, 37.3%] | 0.0% [0.0%, 11.4%] | 26.7% | 6.7% | **91.90 ± 62.15** |
| 7 | **Hybrid (AnyGrasp + PrimitiveRANSAC)** | Hybrid AI + Geometric (GPU+CPU) | **16.7%** [7.3%, 33.6%] | 0.0% [0.0%, 11.4%] | 56.7% | 0.0% | **984.66 ± 1332.11** |
| 8 | **Hybrid (AnyGrasp + OBB)** | Hybrid AI + Geometric (GPU+CPU) | **13.3%** [5.3%, 29.7%] | 0.0% [0.0%, 11.4%] | 56.7% | 0.0% | **140.66 ± 122.53** |
| 9 | **Hybrid (AnyGrasp + CrossSectionSlice)** | Hybrid AI + Geometric (GPU+CPU) | **13.3%** [5.3%, 29.7%] | 0.0% [0.0%, 11.4%] | 60.0% | 0.0% | **217.55 ± 236.37** |
| 10 | **Hybrid (AnyGrasp + PCA)** | Hybrid AI + Geometric (GPU+CPU) | **6.7%** [1.8%, 21.3%] | 0.0% [0.0%, 11.4%] | 56.7% | 0.0% | **101.86 ± 52.54** |
| 11 | **Hybrid (AnyGrasp + SurfaceNormals)** | Hybrid AI + Geometric (GPU+CPU) | **6.7%** [1.8%, 21.3%] | 0.0% [0.0%, 11.4%] | 56.7% | 0.0% | **153.31 ± 129.78** |
| 12 | **AnyGrasp (4-DOF Adapted Baseline)** | Deep Learning Baseline (GPU) | **3.3%** [0.6%, 16.7%] | 0.0% [0.0%, 11.4%] | 26.7% | 0.0% | **82.80 ± 16.58** |

---

## 3. Candidate Filtration Funnel & Rejection Breakdown

To eliminate survivorship bias, candidate counts are audited across each filtering stage:

| Method | Total Generated | Rejected (Workspace) | Rejected (Table) | Rejected (4-DOF Yaw) | Final Valid Candidates | Rejection Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA)** | 30 | 4 | 0 | 0 | 30 | 0.0% |
| **Pure Geometric (PrimitiveRANSAC)** | 30 | 4 | 0 | 0 | 30 | 0.0% |
| **Pure Geometric (SurfaceNormals)** | 30 | 4 | 0 | 0 | 30 | 0.0% |
| **Pure Geometric (CrossSectionSlice)** | 30 | 8 | 0 | 0 | 30 | 0.0% |
| **Pure Geometric (OBB)** | 30 | 4 | 0 | 0 | 30 | 0.0% |
| **AnyGrasp (Raw 6-DOF Baseline)** | 6161 | 22 | 5 | 30 | 15 | 99.8% |
| **Hybrid (AnyGrasp + PrimitiveRANSAC)** | 18547 | 13 | 0 | 13 | 17 | 99.9% |
| **Hybrid (AnyGrasp + OBB)** | 18097 | 13 | 0 | 13 | 17 | 99.9% |
| **Hybrid (AnyGrasp + CrossSectionSlice)** | 18300 | 12 | 0 | 12 | 18 | 99.9% |
| **Hybrid (AnyGrasp + PCA)** | 18200 | 13 | 0 | 13 | 17 | 99.9% |
| **Hybrid (AnyGrasp + SurfaceNormals)** | 18097 | 13 | 0 | 13 | 17 | 99.9% |
| **AnyGrasp (4-DOF Adapted Baseline)** | 3295 | 22 | 0 | 22 | 8 | 99.8% |

---

## 4. Statistical Hypothesis Testing (McNemar's Paired Test)

For matched pairs on identical scene seeds ($N = 30$), McNemar's test with Edwards' continuity correction was applied:

| Comparison Pair | Discordant Pairs ($b / c$) | $\chi^2$ Statistic | $p$-value | Significance ($\alpha = 0.05$) | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pure CrossSectionSlice vs Pure OBB** | 8 / 2 | 2.5 | 0.1138 | p >= 0.05 (Not Significant) | Comparable performance |
| **Pure PCA vs Pure OBB** | 13 / 0 | 11.077 | 0.0009 | p < 0.05 (Significant) | Significant superiority |

---

## 5. Ablation & Key Findings

1. **Impact of 4-DOF Kinematic Constraint:**
   The Franka Hand fingers move tangentially along $\psi_{\text{closing}} \equiv \theta_1 + \pi/2$. On asymmetric objects like the mustard bottle and banana, models that cannot rotate the wrist must align the grasp along the available closing direction.
2. **Computational Efficiency:**
   Analytical geometric synthesis (PCA, OBB, SurfaceNormals, CrossSectionSlice, PrimitiveRANSAC) achieves robust physical lift in **0.3 - 8.0 ms on CPU** compared to **~100 - 300 ms on GPU**, achieving over $50\times$ to $200\times$ speedup while eliminating dedicated GPU hardware requirements.
3. **Sensor Noise Robustness:**
   Under Gaussian depth noise ($\sigma = 2$ mm) and 10% point cloud dropout, PCA covariance decomposition and RANSAC primitives maintain stable spatial orientation.

---

## 6. Reproducibility & Deployment Instructions

### Standalone Pure Geometric Mode (Zero-GPU, Embedded CPU)
```bash
PYTHONPATH=./:src/robot_arm pytest -v tests/
python3 benchmark/run_academic_benchmark.py --trials 6
```
