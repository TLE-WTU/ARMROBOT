# ACADEMIC BENCHMARK REPORT: KINEMATICALLY CONSTRAINED 4-DOF MANIPULATOR GRASP SYNTHESIS
## Benchmark Evaluation on YCB Object Subset (5 Canonical Objects)

- **Date of Experiment:** 2026-10-17 21:25:56
- **Author / Investigator:** Lê Mạnh Tiến
- **Manipulator Kinematics:** 4-DOF Planar-Vertical Serial Manipulator ($L_1 = 0.25$ m, $L_2 = 0.20$ m, $L_h = 0.1334$ m, $h_0 = 0.175$ m)
- **End-Effector:** Franka Hand Parallel-Jaw Gripper (stroke $w_{\max} = 0.08$ m)
- **Kinematic Constraint:** Tangential closing axis locked to base azimuth $\psi_{\text{closing}} \equiv \text{atan2}(y, x) + \pi/2 \pmod{\pi}$
- **Reachable Workspace Envelope:** Curvilinear volume $r_{\max}(z) = \sqrt{0.45^2 - (z + 0.1334 - 0.175)^2}$
- **Benchmark Dataset:** Canonical YCB Object Subset (YcbTomatoSoupCan, YcbMustardBottle, YcbGelatinBox, YcbBanana, YcbFoamBrick)
- **Reference Standard:** Yale-CMU-Berkeley (YCB) Object and Model Set (Calli et al., *IEEE RAM* 2015).
- **Physical Asset Provenance:** High-fidelity meshes and collision URDFs from `eleramp/pybullet-object-models`.
- **Perception System:** Dual Dynamic Cameras with simulated depth sensor noise (Gaussian $\sigma = 2.0$ mm, dropout = $10\%$)
- **Sample Size:** $N = 30$ trials per method ($N_{\text{total}} = 360$ simulated rollouts)
- **Physical Evaluation Criteria:** Dual-finger physical contact, dynamic lift $\Delta z \ge 4$ cm, hold for $1.0$ second (240 steps at 240 Hz), and table collision check via `p.getContactPoints()`.

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
| 1 | **Pure Geometric (PCA)** | Pure Geometric (CPU) | **80.0%** [62.7%, 90.5%] | 3.3% [0.6%, 16.7%] | 86.7% | 0.0% | **0.31 ± 0.03** |
| 2 | **Pure Geometric (PrimitiveRANSAC)** | Pure Geometric (CPU) | **80.0%** [62.7%, 90.5%] | 13.3% [5.3%, 29.7%] | 86.7% | 0.0% | **1.64 ± 0.32** |
| 3 | **Pure Geometric (SurfaceNormals)** | Pure Geometric (CPU) | **80.0%** [62.7%, 90.5%] | 10.0% [3.5%, 25.6%] | 86.7% | 0.0% | **10.82 ± 3.16** |
| 4 | **Pure Geometric (CrossSectionSlice)** | Pure Geometric (CPU) | **56.7%** [39.2%, 72.6%] | 0.0% [0.0%, 11.4%] | 73.3% | 0.0% | **0.49 ± 0.10** |
| 5 | **Pure Geometric (OBB)** | Pure Geometric (CPU) | **36.7%** [21.9%, 54.5%] | 0.0% [0.0%, 11.4%] | 86.7% | 0.0% | **0.30 ± 0.04** |
| 6 | **Hybrid (AnyGrasp + PrimitiveRANSAC)** | Hybrid AI + Geometric (GPU+CPU) | **36.7%** [21.9%, 54.5%] | 13.3% [5.3%, 29.7%] | 90.0% | 0.0% | **6509.99 ± 1705.48** |
| 7 | **Hybrid (AnyGrasp + PCA)** | Hybrid AI + Geometric (GPU+CPU) | **26.7%** [14.2%, 44.4%] | 3.3% [0.6%, 16.7%] | 93.3% | 0.0% | **445.16 ± 102.65** |
| 8 | **Hybrid (AnyGrasp + OBB)** | Hybrid AI + Geometric (GPU+CPU) | **26.7%** [14.2%, 44.4%] | 0.0% [0.0%, 11.4%] | 90.0% | 0.0% | **628.91 ± 148.18** |
| 9 | **Hybrid (AnyGrasp + SurfaceNormals)** | Hybrid AI + Geometric (GPU+CPU) | **26.7%** [14.2%, 44.4%] | 3.3% [0.6%, 16.7%] | 90.0% | 0.0% | **670.90 ± 164.32** |
| 10 | **Hybrid (AnyGrasp + CrossSectionSlice)** | Hybrid AI + Geometric (GPU+CPU) | **23.3%** [11.8%, 40.9%] | 0.0% [0.0%, 11.4%] | 93.3% | 0.0% | **1233.86 ± 355.32** |
| 11 | **AnyGrasp (Raw 6-DOF Baseline)** | Deep Learning Baseline (GPU) | **20.0%** [9.5%, 37.3%] | 0.0% [0.0%, 11.4%] | 73.3% | 10.0% | **41.06 ± 53.68** |
| 12 | **AnyGrasp (4-DOF Adapted Baseline)** | Deep Learning Baseline (GPU) | **3.3%** [0.6%, 16.7%] | 0.0% [0.0%, 11.4%] | 96.7% | 10.0% | **140.21 ± 29.01** |

---

## 3. Candidate Filtration Funnel & Rejection Breakdown

To eliminate survivorship bias, candidate counts are audited across each filtering stage:

| Method | Total Generated | Rejected (Workspace) | Rejected (Table) | Rejected (4-DOF Yaw) | Final Valid Candidates | Rejection Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA)** | 30 | 4 | 0 | 0 | 26 | 13.3% |
| **Pure Geometric (PrimitiveRANSAC)** | 30 | 4 | 0 | 0 | 26 | 13.3% |
| **Pure Geometric (SurfaceNormals)** | 30 | 4 | 0 | 0 | 26 | 13.3% |
| **Pure Geometric (CrossSectionSlice)** | 30 | 8 | 0 | 0 | 22 | 26.7% |
| **Pure Geometric (OBB)** | 30 | 4 | 0 | 0 | 26 | 13.3% |
| **Hybrid (AnyGrasp + PrimitiveRANSAC)** | 30 | 3 | 0 | 0 | 27 | 10.0% |
| **Hybrid (AnyGrasp + PCA)** | 30 | 2 | 0 | 0 | 28 | 6.7% |
| **Hybrid (AnyGrasp + OBB)** | 30 | 3 | 0 | 0 | 27 | 10.0% |
| **Hybrid (AnyGrasp + SurfaceNormals)** | 30 | 3 | 0 | 0 | 27 | 10.0% |
| **Hybrid (AnyGrasp + CrossSectionSlice)** | 30 | 2 | 0 | 0 | 28 | 6.7% |
| **AnyGrasp (Raw 6-DOF Baseline)** | 30 | 8 | 3 | 13 | 6 | 80.0% |
| **AnyGrasp (4-DOF Adapted Baseline)** | 30 | 1 | 2 | 0 | 27 | 10.0% |

---

## 4. Statistical Hypothesis Testing (McNemar's Paired Test with Holm-Bonferroni Correction)

For matched pairs on identical scene seeds ($N = 30$), McNemar's test with Edwards' continuity correction was applied, and Family-Wise Error Rate (FWER) was controlled at $\alpha = 0.05$ using the Holm-Bonferroni step-down procedure:

| Comparison Pair | Discordant Pairs ($b / c$) | $\chi^2$ Statistic | Raw $p$-value | Holm Rank ($k$) | Holm Threshold | Adjusted $p$-value | Significance (FWER $\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Geometric (PCA) vs AnyGrasp (4-DOF Adapted)** | 23 / 0 | 21.043 | 0.0000 | 1 | 0.0100 | 0.0000 | **Significant (p_adj < 0.05)** |
| **Pure Geometric (PCA) vs AnyGrasp (Raw 6-DOF)** | 18 / 0 | 16.056 | 0.0001 | 2 | 0.0125 | 0.0004 | **Significant (p_adj < 0.05)** |
| **Pure PCA vs Pure OBB** | 13 / 0 | 11.077 | 0.0009 | 3 | 0.0167 | 0.0027 | **Significant (p_adj < 0.05)** |
| **Pure Geometric (PCA) vs Hybrid (AnyGrasp + PrimitiveRANSAC)** | 14 / 1 | 9.600 | 0.0019 | 4 | 0.0250 | 0.0038 | **Significant (p_adj < 0.05)** |
| **Pure CrossSectionSlice vs Pure OBB** | 8 / 2 | 2.500 | 0.1138 | 5 | 0.0500 | 0.1138 | Not Significant |

---

## 5. Ablation & Key Findings

1. **Impact of 4-DOF Kinematic Constraint:**
   The Franka Hand fingers move tangentially along $\psi_{\text{closing}} \equiv \theta_1 + \pi/2$. On asymmetric objects like the mustard bottle and banana, models that cannot rotate the wrist must align the grasp along the available closing direction.
2. **Computational Efficiency:**
   Analytical geometric synthesis achieves robust physical lift in **0.3 - 10.8 ms on CPU** compared to **41.1 - 6510.0 ms on GPU**, achieving over $4\times$ to $21797\times$ speedup while eliminating dedicated GPU hardware requirements.
3. **Sensor Depth Uncertainty Handling:**
   All experiments incorporate synthetic Gaussian sensor noise ($\sigma = 2.0$ mm) and 10% random dropout to emulate RealSense D435 characteristics. Volumetric PCA decomposition and primitive RANSAC maintain stable spatial orientation across the point cloud variations.

---

## 6. Reproducibility & Deployment Instructions

### Standalone Pure Geometric Mode (Zero-GPU, Embedded CPU)
```bash
PYTHONPATH=./:src/robot_arm pytest -v tests/
python3 benchmark/run_academic_benchmark.py --trials 6 --gripper_mm 80
```
