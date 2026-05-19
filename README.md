# Airfoil's Aerodynamic Coefficient Prediction using Artificial Neural Networks

## Overview

This repository contains the complete, end-to-end replication and extension of the methodology described in:

> Hassan Moin, Hafiz Zeeshan Iqbal Khan, Surrayya Mobeen, & Jamshed Riaz (2021). *Airfoil's Aerodynamic Coefficients Prediction using Artificial Neural Network*. 2022 19th International Bhurban Conference on Applied Sciences and Technology (IBCAST). [DOI: 10.1109/IBCAST54850.2022.9990112](https://doi.org/10.1109/IBCAST54850.2022.9990112). [arXiv:2109.12149](https://arxiv.org/abs/2109.12149) \[physics.flu-dyn\].

The central contribution is the training of multilayer perceptron networks -- using sparse, normalized two-dimensional airfoil surface coordinates as input -- to predict the lift coefficient ($C_L$), drag coefficient ($C_D$), and pitching moment coefficient ($C_m$) of NACA 4- and 5-digit airfoil families across a range of angle of attack, Reynolds number, and Mach number. This data-driven surrogate offers substantially faster coefficient estimation than either wind-tunnel experimentation or conventional Computational Fluid Dynamics (CFD), at acceptable levels of accuracy for preliminary aerodynamic design.

---

## Research Context and Motivation

Selecting an appropriate airfoil profile is a foundational step in the design of any aircraft or UAV. The aerodynamic coefficients -- $C_L$, $C_D$, and $C_m$ -- govern not only the lift and drag performance of a wing, but also feed into subsystem design tasks such as flight control law development and aeroelastic stability prediction. Conventionally, these coefficients are obtained either through wind-tunnel testing or through CFD simulation of the Navier–Stokes equations, both of which are computationally and experimentally expensive, especially when surveying large parametric design spaces.

Artificial Neural Networks (ANNs) have emerged as a practical alternative in such contexts. Once trained, they provide near-instantaneous coefficient estimates without resolving fluid equations at each query point. The present study investigates the feasibility, accuracy, and generalization capability of multilayer perceptron models for this task, systematically examining the influence of network depth, width, and the resolution of geometric input data.

---

## Repository Structure

```text
aerodynamic-coefficient-prediction-ann/
│
├── images/                      # Figures, conceptual diagrams, and analysis plots
│
├── notebooks/                   # Jupyter notebooks covering individual and combined dataset analysis
│   ├── NACA4D_05.ipynb
│   ├── NACA4D_10 - NACA5D_10.ipynb
│   ├── NACA4D_10 U NACA5D_10.ipynb
│   ├── NACA4D_10.ipynb
│   ├── NACA4D_15.ipynb
│   ├── NACA5D_05.ipynb
│   ├── NACA5D_10 - NACA4D_10.ipynb
│   ├── NACA5D_10.ipynb
│   └── NACA5D_15.ipynb
│
├── scripts/                     # Python scripts used to generate the datasets
│   ├── generate_NACA4D_05.py
│   ├── generate_NACA4D_10.py
│   ├── generate_NACA4D_15.py
│   ├── generate_NACA5D_05.py
│   ├── generate_NACA5D_10.py
│   └── generate_NACA5D_15.py
│
├── Project_1_Report.pdf
└── README.md
```

---

## Methodology

### Airfoil Parameterization and Dataset Generation

Aerodynamic data for NACA 4- and 5-digit airfoils were generated using **JavaFoil** via automated macro scripts. Each airfoil geometry was discretized at 101 cosine-spaced points along the unit chord, producing smooth representations of both the upper and lower surfaces. The use of cosine spacing ensures denser point distribution near the leading and trailing edges, where curvature -- and hence aerodynamic sensitivity -- is greatest.

The aerodynamic coefficients ($C_L$, $C_D$, $C_m$) were evaluated using **CalcFoil** as the stall modelling approach, with the **Drela $e^n$ method** (post-1991 XFoil transition model) for boundary-layer transition prediction. These choices reflect a balance between computational tractability and accuracy for subsonic, incompressible analysis.

Six datasets were constructed -- three for the NACA 4-digit family and three for the NACA 5-digit family -- differing in the number of chordwise coordinate sample points ($N \in \{5, 10, 15\}$):

| Dataset Name | Airfoil Family | Coordinate Points ($N$) |
|---|---|---|
| NACA4D\_05 | NACA 4-digit | 5 |
| NACA4D\_10 | NACA 4-digit | 10 |
| NACA4D\_15 | NACA 4-digit | 15 |
| NACA5D\_05 | NACA 5-digit | 5 |
| NACA5D\_10 | NACA 5-digit | 10 |
| NACA5D\_15 | NACA 5-digit | 15 |

The parameter ranges surveyed are summarized below:

| Parameter | Range | Step |
|---|---|---|
| Thickness (% chord) | 5 - 35 | 5 |
| Location of max. thickness (% chord) | 30 (fixed) | -- |
| Max. camber, 4-digit (% chord) | 0 - 9 | 1 |
| Design $C_L$, 5-digit | 0 - 1.8 | 0.2 |
| Location of max. camber (% chord) | 5 - 75 | 10 |
| Angle of attack (deg) | -10 - 10 | 1 |
| Reynolds number | $1\times10^5$ - $5\times10^5$ | $1\times10^5$ |
| Mach number | 0.1 - 0.3 | 0.1 |

This yields 560 unique NACA 4-digit geometries and 1,120 unique NACA 5-digit geometries, each evaluated at 315 flight conditions. After removal of erroneous samples (arising from JavaFoil convergence failures at extreme camber conditions, and 5-digit cases where the trailing edge did not lie on the chord axis within a tolerance of 0.015 non-dimensionalized chord lengths), the final valid sample counts are:

- **NACA 4-digit datasets:** 176,400 samples (all cases retained)
- **NACA 5-digit datasets:** 236,880 samples (after filtering)

### Data Preprocessing

Each dataset was processed in a dedicated Jupyter notebook. For NACA 4-digit samples, the geometric feature columns (thickness, camber, position of maximum camber) were discarded prior to training, leaving only the interpolated surface coordinates and flow conditions as inputs. An equivalent procedure was applied to the 5-digit datasets. The retained input vector for each sample thus comprised $2N$ surface coordinate values ($y_{U,k}$ and $y_{L,k}$ for $k \in [1, N]$) together with angle of attack, Reynolds number, and Mach number -- a total of $2N + 3$ features. The three outputs are $C_L$, $C_D$, and $C_m$.

All datasets were randomly shuffled and partitioned into training, validation, and test subsets at a **70:15:15** ratio. Input features were standardized (zero mean, unit variance) using statistics computed solely on the training set; the same parameters were applied to the validation and test sets.

### Network Architecture

All models were implemented in **Python** using **TensorFlow/Keras**. The architectural choices mirror the reference paper closely:

- **Optimizer:** Adam ($\text{lr} = 0.0005$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Loss function:** Mean Squared Error (MSE)
- **Evaluation metrics:** Root Mean Squared Error (RMSE), $R^2$
- **Activation function:** Rectified Linear Unit (ReLU) in all hidden layers
- **Training duration:** 50 epochs, mini-batch size 128
- **Learning rate schedule:** Reduced by 10% if validation loss does not improve for 5 consecutive epochs (patience = 5)
- **Output layer:** 3 neurons (one each for $C_L$, $C_D$, $C_m$), no activation

The final architecture selected from systematic experiments is **[512, 256, 128, 3]**.

---

## Experimental Design and Results

All results were obtained by averaging over **20 independent training runs** per configuration, with the model achieving the best average test loss reported. This procedure guards against sensitivity to random weight initialization.

### Effect of Network Depth

Holding the neuron count per layer fixed (halving at each successive layer), network depth was varied from two to five layers. Performance consistently improved from two to four hidden layers across all six datasets; beyond four layers, gains diminished or reversed slightly. The four-layer configuration [64, 32, 16, 3] was adopted as the baseline depth.

Representative results for NACA4D\_10:

| Case | Architecture | RMSE ($C_L$) | RMSE ($C_D$) | RMSE ($C_m$) | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|---|---|---|
| 1 | [64, 3] | 0.005312 | 0.004230 | 0.002296 | 0.999967 | 0.846132 | 0.999583 |
| 2 | [64, 32, 3] | 0.004231 | 0.003978 | 0.001881 | 0.999979 | 0.863597 | 0.999721 |
| **3** | **[64, 32, 16, 3]** | **0.004128** | **0.003866** | **0.001939** | **0.999980** | **0.871552** | **0.999701** |
| 4 | [64, 32, 16, 8, 3] | 0.003975 | 0.003922 | 0.001817 | 0.999981 | 0.867454 | 0.999738 |

### Effect of Network Width

Using the four-layer depth, the number of neurons per layer was progressively doubled. Prediction accuracy improved monotonically with width, but the marginal gain between Case 4 [512, 256, 128, 3] and Case 5 [1024, 512, 256, 3] was negligible relative to the added computational cost. **Case 4 was selected as the final architecture.**

Representative results for NACA4D\_10:

| Case | Architecture | RMSE ($C_L$) | RMSE ($C_D$) | RMSE ($C_m$) | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|---|---|---|
| 1 | [64, 32, 16, 3] | 0.004142 | 0.003848 | 0.001800 | 0.999980 | 0.872731 | 0.999745 |
| 2 | [128, 64, 32, 3] | 0.003152 | 0.003521 | 0.001312 | 0.999888 | 0.893397 | 0.999864 |
| 3 | [256, 128, 64, 3] | 0.002699 | 0.003251 | 0.000991 | 0.999990 | 0.909062 | 0.999923 |
| **4** | **[512, 256, 128, 3]** | **0.002267** | **0.003147** | **0.000868** | **0.999994** | **0.914501** | **0.999940** |
| 5 | [1024, 512, 256, 3] | 0.002266 | 0.003086 | 0.000864 | 0.999993 | 0.917466 | 0.999939 |

### Effect of Geometric Resolution

The final architecture [512, 256, 128, 3] was evaluated across the three coordinate resolutions for the NACA 4-digit family. All three datasets yielded $R^2 > 0.90$ for every output, demonstrating that the network can extract meaningful aerodynamic information even from very sparse coordinate representations. NACA4D\_10 ($N=10$) offered the best overall balance and was selected as the primary dataset.

| Dataset | RMSE ($C_L$) | RMSE ($C_D$) | RMSE ($C_m$) | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|---|---|
| NACA4D\_05 | 0.002750 | 0.003425 | 0.000968 | 0.999990 | 0.904181 | 0.999922 |
| **NACA4D\_10** | **0.002267** | **0.003147** | **0.000868** | **0.999994** | **0.914501** | **0.999940** |
| NACA4D\_15 | 0.002013 | 0.003299 | 0.000912 | 0.999995 | 0.911276 | 0.999931 |

### Cross-Family Generalization

A key question addressed in this study is whether a network trained on one NACA family can transfer to the other. Results show it can, with appreciable accuracy:

| Train Set | Test Set | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|
| NACA4D\_10 | NACA5D\_10 | 0.995584 | 0.471389 | 0.881492 |
| NACA5D\_10 | NACA4D\_10 | 0.999092 | 0.722853 | 0.988516 |
| NACA4D\_10 ∪ NACA5D\_10 | NACA4D\_10 ∪ NACA5D\_10 | 0.999995 | 0.890248 | 0.999941 |

This behavior is interpreted as evidence that the network learns the **collective geometric design space** -- the envelope of all possible normalized coordinate distributions -- rather than individual airfoil profiles. Consequently, it generalizes to unseen geometries that lie within this space.

The drag coefficient consistently yielded the lowest $R^2$ in cross-family tests, which is physically expected: $C_D$ is a quadratic function of angle of attack (the drag polar), and its curvature is harder to capture when the network has been trained primarily on data where lift and moment vary linearly with $\alpha$. A proposed remedy, noted in both the original paper and this replication, is to decompose $C_D$ into its parasitic ($C_{D_0}$) and lift-induced ($K C_L^2$) components and predict them separately.

---

## Key Findings

1. **ANNs are viable aerodynamic surrogates.** Multilayer perceptrons trained on normalized surface coordinates predict $C_L$ and $C_m$ with $R^2 > 0.999$ on in-distribution test data, and $C_D$ with $R^2 > 0.91$ -- sufficient for preliminary design and real-time control applications.

2. **Geometric sparsity is not a barrier.** The network achieves $R^2 > 0.90$ on all outputs even when only five chordwise coordinate pairs are supplied per airfoil, a level of geometric information far below what is required by CFD meshes.

3. **Four hidden layers with [512, 256, 128, 3] neurons represent the optimal depth–width tradeoff** within the tested configurations. Shallower or narrower networks underfit; deeper or wider networks offer diminishing returns at greater computational cost.

4. **Cross-family generalization is robust for $C_L$ and $C_m$, but limited for $C_D$ in single-direction transfer.** Training on the combined dataset removes this limitation almost entirely. The network appears to learn the geometric design space rather than memorizing individual profiles.

5. **Drag prediction in the stall regime remains a known limitation.** The restricted angle-of-attack range (−10° to +10°) and the predominance of linear $C_L$–$\alpha$ samples in the dataset introduce a mild bias that causes the model to underperform near stall angles, particularly for airfoils such as NACA 2412 and NACA 6408.

---

## Software Dependencies

| Package | Purpose |
|---|---|
| Python ≥ 3.8 | Core language |
| TensorFlow ≥ 2.x / Keras | Neural network construction and training |
| NumPy | Numerical array operations |
| Pandas | Dataset loading and manipulation |
| Scikit-learn | Data splitting and standardization |
| Matplotlib | Result visualization |
| Jupyter | Interactive notebook execution |
| JavaFoil | Aerodynamic data generation (external, not a Python package) |

---

