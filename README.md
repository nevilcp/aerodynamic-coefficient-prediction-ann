# Aerodynamic Coefficients Prediction using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end machine learning pipeline that leverages Artificial Neural Networks (ANN) to predict aerodynamic force coefficients (Lift, Drag, and Pitching Moment) across varying airfoil geometries and flow regimes.**

### 💡 Business & Engineering Impact
Traditional Computational Fluid Dynamics (CFD) and wind tunnel tests are accurate but highly resource-intensive. This project solves that bottleneck by mapping complex, non-linear aerodynamic relationships into a highly optimized Neural Network—slashing compute time while maintaining accuracy across varying NACA airfoil geometries.

### 🚀 Technical Highlights
* **Automated Data Pipeline:** Built an end-to-end pipeline to clean, scale (unit variance), and normalize raw coordinate datasets across hundreds of flow regimes.
* **Empirical Architecture Tuning:** Conducted extensive ablation studies to identify the optimal balance between computational complexity and model generalization.
* **Robust Validation:** Evaluated performance rigorously using Root Mean Squared Error (RMSE), $R^2$ scores, and parity plots for Lift, Drag, and Pitching Moment.

### 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Data Engineering:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Environment:** Python 3.10+, Jupyter Notebook, Linux

### 🧠 System Architecture & Network Derivation
The final Multilayer Perceptron was not guessed; it was derived through structured ablation studies to optimize for fluid dynamic regression:

* **Deriving Network Depth:** Evaluated architectures ranging from 2 to 5 layers. A 4-layer structure was chosen as it maximized generalization, avoiding the performance degradation observed when adding a 5th layer.
* **Deriving Network Width:** Iteratively doubled layer neurons. The `[512, 256, 128, 3]` configuration hit the sweet spot—wider networks (e.g., starting with 1024 neurons) offered negligible performance gains at a massively inflated computational cost.
* **Deriving Input Resolution:** Tested geometric inputs of 5, 10, and 15 coordinate points. The model proved that just 10 sparse, cosine-spaced coordinates on the upper and lower surfaces were sufficient to capture the necessary spatial aerodynamic relationships.
* **Optimization Strategy:** Implemented the Adam optimizer (LR: $0.0005$) with a dynamic learning rate decay (10% reduction on a 5-epoch plateau) and ReLU activations to manage vanishing gradients.

### 📂 Repository Structure
Every file is structured to isolate distinct phases of the machine learning lifecycle, from data generation to model cross-validation.

```text
├── images/                                                 # Project visualizations and diagrams
│   ├── ANN Schematic.png
│   ├── Airfoil Nomenclature.png
│   ├── Cosine Spacing.png
│   ├── Drag Coefficient Prediction using ANN.png
│   ├── Lift Coefficient Prediction using ANN.png
│   └── Pitching Coefficient Prediction using ANN.png
├── notebooks/                                              # Exploratory Data Analysis & Model Prototyping
│   ├── NACA4D_05.ipynb                                     # Model training & testing on NACA 4-digit airfoils (5 coordinate points)
│   ├── NACA4D_10.ipynb                                     # Model training & testing on NACA 4-digit airfoils (10 coordinate points)
│   ├── NACA4D_15.ipynb                                     # Model training & testing on NACA 4-digit airfoils (15 coordinate points)
│   ├── NACA5D_05.ipynb                                     # Model training & testing on NACA 5-digit airfoils (5 coordinate points)
│   ├── NACA5D_10.ipynb                                     # Model training & testing on NACA 5-digit airfoils (10 coordinate points)
│   ├── NACA5D_15.ipynb                                     # Model training & testing on NACA 5-digit airfoils (15 coordinate points)
│   ├── NACA4D_10 - NACA5D_10.ipynb                         # Model training on NACA 4-digit & testing on NACA 5-digit airfoils (10 coordinate points)
│   ├── NACA5D_10 - NACA4D_10.ipynb                         # Model training on NACA 5-digit & testing on NACA 4-digit airfoils (10 coordinate points)
│   └── NACA4D_10 U NACA5D_10.ipynb                         # Model training on NACA 4-digit & NACA 5-digit airfoils (10 coordinate points)
├── scripts/                                                # Modular data generation pipeline (Javafoil automation)
│   ├── generate_NACA4D_05.py                               # Generates 5-point coordinate datasets for 4-digit series
│   ├── generate_NACA4D_10.py                               # Generates 10-point coordinate datasets for 4-digit series
│   ├── generate_NACA4D_15.py                               # Generates 15-point coordinate datasets for 4-digit series
│   ├── generate_NACA5D_05.py                               # Generates 5-point coordinate datasets for 5-digit series
│   ├── generate_NACA5D_10.py                               # Generates 10-point coordinate datasets for 5-digit series
│   └── generate_NACA5D_15.py                               # Generates 15-point coordinate datasets for 5-digit series
├── Project_1_Report.pdf                                    # Comprehensive technical report, methodology, and findings
└── README.md                              