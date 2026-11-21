# Airfoil Aerodynamic Coefficient Prediction with Neural Networks

This project trains multilayer perceptrons to predict lift ($C_L$), drag ($C_D$), and pitching moment ($C_m$) for NACA 4- and 5-digit airfoils. The inputs are sparse surface coordinates, angle of attack, Reynolds number, and Mach number. The experiments reproduce the paper's depth, width, resolution, and transfer comparisons and add a benchmark for airfoil shapes held out of training.

The replication reference is Moin, Khan, Mobeen, and Riaz, *Airfoil’s Aerodynamic Coefficients Prediction using Artificial Neural Network*, presented at IBCAST 2022. [Conference paper](https://doi.org/10.1109/IBCAST54850.2022.9990112) · [Preprint](https://arxiv.org/abs/2109.12149)

## Setup

From the repository root, create a virtual environment and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the six training datasets (about 340 MB). The notebooks look for each CSV at `results/<dataset>/<dataset>.csv` first, then at `results/data/<dataset>.csv`. [Hugging Face dataset](https://huggingface.co/datasets/nevilcp/aerodynamic-coefficient-prediction-ann)

```bash
hf download nevilcp/aerodynamic-coefficient-prediction-ann --repo-type dataset --local-dir results
jupyter notebook notebooks/
```

JavaFoil is needed to regenerate the training datasets with the generator scripts. The notebooks train directly from the CSVs. [Dataset generators](scripts/)

To reproduce the reference-airfoil plots, prepare the independent polars before running the pooled notebook. [Reference data workflow](notebooks/README.md#reference-airfoil-data)

```bash
python scripts/import_reference_airfoils.py
```

## Experiments

| Question | Notebooks |
|---|---|
| How do depth, width, and coordinate resolution affect the scores? | Six single-family notebooks, one for each family and resolution |
| How well does a model trained on one NACA family predict the other? | Two directional 10-point transfer notebooks |
| How well does a single model predict held-out rows from both families? | The pooled 10-point notebook, which also compares predictions with published reference polars |
| How well does the model predict shapes held out of training? | `geometry_generalization.ipynb`, which includes matched resolution comparisons, a pooled geometry split, and transfer scores with shared profiles excluded |

The notebook guide links every experiment and explains which exports are needed to rebuild the tables below. [Notebook guide](notebooks/README.md)

## Methodology

### Dataset Generation

The generators use JavaFoil to sample 101 cosine-spaced geometry points and evaluate the aerodynamic coefficients with the CalcFoil stall model and Drela $e^n$ transition model. They resample each surface at 5, 10, or 15 interior chordwise positions.

The six training CSVs contain the following row counts after filtering.

| Family | 5 points | 10 points | 15 points |
|---|---:|---:|---:|
| NACA 4-digit | 176,400 | 176,400 | 176,400 |
| NACA 5-digit | 236,880 | 236,670 | 236,880 |

The generators sample airfoil parameters and operating conditions over these ranges.

| Parameter | Range | Step |
|---|---|---|
| Thickness (% chord) | 5 to 35 | 5 |
| Position of maximum thickness (% chord) | 30 | Fixed |
| Maximum camber, 4-digit (% chord) | 0 to 9 | 1 |
| Design $C_L$, 5-digit | 0 to 1.8 | 0.2 |
| Position of maximum camber (% chord) | 5 to 75 | 10 |
| Angle of attack (degrees) | −10 to 10 | 1 |
| Reynolds number | $1\times10^5$ to $5\times10^5$ | $1\times10^5$ |
| Mach number | 0.1 to 0.3 | 0.1 |

### Model and Evaluation

At resolution $N$, the model takes $2N$ surface coordinates and three operating conditions. Features are standardized using the training split. The models use ReLU hidden layers, three linear outputs, and MSE loss on unscaled coefficient targets. Each configuration runs for 50 epochs with batch size 128 and Adam at a learning rate of 0.0005. The rate is multiplied by 0.9 after five epochs without improvement in validation loss, down to a minimum of 0.00001.

| Protocol | Split | What the test measures |
|---|---|---|
| Single-family and pooled row splits | 70% training, 15% validation, 15% test by row | Prediction on held-out operating-condition rows. Every test row has an exact surface-coordinate match in training. |
| Directional transfer | 80% training and 20% validation within the source family, with the full target family as test | Transfer from one family to the other. The two 10-point datasets share seven exact coordinate profiles. |
| Geometry-grouped benchmark | Approximately 70% training, 15% validation, 15% test by geometry group | Prediction for shapes excluded from training. The three resolutions use the same matched cases and split assignments within each family. |

## Experimental Design and Results

The tables below summarize the original row-split and transfer experiments using means across 20 seeded runs. Architecture ranking in the single-family sweeps uses validation MSE normalized by training-target variance. Resolution and transfer comparisons use `[512, 256, 128, 3]`.

### Effect of Network Depth

The depth sweep uses one to four hidden layers, starting at 64 neurons and halving the width at each successive layer. The three-hidden-layer model `[64, 32, 16, 3]` is the width sweep baseline. The table shows test results for `NACA4D_10`.

| Case | Architecture | RMSE ($C_L$) | RMSE ($C_D$) | RMSE ($C_m$) | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|---|---|---|
| 1 | [64, 3] | 0.005048 | 0.004330 | 0.002214 | 0.999970 | 0.846111 | 0.999612 |
| 2 | [64, 32, 3] | 0.004124 | 0.004050 | 0.001806 | 0.999980 | 0.865516 | 0.999742 |
| **3** | **[64, 32, 16, 3]** | **0.003994** | **0.003983** | **0.001810** | **0.999981** | **0.869982** | **0.999739** |
| 4 | [64, 32, 16, 8, 3] | 0.004033 | 0.004017 | 0.001882 | 0.999981 | 0.867552 | 0.999717 |

### Effect of Network Width

With three hidden layers, the sweep doubles their widths at each step. On `NACA4D_10`, `[1024, 512, 256, 3]` had the lowest mean normalized validation MSE of 0.024982 with 681,475 parameters. The comparison model `[512, 256, 128, 3]` scored 0.025561 with 176,899 parameters.

| Case | Architecture | RMSE ($C_L$) | RMSE ($C_D$) | RMSE ($C_m$) | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|---|---|---|
| 1 | [64, 32, 16, 3] | 0.003994 | 0.003983 | 0.001810 | 0.999981 | 0.869982 | 0.999739 |
| 2 | [128, 64, 32, 3] | 0.003262 | 0.003644 | 0.001334 | 0.999987 | 0.891151 | 0.999856 |
| 3 | [256, 128, 64, 3] | 0.002481 | 0.003328 | 0.000986 | 0.999992 | 0.909163 | 0.999922 |
| **4** | **[512, 256, 128, 3]** | **0.002655** | **0.003174** | **0.000961** | **0.999991** | **0.917363** | **0.999925** |
| 5 | [1024, 512, 256, 3] | 0.002059 | 0.003112 | 0.000850 | 0.999995 | 0.920553 | 0.999941 |

### Effect of Geometric Resolution

For the fixed architecture, the 4-digit mean normalized validation MSE was 0.031509 at 5 points, 0.025561 at 10 points, and 0.026138 at 15 points. The transfer experiments use the 10-point datasets. The table shows the corresponding 4-digit test scores.

| Dataset | RMSE ($C_L$) | RMSE ($C_D$) | RMSE ($C_m$) | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|---|---|
| NACA4D\_05 | 0.002176 | 0.003110 | 0.000907 | 0.999994 | 0.920306 | 0.999934 |
| **NACA4D\_10** | **0.002655** | **0.003174** | **0.000961** | **0.999991** | **0.917363** | **0.999925** |
| NACA4D\_15 | 0.002357 | 0.003337 | 0.000924 | 0.999993 | 0.909303 | 0.999931 |

### Cross-Family Generalization

Each transfer row tests on the full opposite family. The pooled row reports scores on held-out rows from both families using the pooled model.

| Train Set | Evaluation Set | $R^2$ ($C_L$) | $R^2$ ($C_D$) | $R^2$ ($C_m$) |
|---|---|---|---|---|
| NACA4D\_10 source split | Full NACA5D\_10 target family | 0.995300 | 0.495387 | 0.854205 |
| NACA5D\_10 source split | Full NACA4D\_10 target family | 0.999258 | 0.724293 | 0.988522 |
| NACA4D\_10 ∪ NACA5D\_10 training rows | Held-out rows from both families | 0.999994 | 0.888772 | 0.999927 |

Drag had the lowest $R^2$ in both transfer directions. Pitching moment was more sensitive to direction than lift, with an $R^2$ of 0.854205 for 4→5 and 0.988522 for 5→4.

### Unseen-Geometry Benchmark

The pooled 10-point geometry benchmark holds out entire airfoil profiles from both NACA families. With the same `[512, 256, 128, 3]` architecture, the 20-run results on 61,950 test rows are shown below. Drag $R^2$ was 0.853584 on held-out profiles and 0.888772 on the pooled row split. [Geometry benchmark notebook](notebooks/geometry_generalization.ipynb)

| Metric | $C_L$ | $C_D$ | $C_m$ |
|---|---:|---:|---:|
| RMSE | 0.002326 | 0.004303 | 0.001062 |
| $R^2$ | 0.999994 | 0.853584 | 0.999933 |

## Reference-Airfoil Evaluation

The pooled notebook compares its seed-42 predictions with published polars for NACA 0045, 2412, 6408, and 136138 at Mach 0.25 and Reynolds number 250,000. The saved outputs include the geometry envelope with reference profiles and plots of $C_L$, $C_D$, and $C_m$ versus angle of attack. [Saved figures](images/NACA4D_10%20U%20NACA5D_10/) · [Reference data workflow](notebooks/README.md#reference-airfoil-data)

## Model Package

The pooled 10-point package contains a seed-42 Keras checkpoint trained on the geometry-grouped split, the fitted feature standardization, and a runnable prediction example. The model card reports the packaged checkpoint's scores and the 20-run benchmark separately. [GitHub Releases](https://github.com/nevilcp/aerodynamic-coefficient-prediction-ann/releases)

To rebuild the package, use the two 10-point training CSVs from Setup and the pooled split export produced by the geometry notebook. The release archive also includes this export as `benchmark.json.gz`, which can be passed to the build script with `--benchmark`.

```bash
python scripts/build_pooled_model.py naca-pooled-10-geometry
```
