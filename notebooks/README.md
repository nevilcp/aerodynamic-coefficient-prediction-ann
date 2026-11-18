# Notebook replication notes

Use the repository's reference [airfoil-aerodynamic-coefficients-ann.pdf](../airfoil-aerodynamic-coefficients-ann.pdf). `report.pdf` is not the replication specification.

Run notebooks from the repository root or `notebooks/` using a kernel with `requirements.txt` installed. Each training configuration defaults to 20 runs of 50 epochs. `N_RUNS` and `EPOCHS` can be reduced for a smoke run, but those results must not be reported as full replication results. Saved execution outputs are cleared so reduced checks are not presented as full training results.

Figures are saved as PNG only under `images/<notebook-name>/`, using the exact notebook filename without `.ipynb` as the directory name. Image filenames use lowercase letters and hyphens. Each learning curve is accompanied by mean test RMSE and R² for CL, CD, and Cm. Architectures are ranked using validation MSE normalized by training-target variance; parity plots use the fixed seed-42 run. The two transfer notebooks retain their source-family-only training and validation splits.

| Reference content | Notebook coverage |
| --- | --- |
| Tables II–III: depth and width experiments | Six single-family notebooks retain the experiment tables; architecture metric plots supplement them. |
| Table IV: coordinate resolution comparison | Compare the final [512, 256, 128, 3] scores across the 05, 10, and 15 notebooks. |
| Table V: transfer and combined-family experiments | The two directional notebooks and the union notebook. |
| Figure 4: geometric design space | Union notebook plots both families' geometry envelopes. |
| Figures 5–7: CL, Cm, CD versus angle of attack | Union notebook contains plotting code and a validated input contract for `results/reference_airfoils.csv`. Independent JavaFoil data for NACA 0045, 2412, 6408, and 136138 at M=0.25 and Re=250000 is required and is not included in the existing training CSVs. |
| Training/validation loss and held-out parity | Added to all nine notebooks as supplementary diagnostics, not figures claimed to appear in the paper. |

Figures 1–3 are explanatory nomenclature, spacing, and network diagrams rather than missing experimental results. The design-space plot uses the cosine locations used by this repository's generators. The generators discard the two fixed endpoints and retain the interior coordinate samples.

Targets are unscaled physical coefficients. Hidden layers use ReLU and the output layer is linear to allow negative lift and moment. Metrics remain means over all runs; the paper's wording about reporting models with best average test loss is ambiguous. These notebooks do not select models on test performance. Train/validation/test splits and the source-family-only 80/20 split for transfer experiments are retained.

Run `ruff check notebooks` and `ruff format notebooks --check` for lint and formatting checks. Full training requires the scientific dependencies and substantial compute; the current notebook outputs have not been validated by a full training run.
