# Notebook Guide

The notebooks reproduce the paper's original comparisons and include a separate evaluation for unseen airfoil shapes. For dependencies and training CSVs, see the project setup. [Project setup](../README.md#setup) · [Reference paper](https://arxiv.org/abs/2109.12149)

## Notebook Index

| Experiment | Notebooks |
|---|---|
| 4-digit depth and width sweeps | [5 points](NACA4D_05.ipynb), [10 points](NACA4D_10.ipynb), [15 points](NACA4D_15.ipynb) |
| 5-digit depth and width sweeps | [5 points](NACA5D_05.ipynb), [10 points](NACA5D_10.ipynb), [15 points](NACA5D_15.ipynb) |
| Directional transfer | [4-digit → 5-digit](NACA4D_10%20-%20NACA5D_10.ipynb), [5-digit → 4-digit](NACA5D_10%20-%20NACA4D_10.ipynb) |
| Pooled 10-point model and reference polars | [Both families](NACA4D_10%20U%20NACA5D_10.ipynb) |
| Matched resolutions and unseen geometries | [Geometry generalization](geometry_generalization.ipynb) |

The single-family sweeps cover the paper's depth and width comparisons (Tables II–III). To reproduce its resolution comparison (Table IV), compare the fixed `[512, 256, 128, 3]` architecture across the three 4-digit notebooks. The directional and pooled notebooks cover Table V.

The six single-family notebooks and the pooled notebook use 70:15:15 row splits. Every test geometry also appears in training. The directional notebooks train and validate on an 80:20 split of the source family and test on the full target family. Each directional and pooled notebook reports mean test RMSE and R² in a single table, with one column per coefficient. The single-family sweeps rank architectures by validation MSE normalized by training-target variance.

Each plotting notebook exports vector PDFs with embedded fonts and 600 dpi PNGs under `images/<notebook-name>/`. Captions are saved alongside the figures as text files. The notebook displays smaller previews for interactive use.

## Geometry Generalization

The geometry notebook matches the same airfoil and operating conditions at all three resolutions before splitting. It keeps all rows for equivalent shapes together and reuses each family's split assignments across 5, 10, and 15 points. Its three results sections cover:

- **Coordinate resolution.** Six configurations compare the three resolutions within each family.
- **Pooled geometry benchmark.** One 10-point model tests on shapes held out from both families.
- **Cross-family transfer.** Two configurations repeat the original directional splits. Each reports scores for the full target family and for the subset left after excluding exact coordinate matches to any source-family profile.

## Reference-Airfoil Data

The pooled notebook uses published polars for NACA 0045, 2412, 6408, and 136138 at Mach 0.25 and Reynolds number 250,000. The input contains 41 angles per profile, from −10° to 10° in half-degree increments. Both the published profiles and the training inputs use ten interior cosine-spaced chordwise positions. [Published reference data](https://github.com/ZeeshanIqbalKhan/AirfoilANN/tree/b2d0691881e7d5a5db121aec92004c69d48ab5c0/Data) · [Source coordinate conversion](https://github.com/ZeeshanIqbalKhan/AirfoilANN/blob/b2d0691881e7d5a5db121aec92004c69d48ab5c0/Matlab%20Scripts/ReadAirfoilData.m)

To recreate `results/reference_airfoils.csv`, run the importer from the repository root. It verifies the published CSVs and records their source URLs and SHA-256 hashes. Use `--directory` to read local copies of the published CSVs.

```bash
python scripts/import_reference_airfoils.py
```

The pooled notebook overlays the reference profiles on the geometry envelope (Figure 4) and plots seed-42 predictions against the reference coefficients (Figures 5–7). [Saved figures](../images/NACA4D_10%20U%20NACA5D_10/)

## Results Export

The original notebooks write `results/experiments/<notebook-name>.json.gz` from their **Results Export** cell. The geometry notebook writes one export per configuration as it finishes. Each file records per-seed validation and test metrics, training histories, settings, input CSV paths and hashes, and row assignments.

Single-family exports use zero-based CSV row indices. The original pooled export indexes the two concatenated CSVs in manifest order. Directional exports index the source CSV for training and validation and the target CSV for testing. Geometry exports identify each row by dataset name and original CSV row number.

To regenerate the root README tables, use the 20-run, 50-epoch exports from the three 4-digit notebooks, both directional transfer notebooks, and the pooled notebook.

```bash
python scripts/update_readme_results.py
```
