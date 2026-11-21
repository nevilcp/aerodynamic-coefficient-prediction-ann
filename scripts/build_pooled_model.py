"""Train and package seed 42 on the recorded pooled geometry split."""

import argparse
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from predict_pooled_model import load_package, predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
experiment_results = __import__("experiment_results")
geometry_splits = __import__("geometry_splits")
FEATURES = geometry_splits.geometry_columns(10) + geometry_splits.CONDITIONS
TARGETS = experiment_results.TARGETS
LAYERS = (512, 256, 128)
ARCHITECTURE = "M_512_256_128"
SEED = 42


def load_splits(benchmark):
    datasets = {row["name"]: row["sha256"] for row in benchmark["datasets"]}
    manifest = experiment_results.dataset_manifest(ROOT, datasets)
    if {row["name"]: row["sha256"] for row in manifest} != datasets:
        raise ValueError("Dataset contents differ from the recorded benchmark")

    frames = {
        item["name"]: pd.read_csv(ROOT / item["path"], usecols=FEATURES + TARGETS)
        for item in manifest
    }
    data = pd.concat(frames, names=["dataset", "csv_row"], verify_integrity=True)
    splits = {
        name: data.loc[pd.MultiIndex.from_tuples(ids)]
        for name, ids in benchmark["splits"].items()
    }
    return splits["train"], splits["validation"], splits["test"]


def train_model(train, validation, scaler, settings):
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential(
        [keras.Input(shape=(len(FEATURES),))]
        + [keras.layers.Dense(size, activation="relu") for size in LAYERS]
        + [keras.layers.Dense(len(TARGETS), activation="linear")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=settings["learning_rate"],
            beta_1=settings["beta_1"],
            beta_2=settings["beta_2"],
        ),
        loss=settings["loss"],
    )
    model.fit(
        scaler.transform(train[FEATURES]),
        train[TARGETS],
        validation_data=(scaler.transform(validation[FEATURES]), validation[TARGETS]),
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=settings["lr_factor"],
                patience=settings["lr_patience"],
                min_lr=settings["min_lr"],
            )
        ],
        verbose=2,
    )
    return model


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_model_card(directory, benchmark, metrics):
    reference_scores = benchmark["architectures"][ARCHITECTURE]
    mean_test = pd.DataFrame([run["test"] for run in reference_scores]).mean()
    rows = [
        f"| {target} | {metrics[target]['rmse']:.6f} | {metrics[target]['r2']:.6f} "
        f"| {mean_test[f'{target}_rmse']:.6f} | {mean_test[f'{target}_r2']:.6f} |"
        for target in TARGETS
    ]
    (directory / "README.md").write_text(
        """# Pooled 10-point NACA airfoil model

This Keras multilayer perceptron predicts lift (CL), drag (CD), and pitching
moment (Cm) for NACA 4- and 5-digit airfoils. The model has hidden layers of
512, 256, and 128 units and was trained for 50 epochs with seed 42.

## Run the example

From the unpacked directory, install the tested dependencies and run:

```bash
python -m pip install -r requirements.txt
python predict.py example.json
```

Provide a JSON object with normalized surface ordinates yU1 through yU10 and
yL1 through yL10 in leading-edge to trailing-edge order at the ten interior
cosine-spaced chordwise positions, alpha in degrees, Mach number M, and Reynolds
number Re. The prediction script applies the saved training-set standardization
and returns unscaled coefficients.

## Evaluation

The packaged checkpoint was evaluated after reloading on 61,950 rows from
held-out geometries. The original 20-run benchmark used the same split. The
dataset manifest and split assignments are in `benchmark.json.gz`.

| Coefficient | Released RMSE | Released R² | Benchmark mean RMSE | Benchmark mean R² |
|---|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + """

## Scope

Training and evaluation use simulated JavaFoil polars for the sampled NACA
families. The training grid spans −10° to 10° angle of attack, Mach 0.1 to
0.3, and Reynolds number 100,000 to 500,000. The checkpoint is intended for
research and reproduction of this simulation benchmark.

[Training dataset and attribution](https://huggingface.co/datasets/nevilcp/aerodynamic-coefficient-prediction-ann)
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the model package")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "results" / "experiments" / "geometry-pooled-10.json.gz",
    )
    args = parser.parse_args()

    with gzip.open(args.benchmark, "rt", encoding="utf-8") as stream:
        benchmark = json.load(stream)
    settings = benchmark["settings"]
    if (
        benchmark["experiment"] != "geometry-pooled-10"
        or ARCHITECTURE not in benchmark["architectures"]
    ):
        raise ValueError("Expected the pooled 10-point geometry benchmark")

    args.output.mkdir(parents=True)
    train, validation, test = load_splits(benchmark)
    scaler = StandardScaler().fit(train[FEATURES])
    model = train_model(train, validation, scaler, settings)
    model.save(args.output / "model.keras")
    write_json(
        args.output / "preprocessing.json",
        {
            "features": FEATURES,
            "outputs": TARGETS,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    )
    sample = test[FEATURES].iloc[[0]]
    write_json(args.output / "example.json", sample.iloc[0].to_dict())
    shutil.copyfile(
        Path(__file__).with_name("predict_pooled_model.py"), args.output / "predict.py"
    )
    shutil.copyfile(args.benchmark, args.output / "benchmark.json.gz")
    (args.output / "requirements.txt").write_text(
        f"tensorflow=={tf.__version__}\nkeras=={keras.__version__}\nnumpy=={np.__version__}\n"
    )

    reloaded, metadata = load_package(args.output)
    expected = model(scaler.transform(sample), training=False).numpy()
    np.testing.assert_allclose(
        predict(reloaded, metadata, sample.to_dict("records")), expected
    )
    predictions = reloaded.predict(
        scaler.transform(test[FEATURES]), batch_size=128, verbose=0
    )
    rmse = np.sqrt(
        mean_squared_error(test[TARGETS], predictions, multioutput="raw_values")
    )
    r2 = r2_score(test[TARGETS], predictions, multioutput="raw_values")
    metrics = {
        name: {"rmse": error, "r2": score}
        for name, error, score in zip(TARGETS, rmse, r2, strict=True)
    }
    write_json(
        args.output / "evaluation.json",
        {
            "seed": SEED,
            "test_rows": len(test),
            "test": metrics,
            "training_versions": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
            },
        },
    )
    write_model_card(args.output, benchmark, metrics)


if __name__ == "__main__":
    main()
