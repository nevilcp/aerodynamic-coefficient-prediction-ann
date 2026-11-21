"""Predict the coefficients for a JSON object of named airfoil features."""

import argparse
import json
from pathlib import Path

import keras
import numpy as np


def load_package(directory):
    model = keras.models.load_model(directory / "model.keras", compile=False)
    metadata = json.loads((directory / "preprocessing.json").read_text())
    return model, metadata


def predict(model, metadata, rows):
    features = metadata["features"]
    if any(set(row) != set(features) for row in rows):
        raise ValueError(
            f"Each input row must contain exactly these features: {features}"
        )
    inputs = np.array([[row[name] for name in features] for row in rows], dtype=float)
    if not np.isfinite(inputs).all():
        raise ValueError("Features must be finite numbers")
    scaled = (inputs - metadata["mean"]) / metadata["scale"]
    return model(scaled, training=False).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Path to a JSON object of named inputs"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    row = json.loads(args.input.read_text())
    model, metadata = load_package(args.model_dir)
    outputs = predict(model, metadata, [row])[0]
    print(json.dumps(dict(zip(metadata["outputs"], map(float, outputs), strict=True))))


if __name__ == "__main__":
    main()
