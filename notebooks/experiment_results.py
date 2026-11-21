"""Export run metrics and the dataset rows used for each split."""

import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

TARGETS = ["CL", "CD", "Cm"]


# Keep the dataset lookup order used by the experiment notebooks
def dataset_path(root, name):
    candidates = [
        root / "results" / name / f"{name}.csv",
        root / "results" / "data" / f"{name}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Dataset {name} missing; expected one of {candidates}")


# Record the exact input and original row order for each dataset
def dataset_manifest(root, names):
    manifest = []
    for name in names:
        path = dataset_path(root, name)
        with path.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append(
            {
                "name": name,
                "path": str(path.relative_to(root)),
                "sha256": digest.hexdigest(),
            }
        )
    return manifest


# Save settings, row identifiers, and individual seeded scores together
def export_results(
    root, name, datasets, splits, metrics, validation_metrics, histories, settings
):
    destination = root / "results" / "experiments" / f"{name}.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": name,
        "datasets": datasets,
        "settings": settings,
        "splits": {label: list(rows) for label, rows in splits.items()},
        "architectures": {
            architecture: [
                {
                    "seed": 42 + i,
                    "test": test,
                    "validation": validation,
                    "history": history,
                }
                for i, (test, validation, history) in enumerate(
                    zip(
                        runs.to_dict("records"),
                        validation_metrics[architecture].to_dict("records"),
                        histories[architecture],
                        strict=True,
                    )
                )
            ]
            for architecture, runs in metrics.items()
        },
    }
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with gzip.open(path, "wt", encoding="utf-8") as output:
            json.dump(result, output, allow_nan=False, separators=(",", ":"))
        path.replace(destination)
    finally:
        path.unlink(missing_ok=True)


# Record the training settings shared by the experiments
def training_settings(n_runs, epochs, y_train, protocol):
    return {
        "protocol": protocol,
        "split_seed": 42,
        "n_runs": n_runs,
        "epochs": epochs,
        "batch_size": 128,
        "optimizer": "Adam",
        "learning_rate": 0.0005,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "loss": "mse",
        "lr_factor": 0.9,
        "lr_patience": 5,
        "min_lr": 0.00001,
        "hidden_activation": "relu",
        "output_activation": "linear",
        "training_target_variance": y_train[TARGETS].var(ddof=0).to_dict(),
    }
