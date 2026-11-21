#!/usr/bin/env python3

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
TARGETS = __import__("experiment_results").TARGETS

METRICS = [(metric, target) for metric in ("rmse", "r2") for target in TARGETS]
FIXED = "M_512_256_128"
DEPTH = ("M_64", "M_64_32", "M_64_32_16", "M_64_32_16_8")
WIDTH = ("M_64_32_16", "M_128_64_32", "M_256_128_64", FIXED, "M_1024_512_256")


# Read only complete experiment exports for the README tables
def load_results(name):
    path = ROOT / "results" / "experiments" / f"{name}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        result = json.load(source)
    settings = result["settings"]
    if (settings["n_runs"], settings["epochs"]) != (20, 50):
        raise ValueError(f"{name} needs a 20-run, 50-epoch results export")
    if any(len(runs) != 20 for runs in result["architectures"].values()):
        raise ValueError(f"{name} has incomplete per-seed results")
    return result


# Calculate the mean per-run score for each aerodynamic coefficient
def metric_values(result, architecture, metrics=METRICS):
    runs = result["architectures"][architecture]
    means = pd.DataFrame([run["test"] for run in runs]).mean()
    return [f"{means[f'{target}_{metric}']:.6f}" for metric, target in metrics]


# Replace a section's table while preserving its surrounding discussion
def replace_table(readme, heading, rows):
    start = readme.index("\n|", readme.index(f"### {heading}\n")) + 1
    end = readme.index("\n\n", start)
    headers = readme[start:end].splitlines()[:2]
    contents = ["| " + " | ".join(row) + " |" for row in rows]
    return readme[:start] + "\n".join(headers + contents) + readme[end:]


def main():
    path = ROOT / "README.md"
    readme = path.read_text()
    primary = load_results("NACA4D_10")
    for section, architectures, baseline in (
        ("Effect of Network Depth", DEPTH, "M_64_32_16"),
        ("Effect of Network Width", WIDTH, FIXED),
    ):
        rows = []
        for i, name in enumerate(architectures, 1):
            layers = name.removeprefix("M_").replace("_", ", ")
            row = [str(i), f"[{layers}, 3]", *metric_values(primary, name)]
            rows.append([f"**{value}**" for value in row] if name == baseline else row)
        readme = replace_table(readme, section, rows)

    rows = []
    for n in (5, 10, 15):
        result = load_results(f"NACA4D_{n:02}")
        row = [f"NACA4D\\_{n:02}", *metric_values(result, FIXED)]
        rows.append([f"**{value}**" for value in row] if n == 10 else row)
    readme = replace_table(readme, "Effect of Geometric Resolution", rows)

    rows = []
    for name, train, test in (
        (
            "NACA4D_10 - NACA5D_10",
            "NACA4D\\_10 source split",
            "Full NACA5D\\_10 target family",
        ),
        (
            "NACA5D_10 - NACA4D_10",
            "NACA5D\\_10 source split",
            "Full NACA4D\\_10 target family",
        ),
        (
            "NACA4D_10 U NACA5D_10",
            "NACA4D\\_10 ∪ NACA5D\\_10 training rows",
            "Held-out rows from both families",
        ),
    ):
        result = load_results(name)
        rows.append([train, test, *metric_values(result, FIXED, METRICS[3:])])
    path.write_text(replace_table(readme, "Cross-Family Generalization", rows))


if __name__ == "__main__":
    main()
