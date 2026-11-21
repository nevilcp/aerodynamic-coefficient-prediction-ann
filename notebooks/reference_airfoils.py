"""Validate the independent reference-airfoil profiles and polar sweep."""

import numpy as np
import pandas as pd

AIRFOILS = ("0045", "2412", "6408", "136138")
MACH = 0.25
REYNOLDS = 250000
TARGETS = ["CL", "CD", "Cm"]


# Check the shared angle sweep and the geometry supplied with each polar
def load_reference(path, feature_columns):
    reference = pd.read_csv(path, dtype={"airfoil": str, "source": str})
    numeric = list(feature_columns) + TARGETS
    missing = {"airfoil", "source", *numeric} - set(reference)
    if missing:
        raise ValueError(f"Missing reference columns: {sorted(missing)}")
    if set(reference["airfoil"]) != set(AIRFOILS):
        raise ValueError(f"Expected precisely these airfoils: {AIRFOILS}")
    if reference["source"].fillna("").str.strip().eq("").any():
        raise ValueError("Each reference row must identify its source")
    reference[numeric] = reference[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(reference[numeric].to_numpy()).all():
        raise ValueError("Reference data must contain finite numeric values")
    if not (
        np.isclose(reference["M"], MACH).all()
        and np.isclose(reference["Re"], REYNOLDS).all()
    ):
        raise ValueError(f"Reference conditions must be M={MACH} and Re={REYNOLDS}")

    coordinates = [
        column for column in feature_columns if column.startswith(("yU", "yL"))
    ]
    expected = None
    for airfoil, rows in reference.groupby("airfoil"):
        angles = np.sort(rows["alpha"].to_numpy())
        intervals = np.diff(angles)
        if (
            len(angles) < 3
            or (intervals <= 0).any()
            or not np.allclose(intervals, intervals[0])
        ):
            raise ValueError(f"NACA {airfoil} needs a uniform sweep with unique angles")
        if expected is not None and not np.array_equal(angles, expected):
            raise ValueError("All reference airfoils must use the same angle sweep")
        if len(rows[coordinates].drop_duplicates()) != 1:
            raise ValueError(
                f"NACA {airfoil} needs one geometry across its angle sweep"
            )
        expected = angles
    return reference.sort_values(["airfoil", "alpha"]).reset_index(drop=True)
