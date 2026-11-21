#!/usr/bin/env python3

import argparse
import hashlib
import sys
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
reference_airfoils = __import__("reference_airfoils")
geometry_splits = __import__("geometry_splits")

REVISION = "b2d0691881e7d5a5db121aec92004c69d48ab5c0"
SOURCE = (
    f"https://raw.githubusercontent.com/ZeeshanIqbalKhan/AirfoilANN/{REVISION}/Data"
)
CHECKSUMS = {
    "0045": "e1fca76925fef525147cb38ef5b3b30609d6ea94ec4ef8f4757a03c3f3b61c3b",
    "2412": "0271fa7a32ce109340283c8acaff42f89628c6cea12031a9f6dae245a848e3b7",
    "6408": "a2f3aa3dd45a5c041772736523634cac72e559ffed5856f56b25d7854949c4c1",
    "136138": "1bb6f63a1ec30dd555150ac9ef3d550bc4e1bb18df2d81c2f9c1353cdc18b41c",
}


# Read the published reference polar and retain the paper's evaluation conditions
def read_airfoil(name, directory, coordinate_columns):
    filename = f"NACA{name}.csv"
    url = f"{SOURCE}/{filename}"
    if directory is None:
        with urlopen(url, timeout=30) as response:
            contents = response.read()
    else:
        contents = (directory / filename).read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != CHECKSUMS[name]:
        raise ValueError(f"Unexpected checksum for {filename}")

    data = pd.read_csv(BytesIO(contents)).rename(
        columns={
            **{f"{column[:2]}_{column[2:]}": column for column in coordinate_columns},
            "ReynoldsNumber": "Re",
            "MachNumber": "M",
            "Cl": "CL",
            "Cd": "CD",
        }
    )
    rows = data.loc[
        data["M"].eq(reference_airfoils.MACH)
        & data["Re"].eq(reference_airfoils.REYNOLDS)
    ]
    return rows.assign(airfoil=name, source=f"{url} sha256={digest}")


# Validate the complete sweep before replacing the reference CSV
def main():
    parser = argparse.ArgumentParser(
        description="Import the paper's published reference polars at Mach 0.25 and Re 250,000."
    )
    parser.add_argument(
        "--directory", type=Path, help="Read the published CSVs from this directory"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/reference_airfoils.csv"
    )
    args = parser.parse_args()

    coordinates = geometry_splits.geometry_columns(10)
    features = coordinates + geometry_splits.CONDITIONS
    reference = pd.concat(
        [
            read_airfoil(name, args.directory, coordinates)
            for name in reference_airfoils.AIRFOILS
        ],
        ignore_index=True,
    )
    reference = reference[["airfoil", "source", *features, *reference_airfoils.TARGETS]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as temporary:
        path = Path(temporary.name)
    try:
        reference.to_csv(path, index=False)
        reference_airfoils.load_reference(path, features)
        path.replace(args.output)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
