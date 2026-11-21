"""Match operating conditions and keep equivalent geometries in the same split."""

import pandas as pd
from sklearn.model_selection import train_test_split

PARAMETERS = {
    "NACA4D": ["t", "m", "p"],
    "NACA5D": ["t", "CL_design", "p", "s"],
}
CONDITIONS = ["alpha", "M", "Re"]
RESOLUTIONS = (5, 10, 15)


# Use the same coordinate order as the generated datasets
def geometry_columns(resolution):
    return [
        f"y{surface}{i}" for surface in ("U", "L") for i in range(1, resolution + 1)
    ]


# Match rows by airfoil parameters and operating conditions
def match_cases(frames, family):
    keys = PARAMETERS[family] + CONDITIONS
    indexed = {
        n: frame.assign(csv_row=frame.index).set_index(keys, verify_integrity=True)
        for n, frame in frames.items()
    }
    common = (
        indexed[5].index.intersection(indexed[10].index).intersection(indexed[15].index)
    )
    common = common.sort_values()
    matched = {n: frame.loc[common].reset_index() for n, frame in indexed.items()}
    for n in (10, 15):
        if not matched[5][["CL", "CD", "Cm"]].equals(matched[n][["CL", "CD", "Cm"]]):
            raise ValueError(f"Target mismatch between 05 and {n:02} for {family}")
    return matched


# Join parameter IDs whenever any resolution has an identical geometry
def geometry_groups(frames):
    parents = {}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for n in RESOLUTIONS:
        profiles = {}
        for family, datasets in frames.items():
            params = PARAMETERS[family]
            columns = geometry_columns(n)
            unique = datasets[n][params + columns].drop_duplicates()
            for row in unique.itertuples(index=False, name=None):
                key = (family, *row[: len(params)])
                parents.setdefault(key, key)
                profile = row[len(params) :]
                first = profiles.setdefault(profile, key)
                parents[root(key)] = root(first)

    groups = {}
    group_ids = {}
    for family, datasets in frames.items():
        keys = datasets[10][PARAMETERS[family]]
        groups[family] = [
            group_ids.setdefault(root((family, *row)), len(group_ids))
            for row in keys.itertuples(index=False, name=None)
        ]
    return groups


# Split group labels once for each benchmark population
def group_split(groups):
    labels = pd.Series(groups).unique()
    train, remainder = train_test_split(labels, train_size=0.7, random_state=42)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=42)
    return {"train": train, "validation": validation, "test": test}


# Exclude target rows with an exact source coordinate match
def exclusive_target(source, target, columns):
    source_profiles = pd.MultiIndex.from_frame(source[columns]).unique()
    target_profiles = pd.MultiIndex.from_frame(target[columns])
    return ~target_profiles.isin(source_profiles)
