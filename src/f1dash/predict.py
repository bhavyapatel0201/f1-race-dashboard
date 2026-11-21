"""Win probabilities for one historical race, from models trained only on earlier seasons."""
import pandas as pd

from f1dash.features import FEATURE_SETS
from f1dash.model import fit_predict

MIN_TRAIN_SEASONS = 2


def race_probabilities(feats: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """Return one row per driver with the grid-only and relative-strength win probabilities.

    Both models are trained on seasons strictly before `year` (the same rule as the
    walk-forward evaluation), so the race being shown was never seen in training.
    """
    train = feats[feats["Year"] < year]
    test = feats[(feats["Year"] == year) & (feats["Round"] == rnd)]
    if test.empty:
        raise ValueError(f"No data for {year} round {rnd}")
    if train["Year"].nunique() < MIN_TRAIN_SEASONS:
        raise ValueError(f"Need at least {MIN_TRAIN_SEASONS} earlier seasons to train on")

    grid = fit_predict(train, test, FEATURE_SETS["grid"], "logistic")
    rel = fit_predict(train, test, FEATURE_SETS["relative"], "logistic")
    out = test[["Driver", "Grid", "Finish"]].rename(columns={"Driver": "DriverId"}).copy()
    out["GridModel"] = grid["WinProb"].to_numpy()
    out["RelativeModel"] = rel["WinProb"].to_numpy()
    return out.sort_values("GridModel", ascending=False).reset_index(drop=True)
