import numpy as np
import pandas as pd

from f1dash.features import FEATURE_COLUMNS, add_features
from f1dash.model import evaluate, fit_predict, normalise_per_race, pole_prediction


def synthetic(n_races=60, seed=0):
    """20 drivers; the better the grid slot, the likelier the win."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        year, rnd = 2018 + i // 12, i % 12 + 1
        grid = rng.permutation(np.arange(1, 21))
        score = -grid + rng.normal(0, 3, 20)
        finish = (-score).argsort().argsort() + 1
        for d in range(20):
            rows.append(dict(Year=year, Round=rnd, Circuit=f"c{rnd}", Driver=f"d{d}",
                             Team=f"t{d // 2}", Grid=int(grid[d]), Finish=int(finish[d]),
                             Status="Finished"))
    return add_features(pd.DataFrame(rows))


def test_probabilities_sum_to_one_per_race():
    df = synthetic()
    tr, te = df[df["Year"] < 2022], df[df["Year"] >= 2022]
    scored = fit_predict(tr, te, FEATURE_COLUMNS, "logistic")
    sums = scored.groupby("RaceIndex")["WinProb"].sum()
    assert np.allclose(sums, 1.0)


def test_model_beats_random_and_roughly_matches_pole_on_grid_driven_data():
    df = synthetic(n_races=120)
    tr, te = df[df["Year"] < 2023], df[df["Year"] >= 2023]
    model = evaluate(fit_predict(tr, te, ["GridPos"], "logistic"))
    pole = evaluate(pole_prediction(te))
    assert model["top1"] > 0.15                      # random would be 0.05
    assert model["winner_logloss"] < np.log(20)      # better than uniform guessing
    assert abs(model["top1"] - pole["top1"]) <= 0.2


def test_pole_prediction_puts_all_mass_on_pole():
    te = synthetic()
    p = pole_prediction(te)
    top = p.loc[p.groupby("RaceIndex")["WinProb"].idxmax()]
    assert (top["GridPos"] == 1).all()


def test_lightgbm_runs():
    df = synthetic()
    tr, te = df[df["Year"] < 2022], df[df["Year"] >= 2022]
    scored = fit_predict(tr, te, FEATURE_COLUMNS, "lightgbm")
    assert scored["WinProb"].between(0, 1).all()
