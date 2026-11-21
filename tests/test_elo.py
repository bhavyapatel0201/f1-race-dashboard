import numpy as np
import pandas as pd

from f1dash.features import (ELO_START, FEATURE_SETS, build_features)
from f1dash.model import evaluate, walk_forward


def results(n_races=24, winner_always="A"):
    rows = []
    for i in range(n_races):
        year, rnd = 2020 + i // 6, i % 6 + 1
        order = ["A", "B", "C", "D"] if winner_always == "A" else ["D", "C", "B", "A"]
        for pos, drv in enumerate(order, start=1):
            rows.append(dict(Year=year, Round=rnd, Circuit="x", Driver=drv,
                             Team="t" + drv, Grid=pos, Finish=pos, Status="Finished"))
    return pd.DataFrame(rows)


def test_first_race_ratings_are_neutral_and_winner_rises():
    f = build_features(results())
    first = f[f["RaceIndex"] == 0]
    assert (first["DriverElo"] == ELO_START).all() and (first["TeamElo"] == ELO_START).all()
    last = f[f["RaceIndex"] == f["RaceIndex"].max()].set_index("Driver")
    assert last.loc["A", "DriverElo"] > last.loc["B", "DriverElo"] > last.loc["D", "DriverElo"]
    assert last.loc["A", "DriverEloRel"] > 0 > last.loc["D", "DriverEloRel"]


def test_elo_and_points_never_see_current_race():
    base = build_features(results())
    altered = results()
    mask = (altered["Year"] == 2023) & (altered["Round"] == 6)
    altered.loc[mask & (altered["Driver"] == "A"), "Finish"] = 4
    altered.loc[mask & (altered["Driver"] == "D"), "Finish"] = 1
    new = build_features(altered)
    last = base["RaceIndex"].max()
    cols = ["DriverElo", "TeamElo", "PointsBefore", "PointsShare", "DriverFormRel", "TeamFormRel"]
    a = base[base["RaceIndex"] == last].sort_values("Driver")[cols].reset_index(drop=True)
    b = new[new["RaceIndex"] == last].sort_values("Driver")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_points_reset_each_season():
    f = build_features(results())
    first_of_2021 = f[(f["Year"] == 2021) & (f["Round"] == 1) & (f["Driver"] == "A")].iloc[0]
    assert first_of_2021["PointsBefore"] == 0
    third = f[(f["Year"] == 2020) & (f["Round"] == 3) & (f["Driver"] == "A")].iloc[0]
    assert third["PointsBefore"] == 50


def test_walk_forward_never_trains_on_evaluation_year():
    f = build_features(results(n_races=36))
    seen = {}

    import f1dash.model as m
    orig = m.fit_predict

    def spy(train, test, features, kind):
        seen[int(test["Year"].iloc[0])] = int(train["Year"].max())
        return orig(train, test, features, kind)

    m.fit_predict = spy
    try:
        scored = walk_forward(f, [2021, 2022, 2023, 2024], FEATURE_SETS["grid"], "logistic")
    finally:
        m.fit_predict = orig
    assert all(train_max < year for year, train_max in seen.items())
    assert 2021 not in seen  # only 1 prior season: below min_train_seasons=2
    assert set(scored["Year"]) == {2022, 2023, 2024}
    assert evaluate(scored)["top1"] == 1.0  # A always wins and always starts P1
