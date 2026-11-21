import numpy as np
import pandas as pd
import pytest

from f1dash.features import build_features
from f1dash.predict import race_probabilities


def feats():
    rows = []
    for i in range(30):
        year, rnd = 2018 + i // 6, i % 6 + 1
        for pos, drv in enumerate(["A", "B", "C", "D"], start=1):
            rows.append(dict(Year=year, Round=rnd, Circuit="x", Driver=drv, Team="t" + drv,
                             Grid=pos, Finish=pos, Status="Finished"))
    return build_features(pd.DataFrame(rows))


def test_probabilities_sum_to_one_and_cover_all_drivers():
    out = race_probabilities(feats(), 2021, 3)
    assert len(out) == 4
    assert np.isclose(out["GridModel"].sum(), 1.0) and np.isclose(out["RelativeModel"].sum(), 1.0)
    assert out.iloc[0]["DriverId"] == "A"            # A always wins from pole


def test_needs_two_earlier_seasons():
    with pytest.raises(ValueError):
        race_probabilities(feats(), 2019, 1)


def test_unknown_race_raises():
    with pytest.raises(ValueError):
        race_probabilities(feats(), 2021, 99)
