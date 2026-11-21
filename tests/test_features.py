import pandas as pd

from f1dash.dataset import parse_season_results
from f1dash.features import add_features, pole_baseline, time_split


def make_results():
    rows = []
    for year, rnd in [(2022, 1), (2022, 2), (2023, 1), (2023, 2)]:
        # A wins everything, B second; grid: B on pole in 2023 round 2 only
        for drv, team, fin in [("A", "t1", 1), ("B", "t2", 2), ("C", "t3", 3)]:
            grid = fin
            if (year, rnd) == (2023, 2):
                grid = {"A": 2, "B": 1, "C": 3}[drv]
            rows.append(dict(Year=year, Round=rnd, Circuit="x", Driver=drv, Team=team,
                             Grid=grid, Finish=fin, Status="Finished"))
    return pd.DataFrame(rows)


def test_features_use_only_earlier_races():
    f = add_features(make_results())
    first = f[f["RaceIndex"] == 0]
    assert first["DriverForm"].isna().all()           # nothing earlier to learn from
    a_second = f[(f["Driver"] == "A") & (f["RaceIndex"] == 1)].iloc[0]
    assert a_second["DriverForm"] == 1.0              # only race 0 (a win) is visible
    a_last = f[(f["Driver"] == "A") & (f["RaceIndex"] == 3)].iloc[0]
    assert a_last["DriverRecentWins"] == 3.0          # wins from races 0-2, not race 3
    b_circuit = f[(f["Driver"] == "B") & (f["RaceIndex"] == 3)].iloc[0]
    assert b_circuit["CircuitAvgFinish"] == 2.0


def test_current_race_result_cannot_change_its_own_features():
    base = add_features(make_results())
    altered = make_results()
    altered.loc[(altered["Year"] == 2023) & (altered["Round"] == 2) & (altered["Driver"] == "A"),
                "Finish"] = 3
    new = add_features(altered)
    cols = ["DriverForm", "DriverRecentWins", "TeamForm", "CircuitAvgFinish", "GridPos"]
    k = lambda d: d[(d["Driver"] == "A") & (d["RaceIndex"] == 3)][cols].iloc[0].tolist()
    assert k(base) == k(new)


def test_pole_baseline_and_split():
    f = add_features(make_results())
    t = pole_baseline(f).set_index("Year")
    assert t.loc[2022, "PoleWinRate"] == 1.0
    assert t.loc[2023, "PoleWinRate"] == 0.5          # B on pole lost in round 2
    train, valid, test = time_split(f, 2022, 2022)
    assert set(train["Year"]) == {2022} and valid.empty and set(test["Year"]) == {2023}


def test_parse_season_results():
    page = {"RaceTable": {"Races": [{"season": "2024", "round": "1",
            "Circuit": {"circuitId": "bahrain"},
            "Results": [{"position": "1", "grid": "1", "status": "Finished",
                         "Driver": {"driverId": "max_verstappen"},
                         "Constructor": {"constructorId": "red_bull"}}]}]}}
    df = parse_season_results([page])
    assert df.loc[0, "Circuit"] == "bahrain" and df.loc[0, "Grid"] == 1
