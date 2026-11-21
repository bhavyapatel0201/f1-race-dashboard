import numpy as np
import pandas as pd

from f1dash.analysis import (add_stints, clean_laps, lap_gap, stint_summary,
                             strategy_table, track_evolution)


def make_data():
    rows = []
    for drv, base in [("VER", 80.0), ("LEC", 80.5)]:
        for lap in range(1, 41):
            stint_lap = lap if lap <= 20 else lap - 20
            rows.append(dict(Driver=drv, LapNumber=lap, Position=1,
                             LapTimeSeconds=base + 0.05 * stint_lap))
    laps = pd.DataFrame(rows)
    pits = pd.DataFrame([dict(Driver=d, Stop=1, Lap=20, PitSeconds=22.0) for d in ("VER", "LEC")])
    return laps, pits


def test_add_stints_marks_in_out_laps():
    laps, pits = make_data()
    df = add_stints(laps, pits)
    ver = df[df["Driver"] == "VER"].set_index("LapNumber")
    assert ver.loc[20, "IsInLap"] and ver.loc[21, "IsOutLap"]
    assert ver.loc[19, "Stint"] == 1 and ver.loc[21, "Stint"] == 2
    assert ver.loc[21, "TyreLife"] == 1 and ver.loc[30, "TyreLife"] == 10


def test_clean_drops_lap1_pit_laps_and_neutralised_laps():
    laps, pits = make_data()
    laps.loc[laps["LapNumber"].between(30, 32), "LapTimeSeconds"] += 30  # safety car
    df = clean_laps(add_stints(laps, pits))
    assert 1 not in set(df["LapNumber"])
    assert not {20, 21} & set(df["LapNumber"])
    assert not {30, 31, 32} & set(df["LapNumber"])


def test_lap_gap_sign_and_total():
    laps, _ = make_data()
    gap = lap_gap(laps, "VER", "LEC")
    assert (gap["Gap"] < 0).all()
    assert abs(gap["CumulativeGap"].iloc[-1] - (-0.5 * 40)) < 1e-9


def test_stint_summary_recovers_slope():
    laps, pits = make_data()
    s = stint_summary(clean_laps(add_stints(laps, pits)))
    assert len(s) == 4  # 2 drivers x 2 stints
    assert np.allclose(s["DegPerLap"], 0.05, atol=1e-6)


def test_strategy_and_evolution():
    laps, pits = make_data()
    df = add_stints(laps, pits)
    assert len(strategy_table(df)) == 4
    assert len(track_evolution(clean_laps(df))) > 0
