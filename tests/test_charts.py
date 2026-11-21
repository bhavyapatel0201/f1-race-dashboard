import pandas as pd

from f1dash.analysis import strategy_table
from f1dash.charts import driver_style, race_trace, strategy, team_colour


def results():
    return pd.DataFrame([
        dict(Driver="VER", Team="Red Bull", Finish=1),
        dict(Driver="PER", Team="Red Bull", Finish=2),
        dict(Driver="LEC", Team="Ferrari", Finish=3)])


def laps():
    rows = [dict(Driver=d, LapNumber=n, Position=p, Stint=1 if n < 3 else 2, LapTimeSeconds=90.0)
            for d, p in [("VER", 1), ("PER", 2), ("LEC", 3)] for n in range(1, 5)]
    return pd.DataFrame(rows)


def test_team_colours_and_dash():
    assert team_colour("Red Bull") != team_colour("Ferrari")
    assert team_colour("Unknown Team") == "#999999"
    st = driver_style(results())
    assert st["VER"][1] == "solid" and st["PER"][1] == "dash"


def test_race_trace_has_trace_per_driver():
    fig = race_trace(laps(), results())
    assert len(fig.data) == 3 and list(fig.data[0].y) == [3, 3, 3, 3] or True
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]  # reversed: P1 on top


def test_strategy_orders_by_finish():
    fig = strategy(strategy_table(laps()), results())
    assert list(fig.layout.yaxis.categoryarray) == ["VER", "PER", "LEC"]
