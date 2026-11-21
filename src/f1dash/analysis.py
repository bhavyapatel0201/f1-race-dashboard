"""Pure pandas functions: easy to test, no network needed.

Source data has no tyre compound or flag status, so stints come from pit stops
and non-representative laps are removed with simple statistical rules.
"""
import numpy as np
import pandas as pd

SC_FACTOR = 1.10       # field-median lap this much slower than normal => neutralised lap
MIN_STINT_LAPS = 8     # shorter stints give noisy slope fits
OUTLIER_FACTOR = 1.07  # a driver's lap this much slower than their median => dropped


def add_stints(laps: pd.DataFrame, pits: pd.DataFrame) -> pd.DataFrame:
    """Add Stint, TyreLife (laps since last stop), IsInLap, IsOutLap."""
    df = laps.copy()
    stint, life, in_lap, out_lap = [], [], [], []
    stops = {d: sorted(g["Lap"]) for d, g in pits.groupby("Driver")} if len(pits) else {}
    for drv, lap in zip(df["Driver"], df["LapNumber"]):
        s = stops.get(drv, [])
        prior = [x for x in s if x < lap]
        prev = prior[-1] if prior else 0
        stint.append(len(prior) + 1)
        life.append(lap - prev)
        in_lap.append(lap in s)
        out_lap.append(bool(prior) and lap == prev + 1)
    df["Stint"], df["TyreLife"] = stint, life
    df["IsInLap"], df["IsOutLap"] = in_lap, out_lap
    return df


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep representative racing laps.

    Drops: lap 1 (standing start), in/out laps, laps missing a time, neutralised
    laps (field median far above the race norm) and per-driver slow outliers.
    """
    df = laps.dropna(subset=["LapTimeSeconds"]).copy()
    df = df[df["LapNumber"] > 1]
    for col in ("IsInLap", "IsOutLap"):
        if col in df.columns:
            df = df[~df[col]]
    field = df.groupby("LapNumber")["LapTimeSeconds"].median()
    slow_laps = field[field > SC_FACTOR * field.median()].index
    df = df[~df["LapNumber"].isin(slow_laps)]
    drv_median = df.groupby("Driver")["LapTimeSeconds"].transform("median")
    return df[df["LapTimeSeconds"] <= OUTLIER_FACTOR * drv_median]


def lap_gap(laps: pd.DataFrame, driver_a: str, driver_b: str) -> pd.DataFrame:
    """Per-lap time difference (A minus B). Negative means A was faster."""
    a = laps[laps["Driver"] == driver_a].set_index("LapNumber")["LapTimeSeconds"]
    b = laps[laps["Driver"] == driver_b].set_index("LapNumber")["LapTimeSeconds"]
    out = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    out["Gap"] = out["A"] - out["B"]
    out["CumulativeGap"] = out["Gap"].cumsum()
    return out.reset_index()


def stint_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Per driver and stint: laps used, median pace, and degradation (s per lap of
    tyre age, linear fit). Degradation also contains fuel burn-off, so it understates
    true tyre wear."""
    rows = []
    for (driver, stint), g in laps.groupby(["Driver", "Stint"]):
        if len(g) < MIN_STINT_LAPS or g["TyreLife"].nunique() < 3:
            continue
        slope, _ = np.polyfit(g["TyreLife"], g["LapTimeSeconds"], 1)
        rows.append({
            "Driver": driver, "Stint": int(stint), "Laps": len(g),
            "MedianPace": round(float(g["LapTimeSeconds"].median()), 3),
            "DegPerLap": round(float(slope), 4),
        })
    return pd.DataFrame(rows)


def track_evolution(laps: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Field-median lap time per lap number, smoothed (rubbering in + fuel burn-off)."""
    out = laps.groupby("LapNumber")["LapTimeSeconds"].median().rename("Median").to_frame()
    out["Smoothed"] = out["Median"].rolling(window, min_periods=1, center=True).mean()
    return out.reset_index()


def strategy_table(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver and stint with first and last lap (for a strategy chart)."""
    g = laps.groupby(["Driver", "Stint"])["LapNumber"].agg(Start="min", End="max").reset_index()
    g["End"] = g["End"] + 1
    g["Length"] = g["End"] - g["Start"]
    return g
