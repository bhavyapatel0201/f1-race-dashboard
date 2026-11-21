"""Pre-race features. Every rolling feature uses only EARLIER races (shift before
rolling), so nothing from the race being predicted can leak into its own row."""
import pandas as pd

FORM_WINDOW = 5
WIN_WINDOW = 10


def add_features(results: pd.DataFrame) -> pd.DataFrame:
    df = results.sort_values(["Year", "Round"]).reset_index(drop=True).copy()
    df["RaceIndex"] = df.groupby(["Year", "Round"]).ngroup()
    df["Won"] = (df["Finish"] == 1).astype(int)
    # Pit-lane starts (grid 0) are effectively the back of the grid
    df["GridPos"] = df["Grid"].where(df["Grid"] > 0, df.groupby("RaceIndex")["Grid"].transform("max") + 1)

    by_driver = df.groupby("Driver")
    df["DriverForm"] = by_driver["Finish"].transform(
        lambda s: s.shift().rolling(FORM_WINDOW, min_periods=1).mean())
    df["DriverRecentWins"] = by_driver["Won"].transform(
        lambda s: s.shift().rolling(WIN_WINDOW, min_periods=1).sum())

    # Team form: mean finish of the team's cars per race, then rolled over earlier races
    team_race = (df.groupby(["Team", "RaceIndex"], as_index=False)["Finish"].mean()
                   .rename(columns={"Finish": "TeamRaceFinish"}).sort_values("RaceIndex"))
    team_race["TeamForm"] = team_race.groupby("Team")["TeamRaceFinish"].transform(
        lambda s: s.shift().rolling(FORM_WINDOW, min_periods=1).mean())
    df = df.merge(team_race[["Team", "RaceIndex", "TeamForm"]], on=["Team", "RaceIndex"], how="left")

    # Driver's average finish at this circuit in earlier seasons/races
    df = df.sort_values("RaceIndex").reset_index(drop=True)
    df["CircuitAvgFinish"] = df.groupby(["Driver", "Circuit"])["Finish"].transform(
        lambda s: s.shift().expanding().mean())

    # Within-race context: how strong the grid slot is relative to the field
    df["GridRank"] = df.groupby("RaceIndex")["GridPos"].rank(method="min")
    return df


FEATURE_COLUMNS = ["GridPos", "DriverForm", "DriverRecentWins", "TeamForm", "CircuitAvgFinish"]


def time_split(df: pd.DataFrame, train_end: int, valid_end: int):
    """Split by season: train <= train_end < valid <= valid_end < test."""
    train = df[df["Year"] <= train_end]
    valid = df[(df["Year"] > train_end) & (df["Year"] <= valid_end)]
    test = df[df["Year"] > valid_end]
    return train, valid, test


def pole_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Per season: how often the driver starting P1 wins (races with a P1 start)."""
    poles = df[df["Grid"] == 1]
    return (poles.groupby("Year")["Won"].agg(PoleWins="sum", Races="count")
                 .assign(PoleWinRate=lambda d: (d["PoleWins"] / d["Races"]).round(3))
                 .reset_index())
