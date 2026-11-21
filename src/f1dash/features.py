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


# ---------------------------------------------------------------------------
# Step 3: relative-strength features (Elo ratings, championship points, field-relative form)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

ELO_START = 1500.0
ELO_K = 24.0
SEASON_CARRYOVER = 0.75  # at each new season, ratings move 25% back toward the mean
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}  # approximate: no sprint / fastest-lap points

FEATURE_SETS = {
    "grid": ["GridPos"],
    "v1": FEATURE_COLUMNS,
    "relative": ["GridPos", "DriverFormRel", "TeamFormRel", "DriverEloRel", "TeamEloRel", "PointsShare"],
}


def add_elo_and_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add DriverElo, TeamElo (values BEFORE the race) and PointsBefore / PointsShare.

    After each race, every driver is compared with every other driver (who finished
    ahead) and ratings move by how surprising each result was. Team ratings use only
    comparisons against other teams' cars. Ratings are written to the row before they
    are updated, so a race never sees its own result.
    """
    df = df.sort_values(["RaceIndex", "Finish"]).reset_index(drop=True).copy()
    d_elo, t_elo = {}, {}
    out_d = np.full(len(df), ELO_START)
    out_t = np.full(len(df), ELO_START)
    last_year = None
    for _, g in df.groupby("RaceIndex", sort=True):
        year = g["Year"].iloc[0]
        if last_year is not None and year != last_year:
            for k in d_elo:
                d_elo[k] = ELO_START + SEASON_CARRYOVER * (d_elo[k] - ELO_START)
            for k in t_elo:
                t_elo[k] = ELO_START + SEASON_CARRYOVER * (t_elo[k] - ELO_START)
        last_year = year

        idx = g.index.to_numpy()
        drivers, teams = g["Driver"].tolist(), g["Team"].tolist()
        fin = g["Finish"].to_numpy()
        rd = np.array([d_elo.get(d, ELO_START) for d in drivers])
        rt = np.array([t_elo.get(t, ELO_START) for t in teams])
        out_d[idx], out_t[idx] = rd, rt

        n = len(g)
        if n < 2:
            continue
        ahead = (fin[:, None] < fin[None, :]).astype(float)   # 1 if row finished ahead of column
        np.fill_diagonal(ahead, 0.0)

        exp_d = 1 / (1 + 10 ** ((rd[None, :] - rd[:, None]) / 400))
        np.fill_diagonal(exp_d, 0.0)
        delta_d = ELO_K * (ahead - exp_d).sum(axis=1) / (n - 1)

        other_team = np.array([[a != b for b in teams] for a in teams])
        exp_t = 1 / (1 + 10 ** ((rt[None, :] - rt[:, None]) / 400))
        denom = np.maximum(other_team.sum(axis=1), 1)
        delta_t = ELO_K * (((ahead - exp_t) * other_team).sum(axis=1)) / denom

        for drv, dd in zip(drivers, delta_d):
            d_elo[drv] = d_elo.get(drv, ELO_START) + dd
        team_delta = pd.Series(delta_t).groupby(pd.Series(teams)).mean()
        for team, dt in team_delta.items():
            t_elo[team] = t_elo.get(team, ELO_START) + dt

    df["DriverElo"], df["TeamElo"] = out_d, out_t

    df["Points"] = df["Finish"].map(POINTS).fillna(0)
    df["PointsBefore"] = df.groupby(["Year", "Driver"])["Points"].transform(
        lambda s: s.cumsum().shift(fill_value=0))
    best = df.groupby("RaceIndex")["PointsBefore"].transform("max")
    df["PointsShare"] = np.where(best > 0, df["PointsBefore"] / best.replace(0, 1), 0.0)
    return df


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Express strength relative to the other drivers in the same race."""
    out = df.copy()
    for col in ["DriverForm", "TeamForm", "DriverElo", "TeamElo"]:
        out[col + "Rel"] = out[col] - out.groupby("RaceIndex")[col].transform("mean")
    return out


def build_features(results: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: results table -> one row per driver-race with all features."""
    return add_relative(add_elo_and_points(add_features(results)))
