"""Multi-season results table for the winner predictor (Jolpica-F1, free)."""
import pandas as pd

from f1dash.data import _get_all_pages


def parse_season_results(pages: list[dict]) -> pd.DataFrame:
    """One row per driver per race for the pages of /{year}/results.json."""
    rows = []
    for page in pages:
        for race in page["RaceTable"]["Races"]:
            for r in race["Results"]:
                rows.append({
                    "Year": int(race["season"]),
                    "Round": int(race["round"]),
                    "Circuit": race["Circuit"]["circuitId"],
                    "Driver": r["Driver"]["driverId"],
                    "Team": r["Constructor"]["constructorId"],
                    "Grid": int(r["grid"]),
                    "Finish": int(r["position"]),
                    "Status": r["status"],
                })
    return pd.DataFrame(rows)


def load_results(seasons) -> pd.DataFrame:
    """Download (and disk-cache) every race result for the given seasons."""
    frames = [parse_season_results(_get_all_pages(f"{y}/results.json")) for y in seasons]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["Year", "Round", "Finish"]).reset_index(drop=True)
