"""Load race data from the free Jolpica-F1 API (Ergast-compatible, no login).

This is the only module that touches the network. Responses are cached on disk
as JSON so each race is downloaded once.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = Path("cache")
PAGE_SIZE = 100


def _get_json(path: str, params: dict | None = None) -> dict:
    """GET one API page, using the on-disk cache."""
    params = params or {}
    key = hashlib.md5(f"{path}|{sorted(params.items())}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    resp = requests.get(f"{BASE_URL}/{path}", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def _get_all_pages(path: str) -> list[dict]:
    """Fetch every page of a paginated endpoint; return the list of MRData pages."""
    pages, offset = [], 0
    while True:
        page = _get_json(path, {"limit": PAGE_SIZE, "offset": offset})
        pages.append(page["MRData"])
        total = int(page["MRData"]["total"])
        offset += PAGE_SIZE
        if offset >= total:
            return pages


def parse_time(text: str) -> float | None:
    """'1:14.123' -> 74.123 seconds; '59.9' -> 59.9; bad input -> None."""
    try:
        if text is None:
            return None
        if ":" in text:
            minutes, seconds = text.split(":")
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return None


def get_schedule(year: int) -> pd.DataFrame:
    """One row per round: Round, Name, Date."""
    races = _get_all_pages(f"{year}.json")[0]["RaceTable"]["Races"]
    return pd.DataFrame(
        [{"Round": int(r["round"]), "Name": r["raceName"], "Date": r["date"]} for r in races]
    )


def parse_results(pages: list[dict]) -> pd.DataFrame:
    rows = []
    for page in pages:
        for race in page["RaceTable"]["Races"]:
            for r in race["Results"]:
                rows.append({
                    "DriverId": r["Driver"]["driverId"],
                    "Driver": r["Driver"].get("code", r["Driver"]["driverId"][:3].upper()),
                    "Team": r["Constructor"]["name"],
                    "Grid": int(r["grid"]),
                    "Finish": int(r["position"]),
                    "Status": r["status"],
                    "Laps": int(r["laps"]),
                })
    return pd.DataFrame(rows)


def parse_laps(pages: list[dict], id_to_code: dict) -> pd.DataFrame:
    """Flatten lap timings. A lap can be split across pages, so don't group by page."""
    rows = []
    for page in pages:
        for race in page["RaceTable"]["Races"]:
            for lap in race.get("Laps", []):
                for t in lap["Timings"]:
                    rows.append({
                        "Driver": id_to_code.get(t["driverId"], t["driverId"]),
                        "LapNumber": int(lap["number"]),
                        "Position": int(t["position"]),
                        "LapTimeSeconds": parse_time(t["time"]),
                    })
    return pd.DataFrame(rows)


def parse_pits(pages: list[dict], id_to_code: dict) -> pd.DataFrame:
    rows = []
    for page in pages:
        for race in page["RaceTable"]["Races"]:
            for p in race.get("PitStops", []):
                rows.append({
                    "Driver": id_to_code.get(p["driverId"], p["driverId"]),
                    "Stop": int(p["stop"]),
                    "Lap": int(p["lap"]),
                    "PitSeconds": parse_time(p["duration"]),
                })
    return pd.DataFrame(rows)


def load_race(year: int, rnd: int) -> dict:
    """Return {'results', 'laps', 'pits'} DataFrames for one race."""
    results = parse_results(_get_all_pages(f"{year}/{rnd}/results.json"))
    id_to_code = dict(zip(results["DriverId"], results["Driver"]))
    laps = parse_laps(_get_all_pages(f"{year}/{rnd}/laps.json"), id_to_code)
    pits = parse_pits(_get_all_pages(f"{year}/{rnd}/pitstops.json"), id_to_code)
    return {"results": results, "laps": laps, "pits": pits}
