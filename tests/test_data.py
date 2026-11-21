from f1dash.data import parse_laps, parse_pits, parse_results, parse_time


def test_parse_time():
    assert parse_time("1:14.123") == 74.123
    assert parse_time("59.9") == 59.9
    assert parse_time("") is None and parse_time(None) is None


def test_parse_laps_split_across_pages():
    def page(timings):
        return {"RaceTable": {"Races": [{"Laps": [{"number": "1", "Timings": timings}]}]}}
    pages = [page([{"driverId": "max_verstappen", "position": "1", "time": "1:30.0"}]),
             page([{"driverId": "leclerc", "position": "2", "time": "1:31.0"}])]
    df = parse_laps(pages, {"max_verstappen": "VER", "leclerc": "LEC"})
    assert sorted(df["Driver"]) == ["LEC", "VER"] and df["LapNumber"].eq(1).all()


def test_parse_results_and_pits():
    res_page = {"RaceTable": {"Races": [{"Results": [{
        "position": "1", "grid": "2", "status": "Finished", "laps": "78",
        "Driver": {"driverId": "leclerc", "code": "LEC"},
        "Constructor": {"name": "Ferrari"}}]}]}}
    pit_page = {"RaceTable": {"Races": [{"PitStops": [
        {"driverId": "leclerc", "stop": "1", "lap": "20", "duration": "22.5"}]}]}}
    r = parse_results([res_page])
    p = parse_pits([pit_page], {"leclerc": "LEC"})
    assert r.loc[0, "Driver"] == "LEC" and r.loc[0, "Grid"] == 2
    assert p.loc[0, "Lap"] == 20 and p.loc[0, "PitSeconds"] == 22.5
