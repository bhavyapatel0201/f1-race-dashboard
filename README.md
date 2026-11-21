# F1 Race Data Dashboard

Interactive dashboard on real Formula 1 race data from the free Jolpica-F1 API
(no login, no key): results, position trace, driver pace gap, pit-stop strategy,
stint pace and track evolution.

## Quick start
```
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Pick a season and race in the sidebar. Responses are cached in `cache/`.

## Layout
- `src/f1dash/data.py`: Jolpica loader (the only network code) and parsers
- `src/f1dash/analysis.py`: pure pandas functions, unit tested
- `app.py`: Streamlit UI
- `tests/`: run `pytest` (offline, synthetic data)

## Method notes
- Source has no tyre compound or flag status. Stints are inferred from pit stops;
  neutralised laps are detected statistically (field median far above the race norm).
- Laps removed before pace analysis: lap 1, in and out laps, neutralised laps, and
  per-driver slow outliers.
- Degradation = linear slope of lap time against stint lap. It also contains fuel
  burn-off, so it understates true tyre wear.
- Why not FastF1: version 3.9 added an F1TV login requirement for timing data.
