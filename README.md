# F1 Race Data Dashboard and Winner Predictor

An interactive dashboard on real Formula 1 data, plus a race-winner prediction model that is
evaluated honestly: leak-free features, walk-forward validation, and a final test on a season the
model never saw while it was being developed.

All data comes from the free [Jolpica-F1](https://github.com/jolpica/jolpica-f1) API. No account,
key or login is needed.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Pick a season and race in the sidebar. Responses are cached in `cache/`. The first open of the
"Win probabilities" tab downloads 2018-2026 and takes about a minute.

## Dashboard tabs

| Tab | What it shows |
|---|---|
| Results | Finishing order, grid, team, laps, status |
| Race trace | Every driver's position by lap (team colours, dashed = a team's second driver) |
| Driver gap | Cumulative pace difference between any two drivers |
| Strategy | Stints per driver, inferred from pit stops, in finishing order |
| Stint pace | Seconds lost per lap of tyre age per stint (stints of 8+ clean laps) |
| Track evolution | Field median lap time per lap |
| Win probabilities | Modelled win chance per driver, grid-only model vs grid + form + Elo model |

## The winner predictor

**Question:** given what is known before the lights go out, how likely is each driver to win?

**Data:** 2018-2025 for development, 2026 held back as a final test (173 development races).

**Features** (all built only from earlier races):

- Starting grid position
- Driver form (average finish, last 5 races) and recent wins (last 10)
- Team form (average finish of the team's cars, last 5 races)
- The driver's average finish at that circuit
- Elo-style ratings for drivers and teams, updated after every race from who finished ahead of
  whom, pulled 25% back toward the mean at each new season
- Championship points share before the race (approximate; see limitations)
- Form and Elo expressed relative to the rest of that race's field

**Leakage controls.** Every rolling feature is shifted so a race never sees its own result, and
ratings are recorded before they are updated. Tests change a race's result and assert that none of
that race's own features move. Evaluation is by season, never a random split.

**Models:** pole sitter (all probability on P1), logistic regression, LightGBM. Per-driver scores
are normalised so each race's probabilities sum to 1.

### Evaluation 1: walk-forward, 2020-2025 (development)

For each season Y, train on all seasons before Y and predict Y.

Winner picked correctly (top1, higher is better):

| Season | Pole sitter | Grid (logistic) | v1 features (LightGBM) | Relative (logistic) | Relative (LightGBM) |
|---|---|---|---|---|---|
| 2020 | 0.588 | 0.588 | 0.588 | 0.647 | 0.588 |
| 2021 | 0.500 | 0.500 | 0.500 | 0.409 | 0.455 |
| 2022 | 0.455 | 0.455 | 0.455 | 0.364 | 0.364 |
| 2023 | 0.636 | 0.636 | 0.864 | 0.909 | 0.773 |
| 2024 | 0.458 | 0.458 | 0.500 | 0.417 | 0.333 |
| 2025 | 0.667 | 0.667 | 0.458 | 0.542 | 0.542 |

Log loss of the probability given to the actual winner (lower is better):

| Season | Grid (logistic) | v1 (LightGBM) | Relative (logistic) | Relative (LightGBM) |
|---|---|---|---|---|
| 2020 | 1.704 | 1.646 | 1.365 | 1.623 |
| 2021 | 1.711 | 1.808 | 1.495 | 1.497 |
| 2022 | 1.929 | 1.194 | 1.475 | 1.383 |
| 2023 | 1.545 | 0.650 | 0.715 | 0.661 |
| 2024 | 1.738 | 1.675 | 1.593 | 1.774 |
| 2025 | 1.229 | 1.110 | 1.190 | 1.106 |

Reading it: no model picks winners reliably better than the pole sitter (about 55% of races). The
2023 figures are inflated by one dominant driver (he won 19 of 22). The feature models do give
better probabilities than grid-only in every season, and the relative-strength logistic model had the
lowest average log loss, so it was chosen before the final test.

### Evaluation 2: final test, 2026 (20 races, run once)

| Model | Top1 | Log loss | Brier |
|---|---|---|---|
| Pole sitter | 0.70 | 4.145 | 0.0273 |
| Grid (logistic) | 0.70 | 1.665 | 0.0274 |
| Relative (logistic), the chosen model | 0.45 | 1.589 | 0.0311 |

The chosen model did not hold up on picking winners and was mixed on probabilities (better log loss,
worse Brier). 2026 is a new-regulations season, so form and Elo carried over the old order, while
the starting grid reflects the new one immediately. With 20 races, one race moves top1 by 5 points,
so this is evidence, not proof. The honest summary: **in this data the starting grid carries most of
the signal, and form features add little and can hurt when the rules change.**

## Limitations

- Results-only data: no qualifying lap times, tyre compounds, weather or incidents.
- Stints are inferred from pit stops; tyre degradation includes fuel burn-off, so it understates
  true tyre wear. Driver-gap curves bend where drivers pit on different laps.
- Elo treats a mechanical retirement like a bad race.
- Championship points are approximated from finishing order (no sprint or fastest-lap points).
- Small samples: 20 to 24 races per season.
- Not betting advice. Probabilities describe a simple model, not the sport.

## Project layout

```
app.py                  Streamlit dashboard
src/f1dash/data.py      Jolpica loader for one race (the only network code, with dataset.py)
src/f1dash/dataset.py   Multi-season results table
src/f1dash/analysis.py  Stints, cleaning, pace and degradation (pure pandas)
src/f1dash/features.py  Leak-free features, Elo, points, relative strength
src/f1dash/model.py     Logistic / LightGBM, normalisation, evaluation, walk-forward
src/f1dash/predict.py   Win probabilities for one race
src/f1dash/charts.py    Plotly figures
scripts/baseline.py     Pole-sitter win rate by season
scripts/train.py        First comparison on a fixed split
scripts/walkforward.py  Walk-forward comparison, 2020-2025
scripts/final_eval.py   One-time test on 2026
tests/                  pytest, offline
```

## Reproduce the evaluation

```
pytest
python scripts/baseline.py
python scripts/walkforward.py
python scripts/final_eval.py --features relative --kind logistic   # run once
```
