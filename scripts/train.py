"""Step 2: compare pole baseline, grid-only model and full models on held-out seasons.

Run from the repo root:  python scripts/train.py
Split by season (never random): train 2018-2022, validate 2023, test 2024-2025.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from f1dash.dataset import load_results
from f1dash.features import FEATURE_COLUMNS, add_features, time_split
from f1dash.model import GRID_ONLY, evaluate, fit_predict, pole_prediction

if __name__ == "__main__":
    feats = add_features(load_results(range(2018, 2026)))
    train, valid, test = time_split(feats, 2022, 2023)
    print(f"train {train['Year'].min()}-{train['Year'].max()}, "
          f"valid {valid['Year'].min()}, test {test['Year'].min()}-{test['Year'].max()}\n")

    rows = []
    for split_name, held_out, fit_on in [("validation 2023", valid, train),
                                         ("test 2024-25", test, pd.concat([train, valid]))]:
        candidates = {
            "pole sitter": pole_prediction(held_out),
            "grid only (logistic)": fit_predict(fit_on, held_out, GRID_ONLY, "logistic"),
            "all features (logistic)": fit_predict(fit_on, held_out, FEATURE_COLUMNS, "logistic"),
            "all features (lightgbm)": fit_predict(fit_on, held_out, FEATURE_COLUMNS, "lightgbm"),
        }
        for name, scored in candidates.items():
            rows.append({"split": split_name, "model": name, **evaluate(scored)})

    print(pd.DataFrame(rows).to_string(index=False))
    print("\ntop1 = races where the most-likely driver won (higher is better)")
    print("winner_logloss, brier = probability quality (lower is better)")
    print("Pick the model on the validation row; report the test row once, at the end.")
