"""Step 3: walk-forward comparison on 2020-2025 (train on all earlier seasons each time).

Run from the repo root:  python scripts/walkforward.py
This is the DEVELOPMENT evaluation: use it to choose a model. The 2026 season is kept
for scripts/final_eval.py and must not be used for choosing anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from f1dash.dataset import load_results
from f1dash.features import FEATURE_SETS, build_features
from f1dash.model import evaluate, pole_prediction, walk_forward

YEARS = range(2020, 2026)

if __name__ == "__main__":
    df = build_features(load_results(range(2018, 2026)))
    candidates = {
        "pole sitter": pole_prediction(df[df["Year"].isin(YEARS)]),
        "grid (logistic)": walk_forward(df, YEARS, FEATURE_SETS["grid"], "logistic"),
        "v1 features (lightgbm)": walk_forward(df, YEARS, FEATURE_SETS["v1"], "lightgbm"),
        "relative (logistic)": walk_forward(df, YEARS, FEATURE_SETS["relative"], "logistic"),
        "relative (lightgbm)": walk_forward(df, YEARS, FEATURE_SETS["relative"], "lightgbm"),
    }
    pooled = pd.DataFrame([{"model": n, **evaluate(s)} for n, s in candidates.items()])
    print("Pooled over 2020-2025\n" + pooled.to_string(index=False))

    per_year = pd.DataFrame({n: {y: evaluate(s[s["Year"] == y])["top1"] for y in YEARS}
                             for n, s in candidates.items()})
    print("\ntop1 by season\n" + per_year.to_string())
    print("\nlogloss by season (lower is better; pole sitter is all-or-nothing, ignore it)")
    ll = pd.DataFrame({n: {y: evaluate(s[s["Year"] == y])["winner_logloss"] for y in YEARS}
                       for n, s in candidates.items() if n != "pole sitter"})
    print(ll.to_string())
