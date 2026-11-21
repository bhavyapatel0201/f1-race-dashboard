"""Final test on the 2026 season. Run ONCE, after choosing a model with walkforward.py.

Example:  python scripts/final_eval.py --features relative --kind logistic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1dash.dataset import load_results
from f1dash.features import FEATURE_SETS, build_features
from f1dash.model import evaluate, fit_predict, pole_prediction

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURE_SETS, required=True)
    ap.add_argument("--kind", choices=["logistic", "lightgbm"], required=True)
    args = ap.parse_args()

    df = build_features(load_results(range(2018, 2027)))
    train, test = df[df["Year"] <= 2025], df[df["Year"] == 2026]
    if test.empty:
        sys.exit("No 2026 races found in the API yet.")
    print(f"Trained on 2018-2025, testing on {test['RaceIndex'].nunique()} races of 2026\n")
    print("pole sitter     ", evaluate(pole_prediction(test)))
    print("grid (logistic) ", evaluate(fit_predict(train, test, FEATURE_SETS["grid"], "logistic")))
    print(f"{args.features} ({args.kind})".ljust(16),
          evaluate(fit_predict(train, test, FEATURE_SETS[args.features], args.kind)))
