"""Step 1: download results and print the pole-sitter baseline.

Run from the repo root:  python scripts/baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1dash.dataset import load_results
from f1dash.features import add_features, pole_baseline

SEASONS = range(2018, 2026)

if __name__ == "__main__":
    results = load_results(SEASONS)
    print(f"{len(results)} driver-race rows, {results.groupby(['Year', 'Round']).ngroup().nunique()} races")
    results.to_csv("results_2018_2025.csv", index=False)
    feats = add_features(results)
    feats.to_csv("features.csv", index=False)
    table = pole_baseline(feats)
    print(table.to_string(index=False))
    total = table["PoleWins"].sum() / table["Races"].sum()
    print(f"\nOverall pole-sitter win rate: {total:.3f}  <- the model has to beat this")
