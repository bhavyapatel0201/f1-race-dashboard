"""Win-probability model: fit on past seasons, score future seasons."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1dash.features import FEATURE_COLUMNS

GRID_ONLY = ["GridPos"]


def make_model(kind: str):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=1000))
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier  # imported lazily: only needed for this kind
        return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8,
                              min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                              random_state=0, verbose=-1)
    raise ValueError(f"unknown model kind: {kind}")


def normalise_per_race(df: pd.DataFrame, col: str = "RawProb") -> pd.DataFrame:
    """Turn independent per-driver scores into win probabilities that sum to 1 per race."""
    out = df.copy()
    out["WinProb"] = out[col] / out.groupby("RaceIndex")[col].transform("sum")
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features=FEATURE_COLUMNS,
                kind: str = "logistic") -> pd.DataFrame:
    model = make_model(kind)
    model.fit(train[features], train["Won"])
    scored = test.copy()
    scored["RawProb"] = model.predict_proba(test[features])[:, 1]
    return normalise_per_race(scored)


def evaluate(scored: pd.DataFrame, prob_col: str = "WinProb") -> dict:
    """Race-level metrics. top1 = share of races whose most-probable driver won."""
    eps = 1e-6
    p = scored[prob_col].clip(eps, 1 - eps)
    y = scored["Won"]
    top = scored.loc[scored.groupby("RaceIndex")[prob_col].idxmax()]
    winner_p = scored[scored["Won"] == 1][prob_col].clip(eps, 1)
    return {
        "races": int(scored["RaceIndex"].nunique()),
        "top1": round(float(top["Won"].mean()), 3),
        "winner_logloss": round(float(-np.log(winner_p).mean()), 3),  # lower is better
        "brier": round(float(((p - y) ** 2).mean()), 4),
    }


def pole_prediction(test: pd.DataFrame) -> pd.DataFrame:
    """Baseline: all probability on the driver with the best grid slot (ties split)."""
    out = test.copy()
    best = out.groupby("RaceIndex")["GridPos"].transform("min")
    out["RawProb"] = (out["GridPos"] == best).astype(float)
    return normalise_per_race(out)
