"""Grid search: train on the train split, score on valid (log-loss, AUC and net profit of the replay).

    python grid_search/grid_search.py logreg --count              number of combinations
    python grid_search/grid_search.py logreg --task 3 --tasks 20  every 20th combination, starting at 3
    python grid_search/grid_search.py logreg --all                every combination, here
    python grid_search/grid_search.py logreg --summary            best results so far
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "train-template-kit"), str(ROOT / "claude-preprocessing")]
import prep  # noqa: E402
from solution import ROOMS, Solution, net_profit  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
THREADS = int(os.environ.get("OMP_NUM_THREADS", "0")) or None

FEATURE_SETS = {
    name: [f for f in prep.FEATURES if f not in prep.VARIANTS[name].get("drop_features", [])]
    for name in ("baseline", "no_country", "no_hindsight", "no_season")
}

GRIDS = {
    "logreg": {
        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        "min_frequency": [5, 20, 50],
        "features": list(FEATURE_SETS),
        "per_hotel": [False, True],
    },
    "lightgbm": {
        "n_estimators": [200, 500, 1000],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 50, 200],
        "features": list(FEATURE_SETS),
        "per_hotel": [False],
    },
}


def combos(name):
    grid = GRIDS[name]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def make_model(name, cfg, X):
    cats = [c for c in X.columns if X[c].dtype == "category"]
    if name == "logreg":
        numbers = [c for c in X.columns if c not in cats]
        columns = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", min_frequency=cfg["min_frequency"]), cats),
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers))
        return make_pipeline(columns, LogisticRegression(C=cfg["C"], max_iter=2000))
    return LGBMClassifier(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                          num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
                          n_jobs=THREADS, random_state=0, verbose=-1)


def fit(name, cfg, rows, features):
    X = rows[features]
    model = make_model(name, cfg, X).fit(X, rows[prep.TARGET])
    model.used_features = features
    return model


def train_models(name, cfg, rows):
    features = FEATURE_SETS[cfg["features"]]
    if not cfg["per_hotel"]:
        return {"all": fit(name, cfg, rows, features)}
    return {hotel: fit(name, cfg, rows[rows["hotel"] == hotel], features) for hotel in ROOMS}


def predict_cancel(models, rows):
    p = np.zeros(len(rows))
    for key, model in models.items():
        mask = np.ones(len(rows), bool) if key == "all" else (rows["hotel"].astype(str) == key).to_numpy()
        p[mask] = model.predict_proba(rows.loc[mask, model.used_features])[:, 1]
    return p


def setup(history, period):
    """Same setup as solution.py's __main__: commitments, requests in booking order, free rooms per night."""
    start = period["arrival_date"].min()
    commitments = period[period["booking_date"] < start]
    requests = period[period["booking_date"] >= start].sort_values("booking_date", kind="stable")
    stayed = history[(history["arrival_date"] < start) & (history["cancelled"] == 0)
                     & (history["arrival_date"] + pd.to_timedelta(history["nights"], unit="D") > start)]
    capacity = {}
    for hotel, rooms in ROOMS.items():
        free = pd.Series(rooms, index=pd.date_range(start, period["arrival_date"].max()))
        for arrival, nights in stayed.loc[stayed["hotel"] == hotel, ["arrival_date", "nights"]].itertuples(index=False):
            free.loc[start: arrival + pd.Timedelta(days=int(nights) - 1)] -= 1
        capacity[hotel] = free.clip(lower=0)
    return commitments, requests, capacity


def replay(models, history, period):
    commitments, requests, capacity = setup(history, period)
    solution = Solution(commitments, capacity, models)
    accepted = [solution.accept(b, p) for b, p in zip(requests.to_dict("records"), solution.p_show(requests))]
    return net_profit(pd.concat([commitments, requests[accepted]]), capacity), sum(accepted), len(requests)


def run(name, cfg, train, valid):
    key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    out = RESULTS / name / f"{key}.json"
    if out.exists():
        return
    started = time.time()
    models = train_models(name, cfg, train)
    p = predict_cancel(models, valid)
    profit, accepted, requests = replay(models, train, valid)
    result = {
        **cfg,
        "net_profit": round(profit), "accepted": accepted, "requests": requests,
        "log_loss": round(log_loss(valid[prep.TARGET], p), 4),
        "auc": round(roc_auc_score(valid[prep.TARGET], p), 4),
        "mean_pred": round(float(p.mean()), 3), "cancel_rate": round(float(valid[prep.TARGET].mean()), 3),
        "seconds": round(time.time() - started, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(json.dumps(result), flush=True)


def summary(name, top):
    files = sorted((RESULTS / name).glob("*.json"))
    if not files:
        print(f"no results in {RESULTS / name}")
        return
    df = pd.DataFrame([json.loads(f.read_text()) for f in files]).sort_values("net_profit", ascending=False)
    print(f"{len(df)} results ({len(combos(name))} combinations in the current grid)\n")
    print(df.head(top).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(GRIDS))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", action="store_true")
    group.add_argument("--task", type=int)
    group.add_argument("--all", action="store_true")
    group.add_argument("--summary", action="store_true")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    if args.count:
        print(len(combos(args.model)))
        return
    if args.summary:
        summary(args.model, args.top)
        return

    task, tasks = (0, 1) if args.all else (args.task, args.tasks)
    train, valid = prep.load_split("train"), prep.load_split("valid")
    for cfg in combos(args.model)[task::tasks]:
        run(args.model, cfg, train, valid)


if __name__ == "__main__":
    main()
