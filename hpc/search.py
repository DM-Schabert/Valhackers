"""Search over model settings x accept-rule settings, scored by replaying several time windows.

    python hpc/search.py --count                 number of model settings
    python hpc/search.py --task 3 --tasks 32     every 32nd model setting, starting at the 4th
    python hpc/search.py --references            reject all / accept all / hindsight for every window

Every model setting is trained once per window (only on bookings that arrived before the window)
and then replayed under every accept rule of policy.GRID. One JSON per model setting in
hpc/results/runs/; a second run skips what is already there.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "claude-preprocessing")]
import policy  # noqa: E402
import prep  # noqa: E402

warnings.filterwarnings("ignore")

RAW = HERE.parent / "train-template-kit" / "train.csv"
RESULTS = Path(os.environ.get("VH_RESULTS", HERE / "results"))
RUNS = RESULTS / "runs"
SMOKE = bool(os.environ.get("VH_SMOKE"))
THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))

# name: (first arrival day = end of the training data, first day after the window)
WINDOWS = {
    "jan-aug": ("2016-01-01", "2016-09-01"),
    "apr-nov": ("2016-04-01", "2016-12-01"),
    "jul-dec": ("2016-07-01", "2017-01-01"),
    "oct-dec": ("2016-10-01", "2017-01-01"),
}
if SMOKE:
    WINDOWS = {"sep-nov": ("2016-09-01", "2016-12-01"), "nov-dec": ("2016-11-01", "2017-01-01")}

CALIBRATION_DAYS = 90
MIN_CALIBRATION_ROWS = 500
LEARNING_RATE = {200: 0.05, 600: 0.02}

CALENDAR = ["arr_month", "arr_doy", "book_month", "arrival_week"]
FEATURE_SETS = {
    "baseline": [],
    "no_country": ["country", "is_prt"],
    "no_hindsight": prep.HINDSIGHT_FEATURES,
    "no_calendar": CALENDAR,
    "no_country_calendar": ["country", "is_prt"] + CALENDAR,
}
ABLATIONS = {
    "drop_lead_time": ["days_ahead"],
    "drop_deposit": ["deposit"],
    "drop_after_booking": ["amendments", "special_requests", "waitlist_days"],
    "drop_ids": ["agent", "company", "has_agent", "has_company"],
    "drop_duplicates": ["same_request_count"],
    "drop_price": ["rate_capped", "rate_rel", "is_zero_rate"],
    "drop_guests": ["adults", "children", "babies", "total_guests", "no_guests", "has_kids"],
    "drop_history": ["returning_guest", "past_cancellations", "past_stays"],
    "drop_weekday": ["arr_dow", "is_weekend_arrival", "book_dow", "arrival_day"],
    "drop_stay": ["weekend_nights", "weekday_nights", "nights"],
}
DROPPED = {**FEATURE_SETS, **ABLATIONS}

SHARED = {"features": list(FEATURE_SETS), "half_life": [0, 180], "calibration": ["none", "sigmoid", "isotonic"]}
GRIDS = {
    "logreg": {"C": [0.03, 0.3, 3.0], "bins": [0, 10], "per_hotel": [False, True], **SHARED},
    "lightgbm": {"n_estimators": [200, 600], "num_leaves": [7, 15, 31], "min_child_samples": [50, 200],
                 "per_hotel": [False], **SHARED},
    "blend": {"weight": [0.3, 0.5, 0.7], "per_hotel": [False], **SHARED},
}
BLEND = [{"model": "logreg", "C": 0.3, "bins": 0},
         {"model": "lightgbm", "n_estimators": 600, "num_leaves": 15, "min_child_samples": 50}]
PLAIN = {"per_hotel": False, "features": "baseline", "half_life": 0, "calibration": "none"}
CURRENT = {"model": "logreg", "C": 1.0, "bins": 0, **PLAIN}  # train-template-kit/train.py
ABLATION_BASES = [{"model": "logreg", "C": 0.3, "bins": 0, **PLAIN},
                  {"model": "lightgbm", "n_estimators": 200, "num_leaves": 15, "min_child_samples": 50, **PLAIN}]


def key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def grid():
    return [{"model": name, **dict(zip(options, values))}
            for name, options in GRIDS.items() for values in itertools.product(*options.values())]


def ablations():
    return [{**base, "features": name} for base in ABLATION_BASES for name in ABLATIONS]


def configs():
    """Current submission and ablation bases first, then everything else in a fixed random order,
    so that a run cut short by the time budget still covers the grid evenly."""
    if SMOKE:
        return [CURRENT,
                {"model": "logreg", "C": 0.3, "bins": 10, "per_hotel": True, "features": "no_country",
                 "half_life": 180, "calibration": "sigmoid"},
                {"model": "lightgbm", "n_estimators": 200, "num_leaves": 7, "min_child_samples": 50,
                 "per_hotel": False, "features": "no_calendar", "half_life": 0, "calibration": "isotonic"},
                {"model": "blend", "weight": 0.5, "per_hotel": False, "features": "no_hindsight",
                 "half_life": 180, "calibration": "sigmoid"},
                ABLATION_BASES[0], {**ABLATION_BASES[0], "features": "drop_ids"}]
    rest = grid() + ablations()
    random.Random(0).shuffle(rest)
    unique = {key(cfg): cfg for cfg in [CURRENT] + ABLATION_BASES + rest}
    return list(unique.values())


def features_of(cfg):
    return [f for f in prep.FEATURES if f not in DROPPED[cfg["features"]]]


def estimator(cfg, X):
    cats = [c for c in X.columns if X[c].dtype == "category"]
    if cfg["model"] == "lightgbm":
        return LGBMClassifier(n_estimators=cfg["n_estimators"], learning_rate=LEARNING_RATE[cfg["n_estimators"]],
                              num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=THREADS, random_state=0, verbose=-1)
    numbers = [c for c in X.columns if c not in cats]
    parts = [(OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
             (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers)]
    if cfg["bins"]:
        binned = KBinsDiscretizer(n_bins=cfg["bins"], encode="onehot", strategy="quantile")
        parts.append((make_pipeline(SimpleImputer(strategy="median"), binned), numbers))
    return make_pipeline(make_column_transformer(*parts), LogisticRegression(C=cfg["C"], max_iter=2000))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrator(kind, p, y):
    if kind == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y).predict
    line = LogisticRegression(C=1e6).fit(logit(p), y)
    return lambda p: line.predict_proba(logit(p))[:, 1]


class Predictor:
    """P(cancel) for prepared rows: one model or a blend, joint or per hotel, optionally calibrated."""

    def __init__(self, cfg, rows, start):
        self.cfg, self.features, self.start = cfg, features_of(cfg), pd.Timestamp(start)
        self.parts = [(1.0, cfg)]
        if cfg["model"] == "blend":
            self.parts = [(cfg["weight"], BLEND[0]), (1 - cfg["weight"], BLEND[1])]
        hotel = rows["hotel"].astype(str)
        groups = {h: rows[hotel == h] for h in policy.ROOMS} if cfg["per_hotel"] else {"all": rows}
        self.groups = {name: self._fit(part) for name, part in groups.items()}

    def _fit(self, rows):
        """Models on all rows; the calibration map comes from models that have not seen the last 90 days."""
        models = self._fit_parts(rows)
        recent = rows["arrival_date"] >= self.start - pd.Timedelta(days=CALIBRATION_DAYS)
        earlier, held_back = rows[~recent], rows[recent]
        if (self.cfg["calibration"] == "none" or min(len(earlier), len(held_back)) < MIN_CALIBRATION_ROWS
                or min(earlier[prep.TARGET].nunique(), held_back[prep.TARGET].nunique()) < 2):
            return models, None
        raw = self._raw(self._fit_parts(earlier), held_back)
        return models, calibrator(self.cfg["calibration"], raw, held_back[prep.TARGET].to_numpy())

    def _fit_parts(self, rows):
        X, y = rows[self.features], rows[prep.TARGET]
        weight = None
        if self.cfg["half_life"]:
            age = (self.start - rows["arrival_date"]).dt.days.to_numpy()
            weight = 0.5 ** (age / self.cfg["half_life"])
        models = []
        for share, cfg in self.parts:
            model = estimator(cfg, X)
            if isinstance(model, Pipeline):
                model.fit(X, y, logisticregression__sample_weight=weight)
            else:
                model.fit(X, y, sample_weight=weight)
            models.append((share, model))
        return models

    def _raw(self, models, rows):
        X = rows[self.features]
        return sum(share * model.predict_proba(X)[:, 1] for share, model in models)

    def cancel(self, rows):
        p = np.zeros(len(rows))
        hotel = rows["hotel"].astype(str).to_numpy()
        for name, (models, calibrate) in self.groups.items():
            mask = np.ones(len(rows), bool) if name == "all" else hotel == name
            if mask.any():
                raw = self._raw(models, rows[mask])
                p[mask] = calibrate(raw) if calibrate else raw
        return np.clip(p, 1e-6, 1 - 1e-6)


def prepared(raw, until=None):
    """Cleaning parameters learned from arrivals before `until`, and every row cleaned with them."""
    params = prep.fit_params(raw, until=until)
    return params, prep.add_training_columns(prep.prepare(raw, params))


def history(rows, start):
    rows = rows[(rows["arrival_date"] < start) & ~rows["drop_for_training"]]
    return rows[rows["arrival_date"] >= start - pd.Timedelta(days=150)] if SMOKE else rows


class Window:
    def __init__(self, raw, name):
        self.name = name
        self.start, end = (pd.Timestamp(day) for day in WINDOWS[name])
        self.params, rows = prepared(raw, self.start)
        self.history = history(rows, self.start)
        period = rows[(rows["arrival_date"] >= self.start) & (rows["arrival_date"] < end)]
        self.replay = policy.Replay(self.history, period, self.start)


def run(cfg, windows, rule_list):
    out = RUNS / f"{key(cfg)}.json"
    if out.exists():
        return
    started = time.time()
    result = {"cfg": cfg, "windows": {}}
    for window in windows:
        rows = window.replay.rows
        p = Predictor(cfg, window.history, window.start).cancel(rows)
        y = rows[prep.TARGET].to_numpy()
        profit = window.replay.score(1 - p, rule_list)
        result["windows"][window.name] = {
            "log_loss": round(float(log_loss(y, p)), 4), "auc": round(float(roc_auc_score(y, p)), 4),
            "predicted_cancel": round(float(p.mean()), 4), "actual_cancel": round(float(y.mean()), 4),
            "profit": {hotel: np.round(values).astype(int).tolist() for hotel, values in profit.items()},
        }
    result["seconds"] = round(time.time() - started, 1)
    RUNS.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".tmp")
    partial.write_text(json.dumps(result))
    partial.replace(out)
    print(f"{result['seconds']:7.1f}s  {json.dumps(cfg)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", action="store_true")
    group.add_argument("--task", type=int)
    group.add_argument("--references", action="store_true")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--budget-min", type=float, default=35)
    args = parser.parse_args()

    if args.count:
        print(len(configs()))
        return
    started = time.time()
    raw = prep.load_raw(RAW)
    windows = [Window(raw, name) for name in WINDOWS]
    print(f"{len(windows)} windows prepared in {time.time() - started:.0f}s", flush=True)
    if args.references:
        RESULTS.mkdir(parents=True, exist_ok=True)
        references = {window.name: window.replay.references() for window in windows}
        (RESULTS / "references.json").write_text(json.dumps(references, indent=1))
        return

    todo = configs()[args.task::args.tasks]
    rule_list = policy.rules()
    for done, cfg in enumerate(todo):
        if time.time() - started > args.budget_min * 60:
            print(f"time budget reached after {done} of {len(todo)} settings", flush=True)
            break
        try:
            run(cfg, windows, rule_list)
        except Exception as error:
            print(f"FAILED {json.dumps(cfg)}: {error!r}", flush=True)


if __name__ == "__main__":
    main()
