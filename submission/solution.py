"""Our submission: claude-preprocessing features -> blend of logistic regression and LightGBM -> overbooking rule.
Every setting below was chosen by the HPC search over four time windows (hpc/results/best.json, APPROACH.md 11.8).

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

evaluate(Solution) creates Solution(commitments, capacity) once and calls decide(booking) for every
2017 request in booking-date order. __init__ loads the model trained beforehand on all of train.csv
(model.joblib, made by export_model.py on the HPC). If that file is missing, or was made with other
settings, another prep.py or other package versions, it trains from train.csv instead (about a minute).

How it decides:
1. P(cancel) for every booking: prep.py features without the calendar ones
   -> 0.7 x logistic regression + 0.3 x LightGBM, one model for both hotels.
2. Each hotel keeps, per night, the expected number of accepted guests who show up (sum of P(show)).
3. A request is accepted if it pays and, on every night of its stay,
       expected guests + P(show) <= free rooms x (ALPHA + BETA x price tilt)
   with price tilt = rate / normal rate of that hotel and month - 1, limited to -0.5 ... +1.
   ALPHA = 1.1 overbooks by 10%: the model predicts 22-29% cancellations where 35-39% happen.

Check before submitting: bash hpc/hpc.sh check  (runs submission/check.py on the HPC)
"""
import hashlib
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep  # noqa: E402  (submission/prep.py)

# =============================================================================================
# SETTINGS (from hpc/results/best.json)
# =============================================================================================
MODEL_FILE = HERE / "model.joblib"  # trained beforehand by export_model.py
TRAIN_CSV_PLACES = [HERE / "train.csv", HERE.parent / "train-template-kit" / "train.csv"]
CALENDAR = ["arr_month", "arr_doy", "book_month", "arrival_week"]  # feature set "no_calendar"
FEATURES = [f for f in prep.FEATURES if f not in CALENDAR]
LOGREG_WEIGHT = 0.7  # blend: 0.7 x logistic regression + 0.3 x LightGBM
LOGREG = dict(C=0.3, max_iter=2000)
LIGHTGBM = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=0, verbose=-1)
ALPHA = 1.1  # free rooms x 1.1: overbook by 10%
Z = 0.0  # safety margin in standard deviations of the number of shows (made no difference)
BETA = 0.1  # price tilt: bookings above the normal price may fill a night a little more
# =============================================================================================


def train_csv():
    for path in TRAIN_CSV_PLACES:
        if path.exists():
            return path
    raise FileNotFoundError(f"train.csv not found, looked in {[str(p) for p in TRAIN_CSV_PLACES]}")


class Model:
    """P(cancel) = 0.7 x logistic regression + 0.3 x LightGBM."""

    def __init__(self, logreg, lightgbm_model):
        self.logreg, self.lightgbm = logreg, lightgbm_model

    @classmethod
    def fit(cls, rows):
        X, y = rows[FEATURES], rows[prep.TARGET]
        cats = [c for c in FEATURES if X[c].dtype == "category"]
        numbers = [c for c in FEATURES if c not in cats]
        columns = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logreg = make_pipeline(columns, LogisticRegression(**LOGREG)).fit(X, y)
            lightgbm_model = LGBMClassifier(**LIGHTGBM).fit(X, y)
        return cls(logreg, lightgbm_model)

    def cancel(self, rows):
        X = rows[FEATURES]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = (LOGREG_WEIGHT * self.logreg.predict_proba(X)[:, 1]
                 + (1 - LOGREG_WEIGHT) * self.lightgbm.predict_proba(X)[:, 1])
        return np.clip(p, 1e-6, 1 - 1e-6)


def fit_model(train_end=None):
    """Preprocessing parameters and model learned from train.csv (arrivals before train_end; None = all)."""
    raw = prep.load_raw(train_csv())
    if train_end is not None:
        raw = raw[raw["arrival_date"] < pd.Timestamp(train_end)]
    params = prep.fit_params(raw)
    rows = prep.prepare(raw, params)
    return params, Model.fit(rows[~prep._bad_row_mask(rows)])


def fingerprint():
    """What a saved model depends on: the model settings, prep.py and the package versions (major.minor)."""
    def minor(version):
        return ".".join(version.split(".")[:2])
    return {
        "settings": {"features": FEATURES, "logreg_weight": LOGREG_WEIGHT, "logreg": LOGREG, "lightgbm": LIGHTGBM},
        "prep.py": hashlib.md5(Path(prep.__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": minor(np.__version__), "pandas": minor(pd.__version__),
                     "scikit-learn": minor(sklearn.__version__), "lightgbm": minor(lightgbm.__version__)},
    }


def load_saved():
    """(params, model, None) from MODEL_FILE, or (None, None, why it can't be used)."""
    if not MODEL_FILE.exists():
        return None, None, f"{MODEL_FILE.name} not found"
    try:
        saved = joblib.load(MODEL_FILE)
    except Exception as error:
        return None, None, f"{MODEL_FILE.name} could not be loaded: {error!r}"
    want, have = fingerprint(), saved.get("fingerprint", {})
    changed = [key for key in want if want[key] != have.get(key)]
    if changed:
        detail = f" (saved with {have.get('versions')}, installed here {want['versions']})" if "versions" in changed else ""
        return None, None, f"{MODEL_FILE.name} does not fit: {', '.join(changed)} changed{detail}"
    return saved["params"], Model(saved["logreg"], saved["lightgbm"]), None


class Ledger:
    """One hotel: free rooms per night, and the expected shows (mean, variance) of everything accepted."""

    def __init__(self, free):
        self.free = free
        self.mean, self.var = defaultdict(float), defaultdict(float)

    def fits(self, nights, p, rate_rel):
        factor = ALPHA + BETA * price_tilt(rate_rel)
        spread = p * (1 - p)
        return all(self.mean[n] + p + Z * math.sqrt(self.var[n] + spread) <= self.free.get(n, math.inf) * factor
                   for n in nights)

    def add(self, nights, p):
        for n in nights:
            self.mean[n] += p
            self.var[n] += p * (1 - p)


class Solution:
    def __init__(self, commitments, capacity, train_end=None):
        """Called once, before 2017 starts.

        commitments  2017 arrivals booked before 2017 (already confirmed; raw rows, no cancelled column needed).
        capacity     hotel -> rooms free on each night (a Series indexed by date, or a {date: rooms} dict).
        train_end    only for checks: train on arrivals before this date (None = use model.joblib or all of train.csv).
        """
        params = model = None
        if train_end is None:
            params, model, problem = load_saved()
            if problem:
                print(f"{problem}: training from train.csv instead (about a minute)", file=sys.stderr)
        if model is None:
            params, model = fit_model(train_end)
        self.params, self.model = params, model

        self.ledgers = {str(hotel): Ledger(as_series(free)) for hotel, free in capacity.items()}
        self.counter = prep.RequestCounter()
        if len(commitments):
            booked = prep.prepare(commitments, self.params)
            for row, p in zip(booked.to_dict("records"), 1 - self.model.cancel(booked)):
                ledger = self.ledgers.get(str(row["hotel"]))
                if ledger is not None:
                    ledger.add(stay(row), float(p))

    def decide(self, booking):
        """Called once per 2017 booking request, in booking-date order. True = confirm the booking."""
        x = prep.prepare(booking, self.params, counter=self.counter)  # every request, so the counter stays right
        p = float(1 - self.model.cancel(x)[0])
        row = x.iloc[0]
        ledger = self.ledgers.get(str(row["hotel"]))
        if ledger is None or not row["nightly_rate"] > 0:
            return False
        nights = stay(row)
        if ledger.fits(nights, p, row["rate_rel"]):
            ledger.add(nights, p)
            return True
        return False


def price_tilt(rate_rel):
    rate_rel = float(rate_rel)
    return min(max((1.0 if math.isnan(rate_rel) else rate_rel) - 1.0, -0.5), 1.0)


def stay(row):
    return pd.date_range(pd.Timestamp(row["arrival_date"]), periods=int(row["nights"]))


def as_series(free):
    """Free rooms as a float Series with a DatetimeIndex, from a Series or a {date: rooms} dict."""
    s = free.copy() if isinstance(free, pd.Series) else pd.Series(dict(free))
    s.index = pd.to_datetime(s.index)
    return s.astype(float).sort_index()
