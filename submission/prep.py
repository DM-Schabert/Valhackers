"""Data cleaning + features for the submission: a copy of claude-preprocessing/prep.py, the preprocessing the
HPC search used. One change, which only matters if a 2017 booking lacks a column: the duplicate count also
works without booking_id or with a missing raw column (counted as empty) instead of raising.

Shared data cleaning + features for training AND for Solution.decide().

Both places must run the exact same code, otherwise the model sees different features in 2017.
What each step does: docs/cleaning_steps.md. Walkthrough: preprocessing.ipynb, section 3.

Training (whole table at once), see train.py:
    params = fit_params(raw_train, variant="baseline")   # learned from training data only
    df = add_training_columns(prepare(raw_train, params))
    X = df[~df["drop_for_training"]][params["features"]]

Inside the Solution (one booking at a time):
    decide():  x = prepare(booking, params, counter=counter)[params["features"]]
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "cancelled"

# Time split by arrival date: train < VALID_START <= valid < TEST_START <= test
VALID_START = pd.Timestamp("2016-07-01")  # train: 1 year (Jul 2015 - Jun 2016)
TEST_START = pd.Timestamp("2016-10-01")  # valid: Jul - Sep 2016, test: Oct - Dec 2016
SPLIT_DIR = Path(__file__).resolve().parent / "data" / "splits"  # written by preprocessing.ipynb, section 4

# Rows that are clearly broken (removed from the training table only, never from the 2017 stream)
PLACEHOLDER_MIN_ADULTS = 10  # fake group-block rows with 26-55 adults
RATE_TYPO_THRESHOLD = 1000.0  # the single EUR 5,400 booking

# Default preprocessing settings. A variant (VARIANTS below) overrides some of them.
DEFAULT_SETTINGS = {
    "top_countries": 20,  # countries kept; all others become "(other)"
    "min_rows_per_category": 50,  # rarer values (agent, company, room type, ...) become the fallback
    "rate_cap_quantile": 0.999,  # rate_capped = nightly_rate capped at this quantile per hotel
    "max_children": 3,  # larger values are typos
    "max_babies": 2,
    "drop_features": [],  # features the model does not get
}

# Categorical column -> value used for rare or unseen values
CATEGORICAL_FALLBACK = {
    "hotel": "other",
    "meal": "other",
    "country": "(other)",
    "sales_segment": "Other",
    "sales_channel": "Other",
    "room_type": "other",
    "deposit": "other",
    "guest_type": "other",
    "agent": "other",
    "company": "other",
}

RAW_NUMERIC = [
    "booking_id", "cancelled", "days_ahead", "arrival_year", "arrival_week", "arrival_day",
    "weekend_nights", "weekday_nights", "adults", "children", "babies", "returning_guest",
    "past_cancellations", "past_stays", "amendments", "agent", "company", "waitlist_days",
    "nightly_rate", "special_requests", "nights",
]

# Columns that identify "the same request" (everything a 2017 row has, except its id)
REQUEST_KEY_COLUMNS = [
    "hotel", "days_ahead", "arrival_year", "arrival_month", "arrival_week", "arrival_day",
    "weekend_nights", "weekday_nights", "adults", "children", "babies", "meal", "country",
    "sales_segment", "sales_channel", "returning_guest", "past_cancellations", "past_stays",
    "room_type", "amendments", "deposit", "agent", "company", "waitlist_days", "guest_type",
    "nightly_rate", "special_requests", "arrival_date", "booking_date", "nights",
]

HOTELS = {"downtown": "Downtown Hotel", "seaside": "Seaside Hotel"}  # short name -> value in the data

CATEGORICAL_FEATURES = list(CATEGORICAL_FALLBACK)
NUMERIC_FEATURES = [
    "days_ahead", "arrival_week", "arrival_day", "weekend_nights", "weekday_nights", "nights",
    "adults", "children", "babies", "total_guests", "no_guests", "has_kids",
    "returning_guest", "past_cancellations", "past_stays", "amendments", "waitlist_days",
    "special_requests", "rate_capped", "rate_rel", "is_zero_rate",
    "arr_month", "arr_dow", "arr_doy", "is_weekend_arrival", "book_month", "book_dow",
    "is_prt", "has_agent", "has_company", "same_request_count",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# One model per hotel: `hotel` is constant inside each hotel's data, so leave it out
FEATURES_PER_HOTEL = [f for f in FEATURES if f != "hotel"]

# Recorded after booking time (still present in every 2017 row) -> test the model with and without them
HINDSIGHT_FEATURES = ["amendments", "special_requests", "waitlist_days", "country", "is_prt", "deposit"]
# Deliberately NOT features: booking_id leaks the target, absolute dates/years don't extrapolate to 2017
NOT_FEATURES = ["booking_id", "cancelled", "arrival_year", "arrival_month", "arrival_date", "booking_date"]

# =============================================================================================
# PREPROCESSING VARIANTS -- pick one with PREPROCESSING in train-template-kit/solution.py.
# A variant overrides DEFAULT_SETTINGS. Add your own: give it a name and the settings it changes.
# A completely new feature: compute it in _add_features(), add the name to NUMERIC_FEATURES
# (then every variant gets it, unless the variant lists it in drop_features).
# =============================================================================================
VARIANTS = {
    "baseline": {},
    "no_country": {"drop_features": ["country", "is_prt"]},  # Portugal artefact: +22k on valid
    "no_hindsight": {"drop_features": HINDSIGHT_FEATURES},  # only what is known at booking time
    "no_season": {"drop_features": ["arr_month", "arr_doy", "book_month"]},  # season = 2015 behaviour
    "fewer_categories": {"top_countries": 10, "min_rows_per_category": 200},
}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    return _to_frame(pd.read_csv(path))


def _to_frame(data) -> pd.DataFrame:
    """DataFrame, Series or a single booking dict -> DataFrame with parsed dates and numeric columns."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame().T
    else:
        df = pd.DataFrame([data])
    # Already-cleaned files (e.g. data/splits/*.parquet): plain values again, rare IDs stay rare
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].astype(object)
    for col in ("agent", "company"):
        if col in df.columns:
            df[col] = df[col].mask(df[col].eq("other"), -1)
    for col in ("arrival_date", "booking_date"):
        df[col] = pd.to_datetime(df[col])
    for col in RAW_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _bad_row_mask(df: pd.DataFrame) -> pd.Series:
    return (df["adults"] >= PLACEHOLDER_MIN_ADULTS) | (df["nightly_rate"] > RATE_TYPO_THRESHOLD)


# ---------------------------------------------------------------------------------------------
# Step 1: fit the cleaning parameters on training data
# ---------------------------------------------------------------------------------------------
def fit_params(raw: pd.DataFrame, until: str | pd.Timestamp | None = None, variant: str = "baseline") -> dict:
    """Learn category lists, rate caps and reference rates from training data.

    until:   only use arrivals before this date (e.g. TEST_START, so the test period has no influence).
    variant: preprocessing variant from VARIANTS; stored in the params so prepare() and the model
             always use the same one.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown preprocessing variant {variant!r}, use one of {list(VARIANTS)}")
    settings = {**DEFAULT_SETTINGS, **VARIANTS[variant]}
    unknown = set(settings["drop_features"]) - set(FEATURES)
    if unknown:
        raise ValueError(f"variant {variant!r} drops unknown features: {sorted(unknown)}")

    df = _to_frame(raw)
    if until is not None:
        df = df[df["arrival_date"] < pd.Timestamp(until)]
    df = df[~_bad_row_mask(df)]

    base = _normalise_categories(df)
    categories = {}
    for col, fallback in CATEGORICAL_FALLBACK.items():
        counts = base[col].value_counts()
        if col == "country":
            keep = [c for c in counts.index if c != "(missing)"][: settings["top_countries"]] + ["(missing)"]
        else:
            keep = counts.index[counts >= settings["min_rows_per_category"]].tolist()
        categories[col] = sorted(set(keep) | {fallback})

    rate_cap = df.groupby("hotel")["nightly_rate"].quantile(settings["rate_cap_quantile"]).round(2)

    # Reference price per hotel and arrival month, taken from the most recent year that has data
    paid = df.loc[df["nightly_rate"] > 0, ["hotel", "arrival_date", "nightly_rate"]]
    paid = paid.assign(year=paid["arrival_date"].dt.year, month=paid["arrival_date"].dt.month)
    latest_year = paid.groupby(["hotel", "month"])["year"].transform("max")
    rate_ref = paid[paid["year"] == latest_year].groupby(["hotel", "month"])["nightly_rate"].median()

    return {
        "variant": variant,
        "settings": settings,
        "features": [f for f in FEATURES if f not in settings["drop_features"]],  # what the model gets
        "fitted_until": None if until is None else str(pd.Timestamp(until).date()),
        "categories": categories,
        "rate_cap": {hotel: float(v) for hotel, v in rate_cap.items()},
        "rate_ref": {f"{hotel}|{month}": round(float(v), 2) for (hotel, month), v in rate_ref.items()},
    }


def save_params(params: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(params, indent=2))


def load_params(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------
# Step 2-4: clean values, derive features (works for a whole table or a single booking)
# ---------------------------------------------------------------------------------------------
def prepare(data, params: dict, counter: "RequestCounter | None" = None) -> pd.DataFrame:
    """Clean + add features. Raw `nightly_rate` stays untouched (revenue is computed on it).

    counter=None  -> batch mode: `data` is the whole request stream (training).
    counter given -> online mode: rows are fed one by one in booking order (decide()).
    """
    df = _to_frame(data)
    df["same_request_count"] = _same_request_counts(df, counter)  # on raw values, before cleaning
    df = _clean_values(df, params)
    return _add_features(df, params)


def _normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["hotel"] = df["hotel"].astype(str)
    out["meal"] = df["meal"].replace("Undefined", "SC").astype(str)  # source docs: Undefined == SC
    for col in ("sales_segment", "sales_channel"):
        out[col] = df[col].replace("Undefined", "Other").astype(str)
    out["country"] = df["country"].fillna("(missing)").astype(str)  # CN stays separate from CHN
    for col in ("agent", "company"):  # IDs, blank = none
        ids = pd.to_numeric(df[col], errors="coerce")
        out[col] = ids.map(lambda v: "none" if pd.isna(v) else str(int(v)))
    for col in ("room_type", "deposit", "guest_type"):
        out[col] = df[col].astype(str)
    return out


def _clean_values(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    settings = params.get("settings", DEFAULT_SETTINGS)  # old params files have no settings yet
    df["children"] = df["children"].fillna(0).clip(upper=settings["max_children"])
    df["babies"] = df["babies"].clip(upper=settings["max_babies"])
    base = _normalise_categories(df)
    for col, allowed in params["categories"].items():
        values = base[col].where(base[col].isin(allowed), CATEGORICAL_FALLBACK[col])
        df[col] = pd.Categorical(values, categories=allowed)
    return df


def _add_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    arrival, booked = df["arrival_date"], df["booking_date"]
    hotel = df["hotel"].astype(str)

    df["stay_value"] = df["nightly_rate"] * df["nights"]  # revenue if seated; walk costs 2.5x this
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["no_guests"] = (df["total_guests"] == 0).astype(int)
    df["has_kids"] = ((df["children"] + df["babies"]) > 0).astype(int)

    df["arr_month"] = arrival.dt.month
    df["arr_dow"] = arrival.dt.dayofweek
    df["arr_doy"] = arrival.dt.dayofyear
    df["is_weekend_arrival"] = arrival.dt.dayofweek.isin([4, 5]).astype(int)  # Fri / Sat
    df["book_month"] = booked.dt.month
    df["book_dow"] = booked.dt.dayofweek

    df["is_prt"] = (df["country"] == "PRT").astype(int)
    df["has_agent"] = (df["agent"] != "none").astype(int)
    df["has_company"] = (df["company"] != "none").astype(int)
    df["is_zero_rate"] = (df["nightly_rate"] == 0).astype(int)

    df["rate_capped"] = df["nightly_rate"].clip(upper=hotel.map(params["rate_cap"]))
    reference = (hotel + "|" + df["arr_month"].astype(str)).map(params["rate_ref"])
    df["rate_rel"] = df["rate_capped"] / reference
    return df


# ---------------------------------------------------------------------------------------------
# Online feature: identical requests already seen on the same booking date
# ---------------------------------------------------------------------------------------------
class RequestCounter:
    """Counts how many identical requests (all columns except the id) were already seen.

    Group blocks arrive as many identical rows on the same booking date; big blocks cancel a lot
    (10+ identical rows: 75-84% cancelled vs 24% for single rows). decide() sees one row at a time,
    so only the count *so far* is known -- training uses exactly the same running count.
    """

    def __init__(self) -> None:
        self._seen: defaultdict[tuple, int] = defaultdict(int)

    def see(self, values: tuple) -> int:
        key = tuple(_key_value(v) for v in values)
        count = self._seen[key]
        self._seen[key] = count + 1
        return count


def _key_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.value
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def _same_request_counts(df: pd.DataFrame, counter: RequestCounter | None) -> pd.Series:
    if counter is None:  # batch: replay the whole table in booking order
        counter = RequestCounter()
        by = ["booking_date", "booking_id"] if "booking_id" in df.columns else ["booking_date"]
        order = df.sort_values(by, kind="stable").index
    else:
        order = df.index
    rows = df.loc[order].reindex(columns=REQUEST_KEY_COLUMNS).itertuples(index=False, name=None)
    counts = [counter.see(row) for row in rows]
    return pd.Series(counts, index=order, dtype="int64").reindex(df.index)


# ---------------------------------------------------------------------------------------------
# Training-only columns
# ---------------------------------------------------------------------------------------------
def add_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flags for the training table. Never filter the 2017 stream with these."""
    df = df.copy()
    df["drop_for_training"] = _bad_row_mask(df)
    # Occupancy can't be trusted here: guests from before 2015-07-01 are missing,
    # and Downtown Jul-Aug 2015 looks incomplete.
    early = df["arrival_date"] < pd.Timestamp("2015-07-15")
    downtown_gap = (df["hotel"] == "Downtown Hotel") & (df["arrival_date"] < pd.Timestamp("2015-09-01"))
    df["use_for_occupancy"] = ~(early | downtown_gap)
    df["split"] = np.select(
        [df["arrival_date"] < VALID_START, df["arrival_date"] < TEST_START], ["train", "valid"], "test"
    )
    # Booked before its split started -> would be a confirmed commitment in a replay of that split
    split_start = df["split"].map({"train": pd.NaT, "valid": VALID_START, "test": TEST_START})
    df["is_commitment"] = df["booking_date"] < split_start
    return df


def load_split(name: str, hotel: str | None = None) -> pd.DataFrame:
    """Load 'train', 'valid' or 'test' (cleaned + features), written by preprocessing.ipynb section 4.

    One data source for both hotels; hotel: None = both hotels (one joint model),
    'downtown' / 'seaside' = only that hotel (one model per hotel).
    """
    if name not in ("train", "valid", "test"):
        raise ValueError(f"unknown split {name!r}, use 'train', 'valid' or 'test'")
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run preprocessing.ipynb (section 4) first")
    df = pd.read_parquet(path)
    if hotel is None:
        return df
    return df[df["hotel"] == HOTELS[_hotel_key(hotel)]].reset_index(drop=True)


def _hotel_key(hotel: str) -> str:
    key = hotel.lower().split()[0]  # accepts 'downtown' and 'Downtown Hotel'
    if key not in HOTELS:
        raise ValueError(f"unknown hotel {hotel!r}, use one of {list(HOTELS)}")
    return key
