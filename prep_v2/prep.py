"""Preprocessing v2 (lean): cleaning + features, written for a linear model (logistic regression).

    python prep_v2/prep.py [variant]     # variant from VARIANTS, default "baseline"

Writes data/splits_v2/{train,valid,test}.parquet and data/splits_v2/params.json.
Everything learned from data (categories, cancel rates, price levels) comes from the TRAIN period only,
so neither valid nor test leaks into the features.

Features:
- cancel rate per agent and company (smoothed)          -> agent_rate, company_rate
- log of skewed numbers                                  -> log_days_ahead, log_nightly_rate, log_waitlist_days, ...
- price compared to the hotel's usual price that month   -> rate_rel
- identical requests already seen that booking day       -> log_same_request_count
- no country, no season, no lead-time steps, no combinations: those made valid AND test worse (ablation).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

VALID_START, TEST_START = pd.Timestamp("2016-07-01"), pd.Timestamp("2016-10-01")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "splits_v2"

MIN_ROWS = 50  # rarer category values become "other"
SMOOTHING = 50  # cancel rates: a value with n rows is pulled towards the overall rate with weight 50
TARGET_ENCODED = ["agent", "company"]

CATEGORICAL = ["hotel", "meal", "sales_segment", "sales_channel", "room_type", "deposit", "guest_type",
               "agent", "company", "arr_dow"]
NUMERIC = ["log_days_ahead", "nights", "weekend_nights", "adults", "children", "babies", "returning_guest",
           "amendments", "special_requests", "log_waitlist_days", "log_past_cancellations", "log_past_stays",
           "log_nightly_rate", "rate_rel", "log_same_request_count", "agent_rate", "company_rate"]
FEATURES = NUMERIC + CATEGORICAL

VARIANTS = {  # features each variant leaves out
    "baseline": [],
    "no_hindsight": ["amendments", "special_requests", "log_waitlist_days"],  # only what is known at booking time
}


# Raw columns that define an "identical request" (every raw column except booking_id and cancelled)
KEY_COLUMNS = ["hotel", "days_ahead", "arrival_year", "arrival_month", "arrival_week", "arrival_day",
               "weekend_nights", "weekday_nights", "adults", "children", "babies", "meal", "country",
               "sales_segment", "sales_channel", "returning_guest", "past_cancellations", "past_stays",
               "room_type", "amendments", "deposit", "agent", "company", "waitlist_days", "guest_type",
               "nightly_rate", "special_requests", "arrival_date", "booking_date", "nights"]
RAW_NUMERIC = ["days_ahead", "arrival_year", "arrival_week", "arrival_day", "weekend_nights", "weekday_nights",
               "adults", "children", "babies", "returning_guest", "past_cancellations", "past_stays", "amendments",
               "agent", "company", "waitlist_days", "nightly_rate", "special_requests", "nights"]


def to_frame(data):
    """A DataFrame, a Series or a single booking dict -> DataFrame with parsed dates and numeric columns."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame().T
    else:
        df = pd.DataFrame([data])
    for col in ("arrival_date", "booking_date"):
        df[col] = pd.to_datetime(df[col])
    for col in RAW_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class RequestCounter:
    """Running count of identical requests seen so far (group blocks = many identical rows on one booking day).
    decide() sees one booking at a time, so the count has to be remembered between calls; training replays
    the whole table in booking order through the same counter, so both give exactly the same numbers."""

    def __init__(self):
        self.seen = defaultdict(int)

    def see(self, row):
        key = tuple(_key_value(v) for v in row)
        count = self.seen[key]
        self.seen[key] = count + 1
        return count


def _key_value(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.value
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def same_request_counts(df, counter=None):
    if counter is None:  # batch (training): replay the whole table in booking order
        counter = RequestCounter()
        by = ["booking_date", "booking_id"] if "booking_id" in df.columns else ["booking_date"]
        order = df.sort_values(by, kind="stable").index
    else:  # online (decide): rows already arrive in booking order
        order = df.index
    rows = df.loc[order].reindex(columns=KEY_COLUMNS).itertuples(index=False, name=None)
    return pd.Series([counter.see(r) for r in rows], index=order, dtype="int64").reindex(df.index)


def bad_rows(df):
    """Fake group-block rows (26-55 adults) and the EUR 5,400 typo: dropped from training."""
    return (df["adults"] >= 10) | (df["nightly_rate"] > 1000)


def categories(df):
    """All categorical columns as plain text (gaps and 'Undefined' cleaned)."""
    c = pd.DataFrame(index=df.index)
    for col in ("hotel", "room_type", "deposit", "guest_type"):
        c[col] = df[col].astype(str)
    c["meal"] = df["meal"].replace("Undefined", "SC").astype(str)  # source docs: Undefined == SC
    for col in ("sales_segment", "sales_channel"):
        c[col] = df[col].replace("Undefined", "other").astype(str)
    for col in ("agent", "company"):  # IDs, not quantities
        c[col] = df[col].map(lambda v: "none" if pd.isna(v) else str(int(v)))
    c["arr_dow"] = df["arrival_date"].dt.day_name()
    return c


def fit_params(train, variant="baseline"):
    """Learn category lists, smoothed cancel rates and price levels from the training rows."""
    train = train[~bad_rows(train)]
    cats, y = categories(train), train["cancelled"]
    prior = y.mean()
    rates = {}
    for col in TARGET_ENCODED:
        stats = y.groupby(cats[col]).agg(["sum", "count"])
        rates[col] = ((stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)).round(4).to_dict()
    paid = train[train["nightly_rate"] > 0]
    rate_ref = paid.groupby([paid["hotel"], paid["arrival_date"].dt.month])["nightly_rate"].median()
    return {
        "variant": variant,
        "features": [f for f in FEATURES if f not in VARIANTS[variant]],
        "categories": {col: sorted(set(cats[col].value_counts().loc[lambda n: n >= MIN_ROWS].index) | {"other"})
                       for col in CATEGORICAL},
        "rates": rates,
        "prior": round(float(prior), 4),
        "rate_ref": {f"{hotel}|{month}": float(v) for (hotel, month), v in rate_ref.items()},
    }


def prepare(data, params, counter=None):
    """Clean + add all features. Raw nightly_rate stays untouched (revenue is computed on it).
    counter=None -> batch: `data` is the whole request table (training).
    counter given -> online: one booking (dict) at a time in booking order (solution.decide)."""
    df = to_frame(data)
    # identical requests already seen on the same booking date (group blocks cancel a lot)
    df["same_request_count"] = same_request_counts(df, counter)

    df["children"] = df["children"].fillna(0).clip(upper=3)
    df["babies"] = df["babies"].clip(upper=2)
    cats = categories(df)

    for col in TARGET_ENCODED:
        df[f"{col}_rate"] = cats[col].map(params["rates"][col]).fillna(params["prior"])
    for col in ("days_ahead", "nightly_rate", "waitlist_days", "past_cancellations", "past_stays", "same_request_count"):
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    month = df["arrival_date"].dt.month.astype(str)
    df["rate_rel"] = df["nightly_rate"] / (cats["hotel"] + "|" + month).map(params["rate_ref"]).astype(float)

    for col, allowed in params["categories"].items():  # rare / unseen values -> "other"
        df[col] = pd.Categorical(cats[col].where(cats[col].isin(allowed), "other"), categories=allowed)
    return df


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    raw = pd.read_csv(ROOT / "train-template-kit" / "train.csv", parse_dates=["arrival_date", "booking_date"])
    params = fit_params(raw[raw["arrival_date"] < VALID_START], variant)  # train period only
    data = prepare(raw, params)

    split = pd.Series(np.select([data["arrival_date"] < VALID_START, data["arrival_date"] < TEST_START],
                                ["train", "valid"], "test"), index=data.index)
    keep = ~(bad_rows(data) & split.eq("train"))  # broken rows only matter for training
    data, split = data[keep], split[keep]

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        data[split == name].reset_index(drop=True).to_parquet(OUT / f"{name}.parquet", index=False)
    (OUT / "params.json").write_text(json.dumps(params, indent=2))
    print(f"variant '{variant}': {len(params['features'])} features -> {OUT}")
