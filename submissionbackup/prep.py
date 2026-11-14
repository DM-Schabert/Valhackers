"""prep_v2 preprocessing for the submission: a copy of prep_v2/prep.py that can also run one booking at a time.

    params = fit_params(train_rows)              # learn categories, cancel rates, usual prices
    x = prepare(df, params)                      # whole table at once (training, commitments)
    x = prepare(one_row_df, params, counter)     # one booking at a time, in booking order (decide)
"""
import math
from collections import defaultdict

import numpy as np
import pandas as pd

MIN_ROWS = 50  # rarer category values become "other"
SMOOTHING = 50  # cancel rates: a value with n rows is pulled towards the overall rate with weight 50
TARGET_ENCODED = ["agent", "company"]

CATEGORICAL = ["hotel", "meal", "sales_segment", "sales_channel", "room_type", "deposit", "guest_type",
               "agent", "company", "arr_dow"]
NUMERIC = ["log_days_ahead", "nights", "weekend_nights", "adults", "children", "babies", "returning_guest",
           "amendments", "special_requests", "log_waitlist_days", "log_past_cancellations", "log_past_stays",
           "log_nightly_rate", "rate_rel", "log_same_request_count", "agent_rate", "company_rate"]
FEATURES = NUMERIC + CATEGORICAL


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


def fit_params(train):
    """Learn category lists, smoothed cancel rates and price levels from the training rows."""
    train = train[~bad_rows(train)]
    cats, y = categories(train), train["cancelled"]
    prior = y.mean()
    rates = {}
    for col in TARGET_ENCODED:
        stats = y.groupby(cats[col]).agg(["sum", "count"])
        rates[col] = ((stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)).to_dict()
    paid = train[train["nightly_rate"] > 0]
    rate_ref = paid.groupby([paid["hotel"], paid["arrival_date"].dt.month])["nightly_rate"].median()
    return {
        "categories": {col: sorted(set(cats[col].value_counts().loc[lambda n: n >= MIN_ROWS].index) | {"other"})
                       for col in CATEGORICAL},
        "rates": rates,
        "prior": float(prior),
        "rate_ref": {f"{hotel}|{month}": float(v) for (hotel, month), v in rate_ref.items()},
    }


class RequestCounter:
    """Counts identical requests (all columns except booking_id) already seen, one booking at a time."""

    def __init__(self):
        self.seen = defaultdict(int)

    def see(self, values):
        key = tuple(_key(v) for v in values)
        count = self.seen[key]
        self.seen[key] = count + 1
        return count


def _key(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.value
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def prepare(df, params, counter=None):
    """Clean + add all features. Raw nightly_rate stays untouched (revenue is computed on it)."""
    df = df.copy()
    for col in ("arrival_date", "booking_date"):
        df[col] = pd.to_datetime(df[col])

    # identical requests already seen on the same booking date (group blocks cancel a lot)
    key = [c for c in df.columns if c not in ("booking_id", "cancelled")]
    if counter is None:  # whole table at once
        order = df.sort_values(["booking_date", "booking_id"]).index
        df["same_request_count"] = df.loc[order].groupby(key, dropna=False, sort=False).cumcount().reindex(df.index)
    else:  # one booking at a time, in booking order
        df["same_request_count"] = [counter.see(row) for row in df[key].itertuples(index=False, name=None)]

    df["children"] = df["children"].fillna(0).clip(upper=3)
    df["babies"] = df["babies"].clip(upper=2)
    cats = categories(df)

    for col in TARGET_ENCODED:
        df[f"{col}_rate"] = cats[col].map(params["rates"][col]).fillna(params["prior"])
    for col in ("days_ahead", "nightly_rate", "waitlist_days", "past_cancellations", "past_stays", "same_request_count"):
        df[f"log_{col}"] = np.log1p(df[col].astype(float).clip(lower=0))
    month = df["arrival_date"].dt.month.astype(str)
    df["rate_rel"] = df["nightly_rate"] / (cats["hotel"] + "|" + month).map(params["rate_ref"])

    for col, allowed in params["categories"].items():  # rare / unseen values -> "other"
        df[col] = pd.Categorical(cats[col].where(cats[col].isin(allowed), "other"), categories=allowed)
    return df
