"""Minimal preprocessing: only what the model needs to run.

    python minimal_preprocessing/preprocess.py

Reads train-template-kit/train.csv and writes data/splits_minimal/{train,valid,test}.parquet:
1. dates as dates
2. text and ID columns as categories (LightGBM can't use text; agent/company IDs are no quantities)
3. split by arrival date (same dates as prep.py)
Missing values stay missing (LightGBM handles them). booking_id is dropped: it leaks the target.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "train-template-kit" / "train.csv"
OUT = ROOT / "data" / "splits_minimal"
VALID_START, TEST_START = pd.Timestamp("2016-07-01"), pd.Timestamp("2016-10-01")
CATEGORIES = ["hotel", "arrival_month", "meal", "country", "sales_segment", "sales_channel",
              "room_type", "deposit", "guest_type", "agent", "company"]

df = pd.read_csv(SOURCE, parse_dates=["arrival_date", "booking_date"]).drop(columns="booking_id")
df[["agent", "company"]] = df[["agent", "company"]].astype("Int64").astype("string")  # IDs as text, else parquet reads them back as numbers
df[CATEGORIES] = df[CATEGORIES].astype("category")

OUT.mkdir(parents=True, exist_ok=True)
df[df["arrival_date"] < VALID_START].to_parquet(OUT / "train.parquet", index=False)
df[(df["arrival_date"] >= VALID_START) & (df["arrival_date"] < TEST_START)].to_parquet(OUT / "valid.parquet", index=False)
df[df["arrival_date"] >= TEST_START].to_parquet(OUT / "test.parquet", index=False)
print(f"written to {OUT}")
