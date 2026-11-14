"""PCA preprocessing from test_preprocessing/PCA.ipynb, exported with our time split.

    python test_preprocessing/pca_preprocess.py

Same steps as the notebook (median + scale numbers, most-frequent + one-hot categories, PCA 95%),
but learned on TRAIN only and then applied to valid and test.
Writes data/splits_pca/{train,valid,test}.parquet (PC columns + the columns the replay needs) and
data/splits_pca/features.json (the PC list). In solution.py:
    SPLITS = DATA / "splits_pca"
    PARAMS_FILE = SPLITS / "features.json"
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "splits_pca"
VALID_START, TEST_START = pd.Timestamp("2016-07-01"), pd.Timestamp("2016-10-01")
KEEP = ["hotel", "arrival_date", "booking_date", "nights", "nightly_rate", "cancelled"]  # needed by the replay

df = pd.read_csv(ROOT / "train-template-kit" / "train.csv", parse_dates=["arrival_date", "booking_date"])
df["Profit"] = df["nights"] * df["nightly_rate"]
X = df.drop(columns=["booking_id", "cancelled", "arrival_date", "booking_date", "company", "arrival_year"])
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

pca = Pipeline([
    ("prepare", ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])),
    ("pca", PCA(n_components=0.95)),
])

parts = {
    "train": df["arrival_date"] < VALID_START,
    "valid": (df["arrival_date"] >= VALID_START) & (df["arrival_date"] < TEST_START),
    "test": df["arrival_date"] >= TEST_START,
}
pca.fit(X[parts["train"]])  # learned from the train period only
features = [f"PC{i + 1}" for i in range(pca.named_steps["pca"].n_components_)]

OUT.mkdir(parents=True, exist_ok=True)
for name, rows in parts.items():
    keep = df.loc[rows, KEEP].reset_index(drop=True).astype({"hotel": "category"})
    pcs = pd.DataFrame(pca.transform(X[rows]), columns=features)
    pd.concat([keep, pcs], axis=1).to_parquet(OUT / f"{name}.parquet", index=False)
(OUT / "features.json").write_text(json.dumps({"features": features}))
print(f"{len(features)} components -> {OUT}")
