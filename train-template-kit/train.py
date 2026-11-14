"""Train the models on already-preprocessed rows: one over everything, one per hotel.

MODEL switches the model type. Every model gets the same rows and features.
CALIBRATION_DAYS > 0: the model fits on all but the most recent arrivals, and an isotonic curve fitted on
those held-out recent arrivals corrects its probabilities (dev run: train -> fit, valid -> calibrate).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL = "logreg"  # "lightgbm" | "catboost" | "xgboost" | "logreg" | "mlp"
CALIBRATION_DAYS = 92  # last ~3 months of arrivals calibrate the model; 0 = no calibration


def make_model(X):
    cats = [c for c in X.columns if X[c].dtype == "category"]
    if MODEL == "lightgbm":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              random_state=0, verbose=-1)
    if MODEL == "xgboost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, enable_categorical=True,
                             tree_method="hist", random_state=0)
    if MODEL == "catboost":  # categories as text, gaps become "nan"
        as_text = FunctionTransformer(lambda X: X.astype({c: str for c in cats}))
        return make_pipeline(as_text, CatBoostClassifier(iterations=500, learning_rate=0.05, cat_features=cats,
                                                         random_seed=0, verbose=0))


    
    if MODEL in ("logreg", "mlp"):  # one-hot categories, fill gaps in numbers, scale
        numbers = [c for c in X.columns if c not in cats]
        columns = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers))
        if MODEL == "logreg":
            return make_pipeline(columns, LogisticRegression(max_iter=2000))
        # small neural network: 2 hidden layers, L2 penalty, stops when a 10% hold-out stops improving
        return make_pipeline(columns, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                                                    batch_size=256, early_stopping=True, validation_fraction=0.1,
                                                    n_iter_no_change=10, max_iter=200, random_state=0))
    raise ValueError(f"unknown MODEL {MODEL!r}")





class Calibrated:
    """Model + isotonic curve: P(cancel) = iso(model's P(cancel)). Same predict_proba interface."""

    def __init__(self, model, iso):
        self.model, self.iso = model, iso

    def predict_proba(self, X):
        c = self.iso.predict(self.model.predict_proba(X)[:, 1])
        return np.c_[1 - c, c]


def fit(rows, features):
    if CALIBRATION_DAYS:
        recent = (rows["arrival_date"] > rows["arrival_date"].max() - pd.Timedelta(days=CALIBRATION_DAYS)).to_numpy()
        early, late = rows[~recent], rows[recent]
        model = make_model(early[features]).fit(early[features], early["cancelled"])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(model.predict_proba(late[features])[:, 1], late["cancelled"])
        model = Calibrated(model, iso)
    else:
        X = rows[features]
        model = make_model(X).fit(X, rows["cancelled"])
    model.used_features = features  # solution.py predicts with exactly these columns
    return model


def train(rows, features):
    """{"all": model over both hotels, "Downtown Hotel": city only, "Seaside Hotel": Seaside only}"""
    models = {"all": fit(rows, features)}
    for hotel in ("Downtown Hotel", "Seaside Hotel"):
        models[hotel] = fit(rows[rows["hotel"] == hotel], features)
    return models
