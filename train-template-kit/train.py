"""Train the models on already-preprocessed rows: one over everything, one per hotel.

MODEL switches the model type. Every model gets the same rows and features.
"""
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL = "logreg"  # "lightgbm" | "catboost" | "xgboost" | "logreg"


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


    
    if MODEL == "logreg":  # one-hot categories, fill gaps in numbers, scale
        numbers = [c for c in X.columns if c not in cats]
        columns = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers))
        return make_pipeline(columns, LogisticRegression(max_iter=2000))
    raise ValueError(f"unknown MODEL {MODEL!r}")





def fit(rows, features):
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
