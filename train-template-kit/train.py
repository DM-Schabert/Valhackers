"""Train the model on already-preprocessed rows (data/splits/*.parquet)."""
from lightgbm import LGBMClassifier


def train(rows, features):
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                           random_state=0, verbose=-1)
    model.fit(rows[features], rows["cancelled"])
    return model
