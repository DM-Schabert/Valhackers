from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / 'train-template-kit' / 'train.csv'
DATE_COLS = ['arrival_date', 'booking_date']
TARGET = 'cancelled'
DROP_COLS = ['booking_id', 'arrival_date', 'booking_date', 'company', 'arrival_year']
ID_COLS = ['agent']
VALID_START = pd.Timestamp('2016-07-01')
TEST_START = pd.Timestamp('2016-10-01')


def load_data(path=DATA_PATH):
    return pd.read_csv(path, parse_dates=DATE_COLS)


def time_split(df):
    train = df[df['arrival_date'] < VALID_START]
    valid = df[(df['arrival_date'] >= VALID_START) & (df['arrival_date'] < TEST_START)]
    test = df[df['arrival_date'] >= TEST_START]
    return train, valid, test


def make_features(df):
    X = df.copy()
    X['Profit'] = X['nights'] * X['nightly_rate']
    return X.drop(columns=DROP_COLS + [TARGET], errors='ignore')


def split_xy(df):
    return make_features(df), df[TARGET]


def build_linear_preprocessor(X):
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(exclude='number').columns

    numeric_steps = Pipeline(
        [
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]
    )

    categorical_steps = Pipeline(
        [
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ('num', numeric_steps, num_cols),
            ('cat', categorical_steps, cat_cols),
        ]
    )
    preprocess.set_output(transform='pandas')
    return preprocess


def to_lightgbm(X):
    X = X.copy()
    cat_cols = list(X.select_dtypes(exclude='number').columns) + ID_COLS
    X[cat_cols] = X[cat_cols].astype('category')
    return X
