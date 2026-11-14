# How to use this repo

Use a Python environment with `pandas`, `pyarrow` and `lightgbm`, for example `jupyter_env`.

## 1. Preprocess

Run `preprocessing.ipynb` (sections 3 and 4). It writes:

- `data/splits/train.parquet`, `valid.parquet` and `test.parquet`
- `data/prep_params_dev.json`: the feature list, plus what `decide()` needs for raw bookings

## 2. Net profit on test

At the top of `train-template-kit/solution.py`, set the input files:

```python
TRAIN_FILE = DATA / "splits" / "train.parquet"
VALID_FILE = DATA / "splits" / "valid.parquet"
TEST_FILE = DATA / "splits" / "test.parquet"
PARAMS_FILE = DATA / "prep_params_dev.json"   # optional
```

Then run:

```bash
python train-template-kit/solution.py
```

This happens:

- The model trains on train + valid (`train-template-kit/train.py`).
- `TEST_FILE` is replayed:
  - Bookings made before the first arrival in the file are confirmed commitments.
  - All other bookings are requests, decided in booking order.
- It prints the net profit. Rooms are full capacity (124 / 103) every night.

The files can be `.parquet` or `.csv`. Without `PARAMS_FILE`, the features are every numeric or category column except labels, IDs and dates.

## Submission

`evaluate(Solution)`: the `Solution` trains on all three files and uses `PARAMS_FILE` to preprocess the raw 2017 bookings.
