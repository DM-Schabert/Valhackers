# Cleaning steps: what each step does

The code is in `prep.py`; it runs in `preprocessing.ipynb`, section 3. The reasons behind each rule are in `data_preparation.md`.

| # | Step | What it does |
|---|---|---|
| 0 | `load_raw()` | Reads `train.csv`, turns the dates into real dates and makes number columns numeric. |
| 1 | `fit_params()` | Learns from the training data: which categories to keep (top 20 countries; agents, companies and room types with ≥ 50 rows), a rate cap per hotel (99.9th percentile), and a reference rate per hotel and arrival month. |
| 2 | `same_request_count` | Counts how many identical requests were already seen on the same booking date. This spots group blocks (10+ identical rows → 75–91% cancelled). |
| 3 | Fix values | Missing `children` → 0; caps `children` at 3 and `babies` at 2; `meal` Undefined → SC; segment/channel Undefined → Other; missing `country` → `(missing)`; blank `agent`/`company` → `none`; rare or unseen values → `other`. |
| 4 | Add features | `stay_value`, guest counts, arrival and booking calendar (month, weekday, day of year), flags (`is_prt`, `has_agent`, `has_company`, `is_zero_rate`), `rate_capped`, `rate_rel` (price ÷ normal price for that hotel and month). |
| 5 | `add_training_columns()` | Training only: marks rows to drop (13 placeholder rows + the €5,400 booking), sets `use_for_occupancy` (False for periods with incomplete data), and sets `split` (`train` < 2016-07-01 ≤ `valid`). |
| 6 | Checks | No missing features, the raw rate stays unchanged, unseen values are handled, and one-booking-at-a-time results equal whole-table results. |
| 7 | Save | `data/train_clean.parquet` (78,118 rows) and `data/prep_params.json` (the fitted parameters). |

**Important**
- `Solution.decide()` must use the same code: `prep.prepare(booking, params, counter=counter)[prep.FEATURES]`.
- `booking_id` and absolute years/dates are deliberately **not** features.
- `nightly_rate` is never changed, because revenue is calculated on it.
