# Cleaning steps: what each step does

The code is in `prep.py`; it runs in `preprocessing.ipynb`, sections 3 (cleaning) and 4 (splits). The reasons behind each rule are in `data_preparation.md`.

| # | Step | What it does |
|---|---|---|
| 0 | `load_raw()` | Reads `train.csv`, turns the dates into real dates and makes number columns numeric. |
| 1 | `fit_params()` | Learns from the training data: which categories to keep (top 20 countries; agents, companies and room types with ≥ 50 rows), a rate cap per hotel (99.9th percentile), and a reference rate per hotel and arrival month. |
| 2 | `same_request_count` | Counts how many identical requests were already seen on the same booking date. This spots group blocks (10+ identical rows → 75–91% cancelled). |
| 3 | Fix values | Missing `children` → 0; caps `children` at 3 and `babies` at 2; `meal` Undefined → SC; segment/channel Undefined → Other; missing `country` → `(missing)`; blank `agent`/`company` → `none`; rare or unseen values → `other`. |
| 4 | Add features | `stay_value`, guest counts, arrival and booking calendar (month, weekday, day of year), flags (`is_prt`, `has_agent`, `has_company`, `is_zero_rate`), `rate_capped`, `rate_rel` (price ÷ normal price for that hotel and month). |
| 5 | `add_training_columns()` | Training only: marks rows to drop (13 placeholder rows + the €5,400 booking), sets `use_for_occupancy` (False for periods with incomplete data), sets `split` by arrival date (`train` < 2016-07-01 ≤ `valid` < 2016-10-01 ≤ `test`), and sets `is_commitment` (valid/test rows booked before their split started). |
| 6 | Checks | No missing features, the raw rate stays unchanged, unseen values are handled, and one-booking-at-a-time results equal whole-table results. |
| 7 | Save | `data/train_clean.parquet` (78,118 rows) and `data/prep_params.json` (the fitted parameters). |
| 8 | Splits | Fits the parameters on train + valid only (`data/prep_params_dev.json`) and writes `data/splits/{train,valid,test}.parquet`, one data source with both hotels in each file. **train** = Jul 2015 – Jun 2016 (1 year, 48,772 rows), **valid** = Jul – Sep 2016 (¼ year, 14,968 rows), **test** = Oct – Dec 2016 (¼ year, 14,378 rows). Load them with `prep.load_split("train")`. |

## Columns in the processed files

All 32 original columns are kept; 20 are added (52 in total). The cleaned columns keep their names (`children`, `babies`, and the categories `hotel`, `meal`, `country`, `sales_segment`, `sales_channel`, `room_type`, `deposit`, `guest_type`, `agent`, `company`).

| Added column | What it is | Why |
|---|---|---|
| `same_request_count` | Identical requests already seen on the same booking date (0 = first one) | Spots group blocks: from 25% cancelled (first request) up to 91% (50+ before it). Counted only "so far", like `decide()` sees it |
| `stay_value` | `nightly_rate × nights` | What the guest pays if seated; a walk costs 2.5× this. **Not a feature**, it's for the accept/reject rule |
| `total_guests` | adults + children + babies | Group size in one number |
| `no_guests` | 1 if the booking has 0 guests | 77 odd bookings that behave differently |
| `has_kids` | 1 if children or babies | Family stays cancel differently |
| `arr_month`, `arr_dow`, `arr_doy` | Month, weekday, day of year of arrival | Season and weekday without the year (2017 isn't in the training data) |
| `is_weekend_arrival` | 1 if arriving Fri/Sat | Weekend trips vs. business stays |
| `book_month`, `book_dow` | Month and weekday of the booking | When people book says something about the booking type |
| `is_prt` | 1 if country is Portugal | Strongest country signal (57% vs 20% cancelled) |
| `has_agent`, `has_company` | 1 if booked via an agent / company | Booking channel in one flag |
| `is_zero_rate` | 1 if the rate is €0 | Earns nothing, so there is nothing to gain by accepting it |
| `rate_capped` | `nightly_rate` capped at the hotel's 99.9th percentile | Price feature without typos like €5,400 (`nightly_rate` itself stays raw for revenue) |
| `rate_rel` | Rate ÷ normal rate of that hotel and month | Expensive or cheap for its season; robust to the yearly price increase |
| `use_for_occupancy` | False for arrivals before 2015-07-15 and Downtown before 2015-09-01 | Leave these out when estimating occupancy (incomplete data). **Not a feature** |
| `split` | `train` / `valid` / `test` | Which part of the split the row belongs to. **Not a feature** |
| `is_commitment` | valid/test rows booked before their split started | In a replay these are the confirmed bookings, the rest are requests. **Not a feature** |

## Splits: how to use them

- Develop on `train` → `valid` (as often as you like).
- Look at `test` rarely: retrain on `train` + `valid`, then evaluate on `test`, including the replay (`is_commitment` = confirmed bookings, the rest = requests).
- Submission: retrain on all data with `data/prep_params.json`.
- One model per hotel: `prep.load_split("train", hotel="downtown")` (or `"seaside"`) with the features `prep.FEATURES_PER_HOTEL` (same as `prep.FEATURES`, but without the constant `hotel` column). Compare on `valid` against one joint model that uses `prep.FEATURES`.

## Important

- `Solution.decide()` must use the same code: `prep.prepare(booking, params, counter=counter)[prep.FEATURES]`.
- `booking_id` and absolute years/dates are deliberately **not** features.
- `nightly_rate` is never changed, because revenue is calculated on it.
