# Brainstorm

## Data quality report: `train-template-kit/train.csv`

### TL;DR

- The data is mechanically clean. The real problems are **leakage** (`booking_id`, plus a few fields recorded after the booking was made), **31% duplicate rows**, and a few outliers.
- **Capacity is very tight.** The game's capacity is only 55% of each hotel's historical peak, so the task is mainly **choosing which bookings to accept**, not just predicting cancellations.
- **Don't download the public Kaggle dataset.** It includes the 2017 data, and reading that counts as cheating.

### Overview

- **78,132 booking requests**, 32 columns.
- Arrivals 2015-07-01 → 2016-12-31; bookings made 2014-03-18 → 2016-12-31.
- Downtown 51,575 rows (41.5% cancelled); Seaside 26,557 rows (26.6% cancelled). **Overall 36.4% cancelled.**
- Source: the public "Hotel Booking Demand" dataset (Antonio et al., 2019), with renamed columns. City Hotel became Downtown and Resort Hotel became Seaside.
  - `booking_id` is the row number in the original file.
  - Removed compared with the original: 2017 arrivals, zero-night stays, and columns that would give away the outcome (`reservation_status`, assigned room type, parking).
- `train.csv` matches the size listed inside `train-template-kit.7z`.
- I did **not** open `evaluation-kit.7z`. It contains `evaluate.py` and `test.csv` (the 2017 data).

### Sanity checks that pass (100%)

- `days_ahead` = `arrival_date − booking_date`
- `arrival_date` = `arrival_year` / `arrival_month` / `arrival_day`
- `nights` = `weekend_nights + weekday_nights`
- No zero-night stays, no negative rates, no arrival before booking, no bookings made after 2016-12-31.
- Every day has arrivals at both hotels (no gaps).

---

### 1. `booking_id` leaks the target

The IDs are row numbers from the original file, which was assembled from several extracts. Some blocks are almost all cancellations; others are almost all guests who showed up.

| Hotel | ID range | Share cancelled |
|---|---|---|
| Seaside | 2 – 9,774 (one chronological block) | 63% |
| Seaside | 9k – 10k | 99% |
| Seaside | ≥ 20,000 | ~0% |
| Downtown | 40,060 – 62,063 (one chronological block) | 74% |
| Downtown | 52k – 62k and 73k – 76k | 92 – 100% |
| Downtown | ≥ 83,000 | ~0% |

The other 45k rows sit in small scattered runs and are only 10.5% cancelled.

How well the ID alone predicts cancellation, compared with lead time (AUC):

| Hotel | `booking_id` | ID rank within the arrival day | `days_ahead` |
|---|---|---|---|
| Downtown | 0.819 | 0.815 | 0.713 |
| Seaside | 0.845 | 0.845 | 0.676 |

Within one arrival day, the lowest 20% of IDs are 73% cancelled and the highest 20% are 6% cancelled.

`decide()` receives `booking_id`. If the 2017 IDs follow the same numbering, a model would get this signal for free. **Proposal: don't use it as a feature, and ask the organisers whether it's allowed.** Also note that gaps in the training IDs (e.g. Seaside 9,775 – 13,802) show where the 2017 rows sit.

### 2. Capacity is very tight

Game capacity is Downtown **124** and Seaside **103**, which is 55% of each hotel's historical peak (226 × 0.55 ≈ 124, 187 × 0.55 ≈ 103). Rooms occupied per night, counting only guests who showed up, Jul 2015 – Dec 2016:

| Hotel | Capacity | Peak | Mean | 95th percentile | Nights over capacity (of 550) | Room-nights over capacity |
|---|---|---|---|---|---|---|
| Downtown | 124 | 226 | 158 | 224 | 392 | 27,335 |
| Seaside | 103 | 187 | 147 | 183 | 436 | 27,835 |

Counting every request (cancelled or not), the peaks are 607 and 356.

- From March to October 2016, **almost every night is over capacity** at both hotels, even counting only guests who showed up.
- The quiet months are Nov–Jan. For example, Seaside has only 2 over-capacity nights in Jan 2016.
- What this means for the game:
  - Accepting a booking that later cancels costs nothing.
  - Accepting a guest who shows up on a full night means someone gets walked, at 2.5 × their stay value.
  - "A seated guest keeps the room": early bookings with low value can block later ones worth more.
- Stay value (rate × nights) of guests who showed up: p10 70, **median 243**, p90 669, p99 1,494.

### 3. 31% of rows are duplicates

- **24,066 rows** are identical to another row except for `booking_id`; 29,606 rows belong to such groups.
- Groups go up to 180 identical rows. 740 groups have ≥ 10 rows, covering 16,458 rows.
- Mostly `Non Refund` (10,333 of 10,461 non-refundable rows are duplicates) and Transient / Transient-Party guests.
- Duplicates are 57% cancelled, against 24% for the rest.
- They're probably **real multi-room group bookings**, so keep them for occupancy.
- **For validation, split by time**, not random folds; otherwise copies of the same group land on both sides and inflate the score.

### 4. Fields recorded after the booking was made

These fields hold their final values, not their values when the booking came in. The 2017 data should behave the same way, so they will still score well. Just be aware they are hindsight.

| Feature | Share cancelled |
|---|---|
| `amendments` = 0 / > 0 | 40% / **15%** |
| `special_requests` = 0 / ≥ 1 | 47% / ~18% |
| `waitlist_days` > 0 (3,497 rows) | 64% (vs 35%) |
| `country` = PRT / other / missing | **57%** / 20% / 13% |
| `deposit` = Non Refund (10,461 rows) | **99.2%** |

- Portugal is 70% of cancelled bookings but only 30% of guests who showed up. A likely reason is that nationality often isn't confirmed until check-in, so cancelled bookings default to Portugal.
- `Non Refund` is the well-known oddity of this dataset: 96% are from Portugal, and almost all come through Groups (6,395) or Offline TA/TO (3,856). Under the game's rules, a cancelled non-refundable booking earns **0**.

### 5. Other drivers of cancellation

| Lead time (days) | 0 | 1–7 | 8–30 | 31–90 | 91–180 | 181–365 | 365+ |
|---|---|---|---|---|---|---|---|
| Share cancelled | 7% | 12% | 27% | 36% | 44% | 58% | 78% |

- Segment: Groups 62%, Offline TA/TO 35%, Online TA 34%, Aviation 23%, Corporate 16%, Direct 15%, Complementary 11%.
- Guest type: Transient 40%, Contract 35%, Transient-Party 28%, Group 12%.
- Returning guests 22% vs new guests 37%.
- `past_cancellations` > 0 (6,201 rows): **95%** cancelled.
- Arrival year: 2015 37%, 2016 36%.

### 6. Outliers and bad values

- **Rates:**
  - One booking at **€5,400/night** (Downtown 2016-03-25, cancelled); the next highest is €508.
  - **933 rows at €0**, 796 of which showed up. By segment: Complementary 462, Groups 186, Offline TA/TO 162, others 123. They earn nothing, and walking them also costs nothing (2.5 × 0).
  - Rate 99th percentile: Downtown €208, Seaside €259.
- **Guest counts:**
  - 13 placeholder rows with **26–55 adults** (Seaside Sep–Oct 2015, rate 0, all cancelled).
  - 237 rows with 0 adults; 77 rows with no guests at all.
  - `children` = 10 (1 row) and `babies` = 9 or 10 (2 rows), likely typos.
- **Missing values:**
  - `children`: 4 rows (Downtown Aug 2015, all cancelled).
  - `country`: 395 rows, plus one two-letter code (`CN`) among the three-letter codes.
  - `agent` (10,513) and `company` (73,715) are blank when there is none. They are ID numbers, so treat them as categories, not quantities.
- **"Undefined" values:**
  - `meal` "Undefined": 754 rows, all Seaside. The source documentation treats it as SC (no meal).
  - `sales_segment` "Undefined": 2 rows; `sales_channel` "Undefined": 5 rows.
- **Long stays:** 259 longer than 14 nights, 14 longer than 30 nights, max 69 nights.
- **Room types differ by hotel:** H and L exist only at Seaside; B and P are basically Downtown only; `P` appears once.

### 7. Columns that follow a different convention

- `arrival_week` uses **weeks starting on Sunday, with week 1 containing Jan 1** (100% match), **not ISO weeks** (25% match). Don't recompute it with `isocalendar()`.
- `weekend_nights` matches Sat/Sun nights counted from **`arrival_date − 1 day`** (100%), but from `arrival_date` itself only 50% of the time. This off-by-one comes from the source. Trust `arrival_date`, because the evaluator's capacity is indexed by date.

### 8. Coverage and drift over time

| Month | Downtown requests | Downtown cancelled | Downtown median rate | Seaside requests | Seaside cancelled | Seaside median rate |
|---|---|---|---|---|---|---|
| 2015-07 | 1,395 | **67%** | 63 | 1,361 | 24% | 123 |
| 2015-08 | 2,477 | 50% | 75 | 1,392 | 26% | 154 |
| 2015-09 | 3,521 | 44% | 95 | 1,582 | 35% | 80 |
| 2015-10 | 3,371 | 39% | 89 | 1,549 | 27% | 53 |
| 2015-11 | 1,226 | 25% | 72 | 1,089 | 17% | 42 |
| 2015-12 | 1,638 | 41% | 78 | 1,241 | 24% | 55 |
| 2016-01 | 1,355 | 32% | 78 | 866 | 14% | 45 |
| 2016-02 | 2,353 | 39% | 79 | 1,509 | 27% | 53 |
| 2016-03 | 3,031 | 37% | 84 | 1,754 | 21% | 54 |
| 2016-04 | 3,552 | 43% | 96 | 1,848 | 28% | 68 |
| 2016-05 | 3,643 | 39% | 110 | 1,773 | 27% | 69 |
| 2016-06 | 3,915 | 44% | 110 | 1,345 | 28% | 96 |
| 2016-07 | 3,117 | 33% | 107 | 1,427 | 32% | 159 |
| 2016-08 | 3,370 | 37% | 115 | 1,672 | 35% | 198 |
| 2016-09 | 3,864 | 41% | 116 | 1,518 | 30% | 92 |
| 2016-10 | 4,192 | 46% | 108 | 1,962 | 29% | 62 |
| 2016-11 | 3,093 | 44% | 89 | 1,315 | 21% | 48 |
| 2016-12 | 2,462 | 43% | 88 | 1,354 | 24% | 61 |

- **Downtown July–August 2015 looks incomplete.** July 2015 averages 34 occupied rooms, against 213 in July 2016, with 67% cancelled and a median lead time of 257 days.
- Guests who arrived before 2015-07-01 and were still staying are missing.
- Seaside is a summer-peak hotel: median rate about €160–200 in Jul–Aug against €45–55 in winter. Downtown is flatter.
- Prices are rising. July–December median rate: Downtown €80 → €105 (+31%), Seaside €76 → €85 (+12%). **Weight 2016 more and expect higher rates in 2017.**
- The public dataset ends in August 2017, and `test.csv` (6.1 MB) is about half the size of `train.csv`. **The 2017 evaluation likely covers only January–August.**

---

### What to do with this

1. Leave out `booking_id` (unless the organisers say it's allowed).
2. Keep duplicate rows when counting occupancy; validate with a **time-based split**.
3. Treat `agent`, `company` and `country` as categories; map `meal` "Undefined" to SC; cap or drop the €5,400 rate.
4. Build a **local replay of 2016**: 2016 arrivals booked before 2016-01-01 act as `commitments`, capacity is 124 / 103 minus 2015 guests still staying, and the rest go to `decide()` in booking-date order. Score it with the same revenue and walk-penalty rules.
5. Strategy: estimate each booking's chance of showing up, and accept it if its expected value beats the expected walk cost for the nights it covers. Capacity matters most from March to October.











