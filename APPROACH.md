# Hotel booking acceptance: our approach

From the raw data to the search on the HPC: what we did, what we found, and why we decided each thing the way we did.

- **Sections 1–10: round one**, with one validation window. All numbers come from `grid_search/compare.py` and `grid_search/grid_search.py`.
- **Sections 11–14: round two**, built to run on the HPC in under an hour. It fixes what round one got wrong. It ran on DTU's cluster in 14 minutes; the results are in section 11.8 and `hpc/results/RESULTS.md`.
- **Section 13** lists every major decision with its reason. [Reproduce](#reproduce) is at the end.

---

## 1. The task

The hotel gets more reservations than it has rooms, and many guests cancel or don't show up. We build the system that decides, **for each booking request, whether to accept it**.

- Two hotels: **Downtown (124 rooms)** and **Seaside (103 rooms)**.
- `evaluate(Solution)` replays every 2017 booking request in booking order. Our `decide(booking)` returns accept or reject.
- A guest who shows up pays `nightly_rate × nights`. A cancellation costs nothing.
- If a hotel is full at check-in, the cheapest arriving guests are **walked**: they pay nothing and cost **2.5 × their stay value**.
- **Score = revenue of seated guests − walk penalties.**

So the job isn't just "predict cancellations". It's **choosing which bookings to accept so that the rooms fill with paying guests without walking anyone**.

---

## 2. The data and what we found

`train.csv`: **78,132 booking requests**, 32 columns, arrivals **Jul 2015 – Dec 2016**. It's the public "Hotel Booking Demand" dataset with renamed columns. Full analysis: `docs/brainstorm.md`.

| Finding | Why it matters |
|---|---|
| **36.4% cancelled** overall: Downtown 41.5%, Seaside 26.6% | Cancellations are common, so the hotels can safely accept more bookings than they have rooms |
| **Capacity is only 55% of each hotel's historical peak** (124 of 226, 103 of 187). From March to October 2016, almost every night is over capacity | This is mainly a **selection** problem: which bookings deserve the scarce rooms |
| **`booking_id` leaks the target** (AUC 0.82–0.85 on its own) | It's a row number from the source file. We leave it out (worth asking the organisers) |
| **31% of rows are duplicates**: group blocks of up to 180 identical requests, 57% cancelled vs 24% | Counted online as `same_request_count`. It's also why we **split by time, never randomly** |
| **Portugal is 45% of requests and 57% cancelled**, but this is partly a recording artefact (nationality is often only set at check-in). PRT's share among cancellations falls from 93% to 54% over time | The `country` feature is strong in old data but less reliable later. See the results below: dropping it helps on validation |
| Some fields are **recorded after booking time** (`amendments`, `special_requests`, `waitlist_days`, `country`, `deposit`) | They are in the 2017 rows too, so they can be used, but they are hindsight |
| `Non Refund` deposits are **99.2% cancelled** | A well-known oddity of this dataset |
| Bad rows: 13 placeholder bookings with 26–55 adults, one €5,400 rate, 933 bookings at €0 | Removed from training or handled with flags |
| **Drift over time**: prices up 12–31% from 2015 to 2016; Downtown Jul–Aug 2015 looks incomplete | Use relative prices and calendar features without the year. Weight recent data more |

---

## 3. The whole pipeline at a glance

```
train.csv
   │
   ▼
prep.fit_params()        learn categories, price caps, reference prices   (training data only)
   │
   ▼
prep.prepare()           same_request_count → clean values → add features
   │                     (identical code in training and in decide(), one booking at a time)
   ▼
time split               train: Jul 2015–Jun 2016 │ valid: Jul–Sep 2016 │ test: Oct–Dec 2016
   │
   ▼
model                    P(cancel) for every request   (logistic regression / LightGBM / ...)
   │
   ▼
decision rule            accept if the expected guests still fit into the free rooms
   │
   ▼
replay + net_profit()    same rules as the real evaluation → net profit in €
```

---

## 4. Preprocessing

### 4.1 The pipeline we use (`claude-preprocessing/prep.py`)

The key design choice: **one module (`prep.py`) is used for training *and* inside `Solution.decide()`**. In 2017, bookings arrive one at a time and must be cleaned exactly like the training data. The fitted settings are saved as JSON (`prep_params.json`), and `decide()` loads them.

| Step | What it does |
|---|---|
| 1. `fit_params()` | Learned from training data only: keep the top 20 countries and agents/companies/room types with ≥ 50 rows (the rest become `other`), a price cap per hotel (99.9th percentile), and a reference price per hotel and month |
| 2. `same_request_count` | How many identical requests were already seen on the same booking day. Group blocks with 10+ copies cancel 75–91% of the time. Counted "so far", exactly as `decide()` sees it |
| 3. Clean values | Missing children → 0, cap impossible counts, `Undefined` → SC / Other, missing country → `(missing)`, `agent`/`company` are IDs so they become categories (`none` when blank) |
| 4. Add features | Total guests, has kids, calendar without the year (arrival month, weekday, day of year, weekend arrival, booking month/weekday), flags (`is_prt`, `has_agent`, `has_company`, `is_zero_rate`), `rate_capped`, and `rate_rel` (price ÷ normal price for that hotel and month, robust to price increases) |
| 5. Training only | Drop the 14 broken rows, then split by arrival date |

Feature-set **variants** (in `prep.py`): `baseline` (all features), `no_country`, `no_hindsight` (only what is known at booking time) and `no_season`.

### 4.2 Preprocessing approaches we tried

Same model for all of them (logistic regression with the `train.py` settings), trained on **train** and replayed on **valid**:

| Preprocessing | Net profit (valid) | AUC | Predicted / actual cancel rate |
|---|---|---|---|
| Minimal: raw columns, IDs as categories (`minimal_preprocessing/`) | 361,526 | 0.852 | 28.0% / 35.7% |
| PCA, 31 components covering 95% of the variance (`test_preprocessing/`) | 361,207 | 0.819 | 30.8% / 35.7% |
| Full pipeline (`claude-preprocessing/`) | 374,454 | 0.850 | 30.8% / 35.7% |
| **Full pipeline without `country`** | **384,723** | 0.824 | **34.1%** / 35.7% |

- **PCA doesn't help.** The scree plot has no clear elbow. The first 16 components are almost entirely the 18 numeric columns, and all category information (hotel, deposit, segment, country) is squeezed into the last components, with ≤ 1.7% of the variance each. PCA keeps *variance*, not *what predicts cancellation*.
- **The full pipeline beats the minimal one by about 13,000**, and **dropping `country` adds another 10,000**. Note that AUC goes *down* while profit goes *up*. The reason is in the next section.

---

## 5. How we evaluate: the replay

A random train/test split would put copies of the same group booking on both sides, and the data changes over time. So:

- **Time split:** train = Jul 2015 – Jun 2016, valid = Jul – Sep 2016, test = Oct – Dec 2016.
- **Replay** (same code as `solution.py`): bookings made before the period starts are already confirmed (commitments). All other requests go through `decide()` in booking order. Capacity = rooms minus guests from before the period who are still staying. The score is the net profit, computed with the real rules.
- **Test is used rarely**, only for final checks. Settings are chosen on valid.

**Reference strategies**, to put the model numbers in context:

| Strategy | Valid (Jul–Sep 2016) | Test (Oct–Dec 2016) |
|---|---|---|
| Reject every request (only the commitments) | 307,006 | 483,841 |
| Accept every request | **−1,729,306** | −22,691 |
| Hindsight: knows who shows up, fills rooms with the highest rates first* | 415,957 | 929,558 |

\* A greedy heuristic, so an approximate ceiling rather than the exact optimum.

Accepting everything is a disaster in peak season, because walk penalties dominate. Our best models (below) capture **about 60–70% of the gap** between "reject everything" and the hindsight ceiling. LightGBM with default settings captures only about 20% on valid.

---

## 6. The decision rule (`Solution.accept` in `solution.py`)

1. The model gives `P(cancel)` → `P(show) = 1 − P(cancel)`.
2. For every night of the stay: **expected guests already accepted + P(show) ≤ free rooms**.
3. Accept only if that holds on every night and the rate is above €0.

What this means:
- **Calibration matters more than ranking.** The rule *adds up* probabilities, so they must be right in absolute terms. If the model predicts too few cancellations, the hotel looks full too early and good bookings get rejected.
- **Every model we trained predicts too few cancellations** (about 18–35% predicted vs 36–39% actual), so they all reject more than they should.
- The rule **ignores value**: a €40 booking and a €400 booking are treated the same, first come, first served.

---

## 7. Models tried

All with the `train.py` settings, full pipeline, all features, train → valid:

| Model | One model per hotel? | Net profit (valid) | Accepted | AUC | Predicted cancel rate (actual 35.7%) |
|---|---|---|---|---|---|
| **Logistic regression** | no | 374,454 | 491 | 0.850 | 30.8% |
| **Logistic regression** | yes | **379,198** | 531 | 0.833 | 30.8% |
| LightGBM | no | 328,688 | 200 | 0.844 | 20.6% |
| LightGBM | yes | 330,784 | 217 | 0.842 | 22.8% |
| XGBoost | no | 331,313 | 247 | **0.862** | 21.5% |
| XGBoost | yes | 337,988 | 292 | 0.850 | 24.3% |
| CatBoost | no | 328,578 | 203 | 0.860 | 20.8% |
| CatBoost | yes | 336,769 | 280 | 0.855 | 24.1% |

**Why logistic regression wins even though the tree models rank better (higher AUC):** the tree models are badly calibrated. They predict about 21% cancellations instead of 36%, so they think the hotels are full and accept only 200–290 bookings instead of about 500. Much of this comes from `country`: the trees latch onto the Portugal artefact, which is weaker in the later period. Without `country`, LightGBM's predicted cancel rate rises from about 20% to 27% (section 8).

---

## 8. Grid search (`grid_search/`)

Each combination is trained on train and replayed on valid. Results go to `grid_search/results/<model>/*.json`.

| Model | Grid | Done |
|---|---|---|
| Logistic regression | `C` (0.01–3) × `min_frequency` (5, 20, 50) × 4 feature sets × one or two models | **144 / 144** |
| LightGBM | trees (200, 500, 1000) × learning rate × leaves × `min_child_samples` × 4 feature sets | **123 / 324** (stopped early; the 500/1000-tree settings are mostly still missing) |

**Results on valid:**

| | Best net profit | Best settings |
|---|---|---|
| Logistic regression | **386,289** | `C=0.3`, `min_frequency=50`, `no_country`, one model per hotel |
| LightGBM (partial) | 356,253 | 200 trees, learning rate 0.05, 15 leaves, `min_child_samples=50`, `no_country` |

Average net profit per feature set:

| Feature set | Logistic regression | LightGBM (partial) |
|---|---|---|
| baseline | 377,181 | 330,037 |
| **no_country** | **384,420** | **349,541** |
| no_hindsight | 366,947 | 349,354 |
| no_season | 375,439 | 329,484 |

- **The feature set matters far more than the hyperparameters.** For logistic regression, `C` and `min_frequency` move the result by about ±1,000 (noise), while dropping `country` adds about 7,000 on average. LightGBM gains about 20,000 from it.
- For LightGBM, dropping `country` or the hindsight features fixes most of the calibration problem.

---

## 9. Final check on test (trained on train + valid)

| Model | Net profit (test) | Accepted | AUC | Predicted cancel rate (actual 38.6%) |
|---|---|---|---|---|
| Logistic regression, `train.py` settings (current submission) | 767,179 | 2,056 / 5,457 | 0.874 | 28.1% |
| Logistic regression, best on valid | 742,450 | 1,993 / 5,457 | 0.832 | 26.7% |
| **LightGBM, best on valid** | **775,466** | 2,167 / 5,457 | 0.847 | 27.5% |

**The most important lesson so far:** the logistic regression that won on valid **lost 25,000 on test**. One three-month window (Jul–Sep, peak season) is too narrow to choose settings reliably, and test (Oct–Dec) behaves differently. The 2017 evaluation probably covers January–August, so choosing settings on a single window is risky.

---

## 10. Key takeaways

1. **It's a decision problem, not just a prediction problem.** Capacity is 55% of the peak. Accepting everything loses up to €1.7M in a summer quarter, and rejecting everything leaves €100–450k on the table.
2. **Calibrated probabilities beat a high AUC**, because the accept rule adds up probabilities. That's why simple logistic regression beats LightGBM, XGBoost and CatBoost with default settings.
3. **Preprocessing matters more than the model or its hyperparameters**: the full pipeline adds about 13,000 over the minimal one, and dropping the `country` artefact adds about 10,000 more. PCA hurts.
4. **Our best models capture about 60–70% of the possible gain**, but settings chosen on one validation window don't transfer reliably. Round two (section 11) is built around fixing that.

---

## 11. Round two: what we changed and why

Round one left four lessons. Each one led to a change:

| Lesson from round one | Change in round two |
|---|---|
| Settings chosen on one three-month window don't transfer: the best logistic regression on valid lost 25,000 on test | **Four rolling time windows**, settings chosen by their average, plus a held-out check (11.1, 11.5) |
| The accept rule is basic: it ignores risk and price, and it inherits every bias of the model | **Three rule settings** (`alpha`, `z`, `beta`), tuned together with the model (11.4) |
| Every model predicts too few cancellations | **Calibration** on the most recent 90 days, **recency weights**, and `alpha` as a direct correction (11.2, 11.4) |
| The feature set matters more than the hyperparameters | **More feature-set variants**, a drop-one-group study, and only a small hyperparameter grid (11.2, 11.3) |

The search ran on the HPC (section 12). Sections 11.1–11.7 explain the method and the reasons; section 11.8 has the results. The full tables are in `hpc/results/RESULTS.md`.

```
train.csv
   │      for each of 4 time windows:
   ▼
prep.fit_params(until = window start)     cleaning settings learned from earlier data only
prep.prepare()                            same code as in decide()
   │
   ▼
model setting (831 of them)               logistic regression / LightGBM / blend
   │                                      × feature set × recency weights × calibration
   ▼
P(show) for every booking in the window   computed once per model setting and window
   │
   ▼
252 accept rules in one pass              alpha × z × beta, for each hotel
   │
   ▼
net profit per window, hotel and rule     hpc/results/runs/<setting>.json
   │
   ▼
report.py                                 average over the windows, held-out check
   │
   ▼
best.json + RESULTS.md                    hpc/tuned_solution.py reads best.json
```

### 11.1 Validation: four rolling windows

| Window | Trained on arrivals | Replayed arrivals | Bookings | Of which requests | Why it's there |
|---|---|---|---|---|---|
| `jan-aug` | Jul–Dec 2015 | Jan–Aug 2016 | 36,530 | 26,846 (73%) | Same months and same length as the 2017 evaluation probably has |
| `apr-nov` | Jul 2015 – Mar 2016 | Apr–Nov 2016 | 41,606 | 21,372 (51%) | Eight months that include the whole peak season |
| `jul-dec` | Jul 2015 – Jun 2016 | Jul–Dec 2016 | 29,346 | 14,341 (49%) | A full year of training data |
| `oct-dec` | Jul 2015 – Sep 2016 | Oct–Dec 2016 | 14,378 | 5,457 (38%) | The most training data; this is the old test set |

The decisions behind this:

- **Long windows instead of quarters.** In a three-month window, most bookings were made before the window starts. They are commitments and can't be rejected (62% of the bookings in `oct-dec`). The 2017 evaluation probably runs for about eight months, so most of its bookings will be requests. An accept rule tuned on short windows is tuned for the wrong mix.
- **Each window trains only on arrivals before it starts**, like the real evaluation.
- **The preprocessing is refitted for every window** (`prep.fit_params(until=window start)`): category lists, price caps and reference prices come from earlier data only. In round one, the reference price of a month could come from the validation period itself. 2017 won't have that: there, `rate_rel` compares 2017 prices with 2016 reference prices.
- **The score is the share of the possible gain**, not euros. A window's net profit mostly depends on its length and season, so euros can't be averaged across windows. Each window is scaled between two reference strategies:

  `share = (net profit − reject every request) / (hindsight − reject every request)`

  0% means accepting nothing. 100% is the hindsight strategy that knows who shows up and fills the rooms with the highest rates first. That strategy is greedy, so 100% is an approximate ceiling, not a hard limit. Settings are compared by their **average share over the four windows**.

Two weaknesses, stated openly:

- **The windows overlap in time.** With 18 months of data there is no way to get several long windows that don't. They still differ in training data, in what is already committed and in season. The cost is that the held-out check (11.5) is somewhat optimistic.
- **`jan-aug` trains on six months only** (Jul–Dec 2015). Calendar features and reference prices for January–June don't exist yet at that point. It's the hardest window, and also the one closest to 2017 in season.

### 11.2 Preprocessing options in the search

The cleaning pipeline (`prep.py`) stays as it is: round one showed that it beats the minimal preprocessing and PCA. The search varies what the model gets to see, and how:

| Option | Values | Why test it |
|---|---|---|
| Feature set | `baseline`, `no_country`, `no_hindsight`, `no_calendar`, `no_country_calendar` | The biggest effect of anything in round one. `no_calendar` is new and stricter than the old `no_season` (it also drops `arrival_week`). Each calendar month occurs only once or twice in the training data, so month features mostly memorise what happened in that one year |
| Drop one feature group | lead time, deposit, after-booking fields, agent/company IDs, duplicate count, price, guests, guest history, weekday, length of stay | Shows which groups carry the result and which hurt. Run for one logistic regression and one LightGBM (20 settings) |
| Binned numbers (logistic regression only) | off, or 10 quantile bins per number | A linear model can only draw a straight line through lead time, month or price. Bins let it learn curves. Binning is part of the model pipeline, so `decide()` needs no change |
| Recency weights | off, or half-life 180 days | Prices and cancellation behaviour drift (section 2). With the weights, a booking from six months before the window counts half |
| Calibration | off, sigmoid, isotonic | Corrects the predicted probabilities using the most recent 90 days before the window. The correction is learned from a model that hasn't seen those 90 days. The final model is then trained on everything |

Fixed after round one, because they moved the result only by noise (about ±1,000): the rarity threshold for categories (`min_frequency=20`), the number of countries kept (20), and the price cap.

Considered and left out:

- **PCA.** Round one: it removes the category information that predicts cancellations.
- **Booking-pace features** (how many requests were already seen for an arrival date). In 2017 the stream starts on 1 January, so a running count would start at zero in the middle of the booking season, while the training data has the full history. That's a mismatch between training and evaluation that we can't test.
- **Target encoding of categories.** One-hot encoding with a rarity threshold already works and keeps the linear model calibrated. Target encoding leaks easily when 31% of the rows are duplicates.
- **`booking_id`.** It leaks the target (section 2). Using it needs the organisers' OK.

### 11.3 Models in the search

| Model | Grid | Settings |
|---|---|---|
| Logistic regression | `C` 0.03 / 0.3 / 3 × bins off or on × one model or one per hotel | 12 |
| LightGBM | 200 trees (learning rate 0.05) or 600 trees (0.02) × 7 / 15 / 31 leaves × `min_child_samples` 50 / 200. Always with 80% row and column sampling | 12 |
| Blend | Weight of the logistic regression 0.3 / 0.5 / 0.7; the rest is LightGBM | 3 |

Each of these is combined with 5 feature sets × 2 recency settings × 3 calibration settings. That gives 360 + 360 + 90 settings. With the 20 drop-one-group settings and the current submission, the total is **831 model settings**.

Why these models:

- **Logistic regression** won round one because its probabilities are closest to reality, and the accept rule adds probabilities up.
- **LightGBM** ranks bookings better (higher AUC) but was badly calibrated. Calibration and smaller trees (7–15 leaves) target exactly that. XGBoost and CatBoost are left out: in round one all three were within about 1% of each other, and one library means fewer things to install on the cluster.
- **Blend.** Averaging a well-calibrated model with a well-ranking model is the standard way to get both.
- **Small hyperparameter grids on purpose.** In round one, `C` and `min_frequency` moved the profit by about ±1,000, and the feature set by 7,000–20,000. The budget goes where the effect is.
- **No neural network, no Bayesian search (Optuna).** 78,000 rows of tabular data don't call for a network. A plain grid needs no shared database between tasks, can be restarted, and gives the "which setting matters" tables in the report.

### 11.4 The accept rule

Round one's rule: accept if the booking pays and, on every night of the stay, *expected guests + P(show) ≤ free rooms*.

Round two keeps that structure and adds three settings:

```
expected guests + p + z × spread   ≤   free rooms × (alpha + beta × price tilt)
```

| Setting | What it does | Why | Values |
|---|---|---|---|
| `alpha` | Scales the free rooms. Above 1 overbooks more, below 1 keeps a reserve | Even the better models predict too few cancellations (27–31% instead of 36–39% in round one). So "expected guests" is too high and the hotel looks full too early. `alpha` corrects that with one number, in either direction | 0.85 to 1.35 in steps of 0.025 (21 values) |
| `z` | A safety margin, measured in standard deviations of the number of guests who show up that night | The original rule fills a night until overflowing is a coin flip. But a walk costs 2.5× the stay plus the lost revenue, so 3.5× in total, against a gain of 1×. Break-even is an overflow chance of 1/3.5 ≈ 29%, which corresponds to `z` ≈ 0.57 if the probabilities were perfect. The spread treats every booking as a coin flip (variance = Σ p(1−p)) and uses a normal approximation. With 100+ bookings per night that is accurate and costs nothing per decision, so no Monte Carlo simulation is needed | 0, 0.5, 1 |
| `beta` | Price tilt. Bookings above the normal price for that hotel and month may use more of the capacity, cheaper ones less. Price tilt = `rate_rel − 1`, limited to −0.5 … +1 | Rooms are scarce (capacity is 55% of the peak), so it matters *which* bookings get them. Also, the cheapest arriving guests are the ones who get walked, so an expensive booking is rarely the one that pays the penalty. This is a simple stand-in for a bid price that needs no demand forecast | 0, 0.1, 0.2, 0.4 |

21 × 3 × 4 = **252 rules**, each evaluated for each hotel. `alpha=1, z=0, beta=0` is the original rule, so it's always part of the comparison.

**One rule per hotel is allowed.** Downtown and Seaside differ a lot (41.5% vs 26.6% cancellations, city vs resort). The hotels don't share rooms, so their rules can be tuned independently at no extra cost. Whether per-hotel rules hold up is decided by the held-out check (11.5).

**Why `beta` gets the smallest grid.** A test replay of Sep–Nov 2016 shows how the possible gain splits:

| Strategy | Net profit |
|---|---|
| Reject every request | 277,787 |
| Knows who shows up, first come first served | 615,955 |
| Knows who shows up, highest rates first (hindsight) | 667,728 |

About 87% of the possible gain comes from knowing who shows up (probabilities, `alpha`, `z`). About 13% comes from picking the better-paying bookings (`beta`).

**Not done: a full bid-price model** (forecasting the remaining demand per night and its prices). It would need its own validation, and the 13% above is roughly what it could add.

### 11.5 How the final settings are chosen (`hpc/report.py`)

1. **Average share over the four windows**, for every model setting, hotel and rule.
2. **Plateau, not peak.** Before the best rule is picked, each rule's score is averaged with its two `alpha` neighbours. A setting that is only good at exactly `alpha = 1.075` is luck. A good region is real.
3. **Four schemes are compared:**
   - the current submission (logistic regression with the `train.py` settings, original rule);
   - the best model with the original rule;
   - the best model with one tuned rule for both hotels;
   - the best model with one tuned rule per hotel.
4. **A held-out check decides between them.** For each window in turn, the settings are chosen on the other three windows and scored on the one left out. The scheme with the best average held-out share wins. `best.json` is then written from that scheme, using all four windows.

Why this way: with 831 × 252 candidates and only four windows, the single best number is partly luck. The held-out share estimates what the method achieves on a period it hasn't seen. It also stops us from picking a more flexible scheme that only fits the past: if tuning the rule doesn't hold up on unseen windows, the report says so and `best.json` keeps the original rule.

### 11.6 How it stays cheap and correct

- **Train once, replay every rule in one pass.** A model's probabilities don't depend on the accept rule. So each model setting is trained once per window. Then the requests go through in booking order once, and every step updates all 252 rules as one array operation (`hpc/policy.py`). That's 831 × 4 × 252 ≈ 838,000 replays for the cost of 3,324 window trainings.
- **The fast replay matches the original scoring code.** In a small test run (five rules on a two-week window), its net profit equalled `net_profit` from `solution.py` to the cent.
- **The `Solution` class matches the search.** `hpc/tuned_solution.py --check` replays a window through `decide()`, one raw booking at a time, the way the template describes the evaluation. In a small test run, both paths gave 665,899 (1,618 of 2,851 requests accepted). The report job on the HPC repeats this check for the chosen settings and appends it to `RESULTS.md`.

The first check was a small local test run. The report job repeats the second check for the chosen settings: on the cluster, `decide()` and the search agreed exactly (section 11.8).

### 11.7 Size and time limit

| | |
|---|---|
| Model settings | 831, each trained on 4 windows |
| Accept rules | 252 per hotel |
| Parallel tasks | 32 array tasks × 2 cores (`hpc/config.sh`) |
| Time limits | A search task starts no new setting after 30 minutes and is killed at 38. The report job is killed at 20. So **at most 58 minutes of run time**, plus the wait in the queue |
| Actual run time | 7–9 minutes per search task (at most 548 MB of memory), 5 minutes for the report. **14 minutes** from submitting to the finished report, all 831 settings done |

If the time runs out: the settings are processed in a fixed random order, so a cut-off run is still an even sample of the grid. The report works with whatever finished. Submitting again continues where it stopped, because finished settings are skipped.

### 11.8 Results

**Chosen settings** (`hpc/results/best.json`):
- **Model:** a blend of 70% logistic regression (`C=0.3`) and 30% LightGBM (600 trees, 15 leaves).
- **Data:** feature set `no_calendar`, no calibration, no recency weights, one model for both hotels.
- **Accept rule, the same for both hotels:** `alpha` 1.1, `z` 0, `beta` 0.1. That means room for 10% more expected guests than there are free rooms, and a small preference for bookings above the normal price.

**Net profit against the current submission** (logistic regression with the `train.py` settings, original rule):

| Window | Current submission | Chosen, picked without this window* | Gain | Chosen, picked on all windows |
|---|---|---|---|---|
| `jan-aug` | 3,614,480 (58.9%) | ≈ 3,876,000 (65.4%) | ≈ +261,000 | 4,010,947 (68.7%) |
| `apr-nov` | 3,110,049 (57.5%) | ≈ 3,332,000 (71.1%) | ≈ +222,000 | 3,332,087 (71.1%) |
| `jul-dec` | 1,491,300 (63.9%) | ≈ 1,688,000 (80.0%) | ≈ +197,000 | 1,688,899 (80.0%) |
| `oct-dec` | 766,999 (63.5%) | ≈ 837,000 (79.3%) | ≈ +70,000 | 860,521 (84.5%) |
| **Average share** | **61.0%** | **74.0%** | **+13.0 pp** | 76.1% |

\* The held-out check from 11.5: the settings were chosen on the other three windows. The euros are reconstructed from the shares, which are rounded to 0.1%. The windows overlap in time (11.1), so even this column is somewhat optimistic.

**Where the gain comes from: the rule, not the model.**
- With the original rule, the chosen model is within ±35,000 of the current submission in every window. `alpha` 1.1 and `beta` 0.1 add 84,000–380,000 per window.
- Going from `alpha` 1.0 to 1.1 improves every one of the four windows. In share: 63.5 → 68.7%, 68.3 → 71.1%, 77.6 → 80.0% and 78.0 → 84.5%.
- The reason: the chosen model predicts 22–29% cancellations, but 35–39% actually happen. So the expected guests are too high and the hotel looks full too early. `alpha` 1.1 makes up for it.
- **Not luck:** every `alpha` from 1.025 to 1.15 scores within 0.4 points of the best, so it's a plateau and not a single peak.
- **`z` makes no difference** (0, 0.5 and 1 score the same). An extra safety margin only shifts the threshold by a few rooms, and `alpha` already sets the threshold.
- **`beta` 0.1–0.2 adds 2–3 points**, while 0.4 is too much. Taking the price into account helps, but only a little. That fits the about-13% upper limit from 11.4.
- **A separate rule per hotel fits the past better but loses on the held-out windows** (71.6% vs 74.0%). The held-out check caught this, so both hotels get one rule.

**Models and hyperparameters barely matter once the rule is tuned.**
- At their best, the blend, logistic regression and LightGBM are within 0.3 points of each other (76.1%, 75.9% and 75.8%).
- `C` 0.03 regularises too strongly (about −3 points on average). Apart from that, `C`, bins, trees, leaves and `min_child_samples` each move the average by about 1 point or less.
- One model per hotel loses about 4 points on average. Recency weights are slightly worse (−0.9 points).

**Calibration loses to `alpha`.**
- With the original rule, calibration helps (+1.6 to +2.5 points), because nothing else corrects the bias.
- With `alpha` tuned, it hurts. For the chosen model and features: none 76.1%, isotonic 74.1%, sigmoid 73.6%.
- The worst case is `jan-aug` (isotonic 61.4% vs none 68.7%). There the correction is learned on October–December and applied to January–August, the same situation as calibrating on late 2016 for 2017. In that window the calibration model also had only three months of training data, so part of the loss comes from that.
- One overall correction (`alpha`) works better than a curve learned in another season.

**Features: the calendar hurts, and `country` changes sides.**
- **Calendar features hurt.** `no_calendar` is the best feature set: each calendar month appears only once or twice in the training data.
- **`country` with the original rule:** dropping it helps by 4–10 points, which is round one's finding.
- **`country` with a tuned rule:** dropping it hurts by 1–2 points. Most of the harm `country` did came from the too-low cancellation predictions it causes, and `alpha` now corrects that. Exception: in `jan-aug`, which trains only on 2015 where the Portugal artefact is strongest, dropping `country` is still better (71.4% vs 68.7%).
- **Removing the fields recorded after booking costs 3–6 points** (`no_hindsight`).
- **Feature groups:** for logistic regression with a tuned rule, the ones that matter most are `deposit` (−2.2 points when removed), agent/company (−2.0) and guest history (−1.5). Removing the duplicate count, the guest counts or the weekdays changes almost nothing.

**The `Solution` class reproduces the search.** All raw October–December bookings went one at a time through `hpc/tuned_solution.py`. That gave 860,521, exactly the search's number (2,610 of 5,457 requests accepted), at 45.5 ms per booking on the cluster.

**Comparison with `submission/`** (prep_v2 + CatBoost with isotonic calibration + the bid-price rule): its reported result on October–December 2016 is 823,400, with the same setup (its first-come reference value, 901,761, matches exactly). The chosen settings give 860,521 when chosen with this window, and about 837,000 when chosen without it. The bid-price rule was chosen on this one window only, and it hasn't been through the four-window comparison yet (section 14).

---

## 12. Running it on the HPC

Nothing is computed on the laptop. `hpc/hpc.sh` only copies files and talks to the cluster over ssh.

| File | What it is |
|---|---|
| `hpc/config.sh` | Cluster settings: ssh host, folder, number of tasks, time budget, Python module |
| `hpc/hpc.sh` | Run on the laptop: upload, submit, status, fetch |
| `hpc/job.sh` | Runs on the cluster: sets up Python, submits the jobs. Works with LSF (`bsub`) and SLURM (`sbatch`) |
| `hpc/search.py` | The search: windows, model settings, one JSON per setting |
| `hpc/policy.py` | The accept rule, and the fast replay that scores it |
| `hpc/report.py` | Chooses the settings; writes `best.json` and `RESULTS.md` |
| `hpc/tuned_solution.py` | The `Solution` class that uses `best.json` |

```bash
# once: check hpc/config.sh
#   HPC_HOST  the ssh name of the cluster (default: dtu)
#   MODULES   only if the cluster's python3 is older than 3.9, e.g. "python3/3.11.9"

bash hpc/hpc.sh go        # upload code + train.csv, create the Python environment, submit the jobs
bash hpc/hpc.sh status    # queue, and how many model settings are finished
bash hpc/hpc.sh fetch     # copy the results back into hpc/results/
```

What `go` starts on the cluster:

1. An **array job** of 32 tasks. Each takes every 32nd model setting and writes one file per setting to `hpc/results/runs/`.
2. A **report job** that starts when the array has ended. It computes the reference strategies, runs `report.py`, and runs the `Solution` check.

What comes back with `fetch`:

- `hpc/results/best.json`: the model setting and the accept rule per hotel. **Commit this file**; the `Solution` needs it.
- `hpc/results/RESULTS.md`: the chosen settings, the held-out check, net profit per window, which settings and feature groups matter, and the `Solution` check.
- `hpc/logs/`: one log per task, for when something fails.

Optional, before `go`: `bash hpc/hpc.sh upload && bash hpc/hpc.sh setup && bash hpc/hpc.sh test` runs a tiny end-to-end version on the cluster's login node (about 3 minutes on one core). It catches installation problems before any queue time is spent.

### The submission (`submission/`)

`submission/` holds the search's winner, and it doesn't depend on `hpc/`:

| File | What it is |
|---|---|
| `submission/solution.py` | The `Solution` class. The chosen model and rule settings are written at the top, copied from `best.json` |
| `submission/prep.py` | A copy of `claude-preprocessing/prep.py`. The duplicate count also works if a 2017 booking lacks `booking_id` or another raw column |
| `submission/check.py` | End-to-end check: trains on arrivals before October 2016, feeds Oct–Dec 2016 through `decide()` one raw booking at a time, and compares with the search (860,521) |
| `submission/export_model.py` | Trains the final model on all of `train.csv` and saves it as `submission/model.joblib`, so the evaluation doesn't have to train. It reloads the file and checks that the predictions are identical |
| `submission/model.joblib` | The trained model: preprocessing parameters, logistic regression, LightGBM. It is made on the cluster |
| `submission/requirements.txt` | Exact package versions: those of the laptop the evaluation runs on. The cluster builds a second environment with them (`hpc/venv-submission`) for the check and the export |

**Why the versions are pinned.** A saved model only loads reliably with the same versions of numpy, pandas, scikit-learn and LightGBM. The search's environment on the cluster has newer ones than the laptop. So the model is exported in a cluster environment that matches the laptop.

`Solution` checks the file before using it: the model settings, `prep.py` and the package versions must match. If they don't, or if the file is missing, it trains from `train.csv` instead. That takes about a minute and gives the same decisions, so the submission works either way.

```python
import sys
sys.path.insert(0, "submission")
from solution import Solution

evaluate(Solution)
```

At start-up, `Solution` loads `model.joblib`. Without a usable file, it fits the preprocessing and the blend on all of `train.csv` (next to `solution.py`, or in `train-template-kit/`). After that, `decide()` applies the overbooking rule to every booking. It needs the packages in `submission/requirements.txt`.

Before submitting, run both steps on the cluster. Training a model locally is exactly what we avoid:

```bash
bash hpc/hpc.sh check      # end-to-end check; hpc/logs/submission.out should say MATCH
bash hpc/hpc.sh export     # trains the final model and saves submission/model.joblib
bash hpc/hpc.sh fetch      # copies model.joblib and the logs back
python submission/export_model.py --status   # no training: "usable here" means the laptop will load it
```

Then commit `submission/model.joblib`. It's a few MB.

If the search is run again and `best.json` changes, update the settings at the top of `submission/solution.py`. `check.py` reports a mismatch between the two.

The previous submission (prep_v2 + CatBoost + bid prices, by David) is in git commit `e92558c`.

---

## 13. Decision log

Every major decision in one place.

| # | Decision | Instead of | Why |
|---|---|---|---|
| 1 | Split by time | Random split, cross-validation | 31% of the rows are duplicates from group blocks and would land on both sides. Behaviour also drifts over time |
| 2 | Judge everything by the replayed net profit | AUC, log-loss | Net profit is the hackathon's score. In round one, AUC went down while profit went up |
| 3 | One preprocessing module for training and `decide()` | Cleaning in a notebook | 2017 bookings arrive one at a time and must be cleaned exactly like the training data |
| 4 | Keep the full pipeline | Minimal preprocessing, PCA | +13,000 over minimal. PCA removes the category information |
| 5 | Don't use `booking_id` | Using it (AUC 0.82 alone) | It's a row number that leaks the target. Needs the organisers' OK |
| 6 | Test feature sets without `country` and without calendar features | Always all features | `country` is partly a recording artefact (+10,000 without it). Each calendar month is seen only once or twice. **Result:** calendar features dropped, `country` kept. With a tuned rule it helps (11.8) |
| 7 | Four long rolling windows | One validation quarter | The best setting on one quarter lost 25,000 on the next. 2017 is a long window where most bookings are requests |
| 8 | Refit the preprocessing per window | One set of parameters for all | Otherwise reference prices from the window itself leak in, which 2017 won't have |
| 9 | Score = share of the possible gain, averaged over the windows | Sum of euros | Euros depend on the window's length and season, so one window would dominate |
| 10 | Logistic regression, LightGBM and a blend | Also XGBoost, CatBoost, a neural network | The three tree libraries were within 1% of each other. One library is simpler on the cluster |
| 11 | Small hyperparameter grid, larger preprocessing grid | A deep hyperparameter search | Hyperparameters moved profit by ±1,000, feature sets by 7,000–20,000 |
| 12 | Calibration on the latest 90 days, and recency weights | Using raw probabilities only | Every model predicted too few cancellations. The rule adds probabilities up, so they must be right in absolute terms. **Result:** both lost to tuning `alpha`, so the chosen settings use neither (11.8) |
| 13 | Accept rule with `alpha`, `z`, `beta` | The original rule; a full bid-price model | `alpha` fixes the bias, `z` covers the risk (walks cost 3.5× the gain), `beta` favours better-paying bookings. Each has one clear job, and the original rule is included |
| 14 | Normal approximation for the risk | Monte Carlo simulation | With 100+ bookings per night it is accurate, and it costs nothing per decision |
| 15 | Rules may differ per hotel | One rule for both | The hotels differ a lot and share no rooms. The held-out check decides whether it helps. **Result:** it doesn't (71.6% vs 74.0% held out), so one rule for both |
| 16 | Pick a plateau: average over neighbouring `alpha` values | The single best value | A lone peak is usually luck |
| 17 | The held-out check picks the scheme | Trusting the best average | With 209,000 candidates and four windows, the best number is partly luck |
| 18 | Train once, replay all rules in one pass | One run per rule | 838,000 replays for the cost of 3,324 trainings. Checked against the original scoring code |
| 19 | Plain grid in random order, with a time budget | Optuna; a fixed order | No shared database, restartable, an early stop still covers the grid evenly, and it yields the tables for the report |
| 20 | Everything heavy runs on the cluster, in under an hour | Running on the laptop | The laptop stays free. The time limits are enforced by the scheduler |
| 21 | Overbook by 10% (`alpha` 1.1) with a small price tilt (`beta` 0.1) | The original rule; calibrating the probabilities | +13 points of held-out share, and it improves all four windows. It sits on a broad plateau (1.025–1.15) |

---

## 14. Still open

- **The bid-price submission has not been compared on equal terms.** `submission/` now holds the search's winner (section 12). The previous version (prep_v2, CatBoost with isotonic calibration and bid prices, git commit `e92558c`) was chosen on October–December 2016 only. A fair comparison would run both through the same four windows. Two of its findings work against `submission/`: isotonic calibration lost to `alpha`, and the risk check (reject at a 1/3.5 overflow chance) is more cautious than the original rule, while the search found that accepting more is better. Bid prices can only be judged on windows from July 2016 on, because their forecast needs the same weeks one year earlier.
- **`decide()` is slow:** 45.5 ms per booking on the cluster, almost all of it in `prep.prepare()` on a single row. If 2017 has as many requests as January–August 2016 (26,846), that's about 20 minutes. It's worth checking whether the evaluation has a time limit.
- **Package versions.** The code was tested with Python 3.14, pandas 3.0, scikit-learn 1.8 and LightGBM 4.6. On the cluster, a Python of 3.11 or newer gets the same major versions. `bash hpc/hpc.sh test` shows quickly whether an older one works too.
- **Not in the search:** bid prices (built in `submission/`, see the first point), booking-pace features and `booking_id` (11.2, 11.4).
- **`claude-preprocessing/data/` is not in git.** Only round one needs it. `hpc/` reads `train-template-kit/train.csv` directly.
- **LightGBM and parallel runs:** LightGBM ignores `OMP_NUM_THREADS`, so several processes side by side start one thread per core each and slow down about 500×. Both `grid_search.py` and `hpc/search.py` pass the thread count to LightGBM explicitly.

---

## Reproduce

```bash
# round one (sections 4-9): needs the splits from claude-preprocessing/preprocessing.ipynb, sections 3 and 4
python test_preprocessing/pca_preprocess.py            # PCA splits (only for the comparison)
python minimal_preprocessing/preprocess.py             # minimal splits (only for the comparison)
python grid_search/compare.py                          # sections 4.2, 5 and 7 (train -> valid)
python grid_search/compare.py --test                   # section 9 (train + valid -> test)
python grid_search/grid_search.py logreg --summary     # section 8

# round two (sections 11-13): on the HPC
bash hpc/hpc.sh go
bash hpc/hpc.sh status
bash hpc/hpc.sh fetch                                  # hpc/results/best.json and RESULTS.md
```
