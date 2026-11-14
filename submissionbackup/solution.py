"""Our submission: prep_v2 + CatBoost (isotonic-calibrated) + BidPrice accept rule.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

evaluate(Solution) creates Solution(commitments, capacity) once and calls decide(booking) for every
2017 request in booking-date order. __init__ trains everything from train.csv (about a minute),
so there are no model files to keep in sync.

How it decides (BidPrice):
1. P(show) for every booking: prep_v2 features -> CatBoost -> isotonic calibration.
2. Risk check: exact distribution of how many accepted guests show up on each night; reject if the
   hotel is already full on a night of the stay with probability >= Q_MAX.
3. Value check: each night gets a bid price (what a room is worth if we keep it for later requests),
   from a linear program over last year's bookings shifted by 52 weeks. Accept if the stay value is
   at least THETA x the sum of the stay's bid prices.

Local end-to-end check: python submissionbackup/check.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep  # noqa: E402  (submissionbackup/prep.py)

# =============================================================================================
# SETTINGS
# =============================================================================================
TRAIN_CSV = Path(__file__).resolve().parent.parent / "train-template-kit" / "train.csv"
TRAIN_END = pd.Timestamp("2017-01-01")  # learn from every arrival before this date
CAL_MONTHS = 3  # the last 3 months before TRAIN_END calibrate the probabilities (isotonic)
WALK_COST = 2.5
Q_MAX = 1 / (1 + WALK_COST)  # reject if P(hotel already full on a night of the stay) >= Q_MAX
THETA = 1.0  # accept if stay value >= THETA x sum of the stay's bid prices
SHIFT = pd.Timedelta(days=364)  # demand forecast: last year's bookings, same weekday
CATBOOST = dict(iterations=500, learning_rate=0.05, random_seed=0, verbose=0, allow_writing_files=False)
# =============================================================================================


class Solution:
    def __init__(self, commitments: pd.DataFrame, capacity: dict[str, pd.Series], train_end=TRAIN_END) -> None:
        """Called once, before 2017 starts.

        commitments  2017 arrivals booked before 2017-01-01 (already confirmed; no cancelled column).
        capacity     hotel -> rooms free on each night of 2017 (indexed by date).
        train_end    only for local checks: learn from arrivals before this date.
        """
        train_end = pd.Timestamp(train_end)
        history = pd.read_csv(TRAIN_CSV, parse_dates=["arrival_date", "booking_date"])
        history = history[history["arrival_date"] < train_end]
        self._train(history, train_end)

        self.capacity = capacity
        self.expected = {hotel: defaultdict(float) for hotel in capacity}  # night -> expected guests
        self.dist = {}  # (hotel, night) -> P(k guests show) for k < free rooms, last entry = P(>= free rooms)
        self.bids = {}  # hotel -> (booking day, {night: bid price})
        self.counter = prep.RequestCounter()
        self._forecast(history[~prep.bad_rows(history)])

        x = prep.prepare(commitments, self.params)
        for booking, p in zip(commitments.to_dict("records"), self.p_show(x)):
            self.book(booking, p)

    def decide(self, booking: dict) -> bool:
        """Called once per 2017 booking request, in booking-date order. True = confirm the booking."""
        x = prep.prepare(pd.DataFrame([booking]), self.params, counter=self.counter)  # every request: keeps the counter right
        return self.accept(booking, float(self.p_show(x)[0]))

    # -----------------------------------------------------------------------------------------
    # Model: CatBoost on the prep_v2 features, calibrated with isotonic regression
    # -----------------------------------------------------------------------------------------
    def _train(self, history, train_end):
        cal_start = train_end - pd.DateOffset(months=CAL_MONTHS)
        self.params = prep.fit_params(history[history["arrival_date"] < cal_start])  # learned before the calibration months
        data = prep.prepare(history, self.params)
        fit = data[(data["arrival_date"] < cal_start) & ~prep.bad_rows(data)]
        cal = data[data["arrival_date"] >= cal_start]
        self.model = CatBoostClassifier(cat_features=prep.CATEGORICAL, **CATBOOST)
        self.model.fit(self._X(fit), fit["cancelled"])
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.iso.fit(self.model.predict_proba(self._X(cal))[:, 1], cal["cancelled"])

    @staticmethod
    def _X(x):
        X = x[prep.FEATURES].copy()
        X[prep.CATEGORICAL] = X[prep.CATEGORICAL].astype(str)  # CatBoost takes categories as text
        return X

    def p_show(self, x):
        """Probability that the guest shows up."""
        return 1 - self.iso.predict(self.model.predict_proba(self._X(x))[:, 1])

    # -----------------------------------------------------------------------------------------
    # Accept rule: risk check + bid prices
    # -----------------------------------------------------------------------------------------
    def accept(self, booking, p):
        if booking["nightly_rate"] <= 0 or self.risk(booking) >= Q_MAX:
            return False
        hotel, today = str(booking["hotel"]), booking["booking_date"]
        if self.bids.get(hotel, (None,))[0] != today:  # bid prices are re-solved once per booking day
            self.bids[hotel] = (today, self._bid_prices(hotel, today))
        bids = self.bids[hotel][1]
        if booking["nightly_rate"] * booking["nights"] >= THETA * sum(bids.get(n, 0.0) for n in _nights(booking)):
            self.book(booking, p)
            return True
        return False

    def book(self, booking, p):
        hotel = str(booking["hotel"])
        for night in _nights(booking):
            self.expected[hotel][night] += p
            d = self._dist(hotel, night)
            if d is None or len(d) == 1:
                continue
            new = d * (1 - p)
            new[1:] += d[:-1] * p
            new[-1] += d[-1] * p  # the ">= free rooms" bucket keeps everything that reaches it
            self.dist[(hotel, night)] = new

    def risk(self, booking):
        """P(the guests already accepted fill the hotel), on the tightest night of the stay."""
        hotel = str(booking["hotel"])
        tails = [d[-1] for d in (self._dist(hotel, n) for n in _nights(booking)) if d is not None]
        return max(tails, default=0.0)

    def _dist(self, hotel, night):
        d = self.dist.get((hotel, night))
        if d is None:
            free = self.capacity[hotel].get(night, None)
            if free is None:  # night after the period: unconstrained
                return None
            d = np.zeros(int(free) + 1)
            d[0] = 1.0
            self.dist[(hotel, night)] = d
        return d

    def _forecast(self, rows):
        """Expected future requests = last year's shown bookings shifted by 52 weeks, scaled by recent growth."""
        self.demand = {}
        end = rows["arrival_date"].max()
        for hotel, cap in self.capacity.items():
            is_hotel = rows["hotel"].astype(str) == hotel
            recent = (is_hotel & (rows["arrival_date"] > end - pd.Timedelta(days=91))).sum()
            before = (is_hotel & (rows["arrival_date"] > end - pd.Timedelta(days=91) - SHIFT)
                      & (rows["arrival_date"] <= end - SHIFT)).sum()
            growth = recent / before if before else 1.0
            shown = rows[is_hotel & (rows["cancelled"] == 0)]
            arrival = shown["arrival_date"] + SHIFT
            shown = shown[(arrival >= cap.index.min()) & (arrival <= cap.index.max())]
            self.demand[hotel] = (pd.DataFrame({
                "booked": shown["booking_date"] + SHIFT, "arrival": shown["arrival_date"] + SHIFT,
                "nights": shown["nights"].astype(int), "value": shown["nightly_rate"] * shown["nights"]}), growth)

    def _bid_prices(self, hotel, today):
        """max sum(value x) s.t. per night sum(x) <= free rooms - expected guests, 0 <= x <= growth.
        The dual value of each night's constraint is its bid price."""
        demand, growth = self.demand[hotel]
        future = demand[(demand["booked"] > today) & (demand["value"] > 0)]
        cap = self.capacity[hotel]
        nights = cap.index[cap.index >= today]
        if future.empty or len(nights) == 0:
            return {}
        offset = (future["arrival"] - nights[0]).dt.days.to_numpy()
        k = future["nights"].to_numpy()
        cols = np.repeat(np.arange(len(future)), k)
        rows = np.repeat(offset, k) + np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        ok = (rows >= 0) & (rows < len(nights))
        A = coo_matrix((np.ones(ok.sum()), (rows[ok], cols[ok])), shape=(len(nights), len(future))).tocsr()
        free = np.array([max(cap[n] - self.expected[hotel][n], 0.0) for n in nights])
        res = linprog(-future["value"].to_numpy(), A_ub=A, b_ub=free, bounds=(0, growth), method="highs")
        if res.status != 0:
            return {}
        return dict(zip(nights, -res.ineqlin.marginals))


def _nights(booking):
    return pd.date_range(booking["arrival_date"], periods=int(booking["nights"]))
