"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.

Run `python train-template-kit/solution.py`: trains on TRAIN_FILE + VALID_FILE, prints the net profit on TEST_FILE.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "prep_v2")]  # train.py next to this file, prep.py in prep_v2/ (made splits_v2)
import prep  # noqa: E402
from train import MODEL, train  # noqa: E402

# =============================================================================================
# INPUT FILES (preprocessed .parquet or .csv with the train.csv columns + features)
# =============================================================================================
DATA = HERE.parent / "data"
SPLITS = DATA / "splits_v2" 
TRAIN_FILE = SPLITS / "train.parquet"
VALID_FILE = SPLITS / "valid.parquet"
TEST_FILE = SPLITS / "test.parquet"
PARAMS_FILE = SPLITS / "params.json"  # optional, None for splits_minimal; evaluate.py needs it for raw bookings
# =============================================================================================

PER_HOTEL = False  # submission: False = one model for both hotels, True = one model per hotel

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
NOT_FEATURES = ["cancelled", "booking_id", "arrival_year", "arrival_date", "booking_date",
                "split", "is_commitment", "use_for_occupancy", "stay_value"]
PARAMS = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE and PARAMS_FILE.exists() else None


def load(path):
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    return df


def features(df):
    """From the params file if given, else every numeric / category column that is no label, id or date."""
    if PARAMS:
        return PARAMS["features"]
    return [c for c in df.select_dtypes(["number", "category"]).columns if c not in NOT_FEATURES]


def pick(models, per_hotel):
    """Together: {"all": model}. Separated: {"Downtown Hotel": model, "Seaside Hotel": model}."""
    return {h: m for h, m in models.items() if h != "all"} if per_hotel else {"all": models["all"]}


def nights_of(booking):
    return pd.date_range(pd.Timestamp(booking["arrival_date"]), periods=int(booking["nights"]))


def as_series(free):
    """Free rooms per night as a Series with a DatetimeIndex (accepts a Series or a {date: rooms} dict)."""
    s = free.copy() if isinstance(free, pd.Series) else pd.Series(free, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


class Solution:
    """Accept rule = risk check + bid prices.
    1. Risk: per night the exact distribution of how many accepted guests show up (each shows with its p).
       Reject if P(hotel already full on a night of the stay) >= Q_MAX. Break-even: a seated guest earns v,
       a walk costs 2.5 v, so accepting pays while (1 - q) v > 2.5 q v, i.e. q < 1 / 3.5.
    2. Value: each night has a bid price = what one more free room is worth for the requests still to come
       (dual of an LP over last year's bookings, shifted 52 weeks). Reject if v < THETA x sum of bid prices."""
    Q_MAX = 1 / 3.5
    THETA = 1.0  # 0 switches the bid prices off (risk check only)
    SHIFT = pd.Timedelta(days=364)  # forecast = bookings 52 weeks earlier (same weekday)
    MAX_GROWTH = None  # cap on the year-over-year demand growth used in the forecast (None = no cap)

    def __init__(self, commitments, capacity, models=None, history=None):
        """evaluate.py calls Solution(commitments, capacity): the models then train on all three files, which are
        also the booking history behind the demand forecast. Dev runs pass models and history (train + valid)."""
        if models is None:
            rows = pd.concat([load(TRAIN_FILE), load(VALID_FILE), load(TEST_FILE)])
            models = pick(train(rows, features(rows)), PER_HOTEL)
            history = rows if history is None else history
        if history is None:  # never let test bookings into the forecast of a dev run
            history = pd.concat([load(TRAIN_FILE), load(VALID_FILE)])
        self.models = models
        self.capacity = {hotel: as_series(free) for hotel, free in capacity.items()}
        self.expected = {hotel: defaultdict(float) for hotel in capacity}  # night -> expected guests
        self.dist = {}  # (hotel, night) -> P(k shows) for k < free rooms, last entry P(>= free rooms)
        self.bids = {}  # hotel -> (booking day, {night: bid price})
        self.demand = self.forecast(history)
        self.counter = prep.RequestCounter()

        if not set(next(iter(models.values())).used_features) <= set(commitments.columns):  # raw rows (evaluate.py)
            commitments = prep.prepare(commitments, PARAMS)
        for booking, p in zip(commitments.to_dict("records"), self.p_show(commitments)):
            self.book(booking, p)

    def decide(self, booking):
        """evaluate.py: one raw booking -> preprocess (needs PARAMS_FILE) -> accept or reject."""
        x = prep.prepare(booking, PARAMS, counter=self.counter)
        return self.accept(booking, self.p_show(x)[0])

    def p_show(self, x):
        """Probability that the guest shows up, from the joint model or the model of the row's hotel."""
        key = pd.Series("all", index=x.index) if "all" in self.models else x["hotel"].astype(str)
        p = pd.Series(0.0, index=x.index)
        for name, model in self.models.items():
            rows = key == name
            if rows.any():
                p[rows] = 1 - model.predict_proba(x.loc[rows, model.used_features])[:, 1]
        return p.to_numpy()

    def accept(self, booking, p):
        """Accept if it pays, the risk of being full is below Q_MAX and it is worth the nights' bid prices."""
        if booking["nightly_rate"] <= 0 or self.q(booking) >= self.Q_MAX:
            return False
        if self.THETA > 0:
            hotel, today = str(booking["hotel"]), pd.Timestamp(booking["booking_date"]).normalize()
            if self.bids.get(hotel, (None,))[0] != today:  # re-solve once per booking day and hotel
                self.bids[hotel] = (today, self.bid_prices(hotel, today))
            bids = self.bids[hotel][1]
            value = float(booking["nightly_rate"]) * int(booking["nights"])
            if value < self.THETA * sum(bids.get(n, 0.0) for n in nights_of(booking)):
                return False
        self.book(booking, p)
        return True

    def book(self, booking, p):
        hotel = str(booking["hotel"])
        for night in nights_of(booking):
            self.expected[hotel][night] += p
            d = self._dist(hotel, night)
            if d is not None and len(d) > 1:
                new = d * (1 - p)
                new[1:] += d[:-1] * p
                new[-1] += d[-1] * p  # the ">= free rooms" bucket only grows
                self.dist[(hotel, night)] = new

    # --- risk: exact (Poisson-binomial) distribution of shows per night -----------------------------
    def _dist(self, hotel, night):
        d = self.dist.get((hotel, night))
        if d is None:
            free = self.capacity[hotel].get(night) if hotel in self.capacity else None
            if free is None or pd.isna(free):  # night outside the capacity window: unconstrained
                return None
            d = np.zeros(int(free) + 1)
            d[0] = 1.0
            self.dist[(hotel, night)] = d
        return d

    def q(self, booking):
        """P(the guests accepted so far already fill the hotel) on the tightest night of the stay."""
        hotel = str(booking["hotel"])
        tails = [d[-1] for d in (self._dist(hotel, n) for n in nights_of(booking)) if d is not None]
        return max(tails, default=0.0)

    # --- value: bid prices from forecast demand ----------------------------------------------------
    def forecast(self, rows):
        """Per hotel: last year's shown bookings shifted 52 weeks into the capacity window, scaled by growth
        (arrivals in the last 3 months of history vs the same 3 months a year earlier), plus the
        nights x bookings matrix the LP needs."""
        demand = {}
        end = rows["arrival_date"].max()
        quarter = pd.Timedelta(days=91)
        hotels = rows["hotel"].astype(str)
        for hotel, free in self.capacity.items():
            mine = rows[hotels == hotel]
            recent = (mine["arrival_date"] > end - quarter).sum()
            before = ((mine["arrival_date"] > end - quarter - self.SHIFT) & (mine["arrival_date"] <= end - self.SHIFT)).sum()
            growth = recent / before if before else 1.0
            if self.MAX_GROWTH is not None:
                growth = min(growth, self.MAX_GROWTH)
            shown = mine[(mine["cancelled"] == 0) & (mine["nightly_rate"] > 0)]
            arrival = shown["arrival_date"] + self.SHIFT
            keep = ((arrival >= free.index.min()) & (arrival <= free.index.max())).to_numpy() if len(free) else False
            d = pd.DataFrame({"booked": (shown["booking_date"] + self.SHIFT)[keep].to_numpy(),
                              "arrival": arrival[keep].to_numpy(),
                              "nights": shown["nights"][keep].astype(int).to_numpy(),
                              "value": (shown["nightly_rate"] * shown["nights"])[keep].to_numpy()})
            k = d["nights"].to_numpy()
            col = np.repeat(np.arange(len(d)), k)
            offset = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
            row = free.index.get_indexer(np.repeat(d["arrival"].to_numpy(), k) + offset.astype("timedelta64[D]"))
            ok = row >= 0
            A = csr_matrix((np.ones(ok.sum()), (row[ok], col[ok])), shape=(len(free), len(d)))
            demand[hotel] = (d, growth, A)
        return demand

    def bid_prices(self, hotel, today):
        """max sum v_j x_j  s.t. per night sum x_j <= free rooms - expected guests,  0 <= x_j <= growth,
        over the forecast bookings still to come. Bid price of a night = dual of its room constraint."""
        if hotel not in self.demand:
            return {}
        d, growth, A = self.demand[hotel]
        free = self.capacity[hotel]
        future = (d["booked"] > today).to_numpy()
        nights = (free.index >= today)
        if not future.any() or not nights.any():
            return {}
        left = np.array([max(rooms - self.expected[hotel][n], 0.0) for n, rooms in free[nights].items()])
        res = linprog(-d["value"].to_numpy()[future], A_ub=A[nights][:, future], b_ub=left,
                      bounds=(0, growth), method="highs")
        return dict(zip(free.index[nights], -res.ineqlin.marginals)) if res.status == 0 else {}


def net_profit(confirmed, capacity):
    """Check in day by day: the most valuable guests get a room, the cheapest are walked (-2.5x)."""
    shows = confirmed[confirmed["cancelled"] == 0]
    total = 0.0
    for hotel, free in capacity.items():
        taken = defaultdict(int)  # night -> rooms taken by seated guests
        for day, guests in shows[shows["hotel"] == hotel].groupby("arrival_date"):
            values = (guests["nightly_rate"] * guests["nights"]).sort_values(ascending=False)
            rooms = max(int(free.get(day, ROOMS[hotel])) - taken[day], 0)
            for nights in guests.loc[values.index[:rooms], "nights"]:
                for night in pd.date_range(day, periods=nights):
                    taken[night] += 1
            total += values.iloc[:rooms].sum() - 2.5 * values.iloc[rooms:].sum()
    return total


if __name__ == "__main__":
    rows = pd.concat([load(TRAIN_FILE), load(VALID_FILE)])
    models = train(rows, features(rows))

    test = load(TEST_FILE)
    start = test["arrival_date"].min()
    commitments = test[test["booking_date"] < start]  # booked before the test period: already confirmed
    requests = test[test["booking_date"] >= start].sort_values("booking_date", kind="stable")
    # Free rooms = rooms minus guests who arrived before the test period, showed up and are still in house
    stayed = rows[(rows["arrival_date"] < start) & (rows["cancelled"] == 0)
                  & (rows["arrival_date"] + pd.to_timedelta(rows["nights"], unit="D") > start)]
    capacity = {}
    for hotel, rooms in ROOMS.items():
        free = pd.Series(rooms, index=pd.date_range(start, test["arrival_date"].max()))
        for arrival, nights in zip(stayed.loc[stayed["hotel"] == hotel, "arrival_date"], stayed.loc[stayed["hotel"] == hotel, "nights"]):
            free.loc[start: arrival + pd.Timedelta(days=int(nights) - 1)] -= 1
        capacity[hotel] = free.clip(lower=0)

    for name, per_hotel in [("together", False), ("separated", True)]:
        solution = Solution(commitments, capacity, pick(models, per_hotel), history=rows)
        accepted = [solution.accept(b, p) for b, p in zip(requests.to_dict("records"), solution.p_show(requests))]
        profit = net_profit(pd.concat([commitments, requests[accepted]]), capacity)
        print(f"{MODEL} {name:<9} accepted {sum(accepted)} of {len(requests)} requests | net profit on test: {profit:,.0f}")
