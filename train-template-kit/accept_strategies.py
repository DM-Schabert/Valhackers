"""Different accept() rules, each a drop-in subclass of solution.Solution.

Run `python train-template-kit/accept_strategies.py` to train once and compare all strategies on TEST_FILE.
To submit one: `from accept_strategies import Threshold as Solution` (or pass the class to evaluate()).

Notation: v = nightly_rate x nights (stay value), p = P(show), C = free rooms on a night,
q = P(guests already accepted fill the hotel on the tightest night of the stay).
Accepting gains p*v when there is room and costs p*2.5*v when someone is walked,
so with q alone the break-even point is q < 1/3.5 ~ 0.286.
"""
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from solution import (ROOMS, TEST_FILE, TRAIN_FILE, VALID_FILE, Solution, features, load, net_profit,
                      pick)
from train import MODEL, train

WALK_COST = 2.5


def nights_of(booking):
    return pd.date_range(booking["arrival_date"], periods=int(booking["nights"]))


def stay_value(booking):
    return float(booking["nightly_rate"]) * int(booking["nights"])


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------------------------
# 1. Baseline: the original rule (expected guests + p must fit on every night)
# ---------------------------------------------------------------------------------------------
class Baseline(Solution):
    def accept(self, booking, p):
        """Original rule: accept if it pays and the expected guests still fit on every night."""
        hotel = str(booking["hotel"])
        fits = all(self.expected[hotel][n] + p <= self.capacity[hotel].get(n, float("inf")) for n in nights_of(booking))
        if booking["nightly_rate"] > 0 and fits:
            self.book(booking, p)
            return True
        return False


# ---------------------------------------------------------------------------------------------
# 2. Threshold: normal approximation of the number of shows, accept if P(full) < Q_MAX
# ---------------------------------------------------------------------------------------------
class Threshold(Solution):
    Q_MAX = 1 / (1 + WALK_COST)  # break-even overflow probability
    K_SIGMA = 1.0  # >1 inflates the spread to account for correlated cancellations

    def __init__(self, *args, **kwargs):
        self.var = defaultdict(lambda: defaultdict(float))  # hotel -> night -> sum p(1-p)
        super().__init__(*args, **kwargs)

    def book(self, booking, p):
        super().book(booking, p)
        hotel = str(booking["hotel"])
        for night in nights_of(booking):
            self.var[hotel][night] += p * (1 - p)

    def q(self, booking):
        """P(shows among already-accepted guests >= free rooms), maximised over the stay's nights."""
        hotel = str(booking["hotel"])
        worst = 0.0
        for night in nights_of(booking):
            cap = self.capacity[hotel].get(night, float("inf"))
            if cap == float("inf"):
                continue
            mu = self.expected[hotel][night]
            sd = self.K_SIGMA * math.sqrt(self.var[hotel][night])
            if sd == 0:
                q = 1.0 if mu >= cap - 0.5 else 0.0
            else:
                q = 1 - phi((cap - 0.5 - mu) / sd)  # continuity correction: N >= cap
            worst = max(worst, q)
        return worst

    def accept(self, booking, p):
        if booking["nightly_rate"] > 0 and self.q(booking) < self.Q_MAX:
            self.book(booking, p)
            return True
        return False


# ---------------------------------------------------------------------------------------------
# 3. MarginalValue: if hotel is full the cheapest arrival is walked, so cost is 2.5 * min(v, w_min)
# ---------------------------------------------------------------------------------------------
class MarginalValue(Threshold):
    def __init__(self, *args, **kwargs):
        self.cheapest = defaultdict(dict)  # hotel -> arrival day -> cheapest booked stay value
        super().__init__(*args, **kwargs)

    def book(self, booking, p):
        super().book(booking, p)
        hotel, day, v = str(booking["hotel"]), booking["arrival_date"], stay_value(booking)
        self.cheapest[hotel][day] = min(self.cheapest[hotel].get(day, v), v)

    def accept(self, booking, p):
        if booking["nightly_rate"] <= 0:
            return False
        v = stay_value(booking)
        w_min = self.cheapest[str(booking["hotel"])].get(booking["arrival_date"], v)
        q = self.q(booking)
        # Room free: +v. Full: either we are walked (-2.5v) or we displace a cheaper guest (+v - 3.5w).
        gain = p * ((1 - q) * v + q * (v - (1 + WALK_COST) * min(v, w_min)))
        if gain > 0:
            self.book(booking, p)
            return True
        return False


# ---------------------------------------------------------------------------------------------
# 4. MonteCarlo: S scenarios of who shows up, optional shared shock per arrival day
# ---------------------------------------------------------------------------------------------
class MonteCarlo(Solution):
    S = 500  # number of scenarios
    SIGMA_DAY = 0.0  # std of the per-arrival-day shock on the logit scale (0 = independent guests)
    SEED = 0

    def __init__(self, *args, **kwargs):
        self.rng = np.random.default_rng(self.SEED)
        self.shows = defaultdict(lambda: defaultdict(lambda: np.zeros(self.S, dtype=np.int32)))
        self.day_shock = {}  # (hotel, arrival day) -> N(0,1)^S, shared by everyone arriving that day
        self._pending = None  # samples drawn in accept(), reused by book() (common random numbers)
        super().__init__(*args, **kwargs)

    def draw(self, booking, p):
        """S show/no-show samples for this booking."""
        u = self.rng.random(self.S)
        if self.SIGMA_DAY <= 0:
            return u < p
        key = (str(booking["hotel"]), booking["arrival_date"])
        if key not in self.day_shock:
            self.day_shock[key] = self.rng.standard_normal(self.S)
        pc = min(max(p, 1e-6), 1 - 1e-6)
        logit = math.log(pc / (1 - pc)) + self.SIGMA_DAY * self.day_shock[key]
        return u < 1 / (1 + np.exp(-logit))

    def book(self, booking, p):
        super().book(booking, p)
        samples, self._pending = (self._pending if self._pending is not None else self.draw(booking, p)), None
        hotel = str(booking["hotel"])
        for night in nights_of(booking):
            self.shows[hotel][night] += samples

    def mc_gain(self, booking, p):
        """Mean over scenarios of the marginal profit of accepting; stores the samples for book()."""
        hotel, v = str(booking["hotel"]), stay_value(booking)
        samples = self.draw(booking, p)
        full = np.zeros(self.S, dtype=bool)
        for night in nights_of(booking):
            cap = self.capacity[hotel].get(night, float("inf"))
            if cap != float("inf"):
                full |= self.shows[hotel][night] >= cap
        self._pending = samples
        return float(np.mean(samples * np.where(full, -WALK_COST * v, v)))

    def accept(self, booking, p):
        if booking["nightly_rate"] > 0 and self.mc_gain(booking, p) > 0:
            self.book(booking, p)
            return True
        self._pending = None
        return False


# ---------------------------------------------------------------------------------------------
# 5. Hybrid: analytic q decides the clear cases, Monte Carlo only the borderline ones
# ---------------------------------------------------------------------------------------------
class Hybrid(MonteCarlo, Threshold):
    LO, HI = 0.1, 0.5

    def accept(self, booking, p):
        if booking["nightly_rate"] <= 0:
            return False
        q = self.q(booking)
        if q > self.HI:
            return False
        if q < self.LO or self.mc_gain(booking, p) > 0:
            self.book(booking, p)
            return True
        self._pending = None
        return False


# ---------------------------------------------------------------------------------------------
# 6./7. ExactPB and BidPrice now live in solution.Solution (risk check + bid prices)
# ---------------------------------------------------------------------------------------------
class ExactPB(Solution):
    """Exact risk check only (bid prices switched off)."""
    THETA = 0.0


class BidPrice(Solution):
    """Exact risk check + bid prices = the submitted Solution."""


def variant(cls, **params):
    """Subclass of cls with some class attributes overridden, e.g. variant(Threshold, Q_MAX=0.2)."""
    name = cls.__name__ + "(" + ", ".join(f"{k}={v:g}" for k, v in params.items()) + ")"
    return type(name, (cls,), params)


STRATEGIES = [
    Baseline,
    variant(Threshold, Q_MAX=0.10),
    variant(Threshold, Q_MAX=0.20),
    Threshold,
    variant(Threshold, Q_MAX=0.20, K_SIGMA=1.5),
    MarginalValue,
    MonteCarlo,
    variant(MonteCarlo, SIGMA_DAY=0.5),
    Hybrid,
]


def setup(train_file=TRAIN_FILE, valid_file=VALID_FILE, test_file=TEST_FILE):
    """Same test setup as solution.py: train rows, commitments, requests, free-room capacity."""
    rows = pd.concat([load(train_file), load(valid_file)])
    test = load(test_file)
    start = test["arrival_date"].min()
    commitments = test[test["booking_date"] < start]
    requests = test[test["booking_date"] >= start].sort_values("booking_date", kind="stable")
    stayed = rows[(rows["arrival_date"] < start) & (rows["cancelled"] == 0)
                  & (rows["arrival_date"] + pd.to_timedelta(rows["nights"], unit="D") > start)]
    capacity = {}
    for hotel, rooms in ROOMS.items():
        free = pd.Series(rooms, index=pd.date_range(start, test["arrival_date"].max()))
        h = stayed[stayed["hotel"] == hotel]
        for arrival, nights in zip(h["arrival_date"], h["nights"]):
            free.loc[start: arrival + pd.Timedelta(days=int(nights) - 1)] -= 1
        capacity[hotel] = free.clip(lower=0)
    return rows, commitments, requests, capacity


if __name__ == "__main__":
    rows, commitments, requests, capacity = setup()
    models = pick(train(rows, features(rows)), per_hotel=False)
    p_requests = None

    results = []
    for cls in STRATEGIES:
        t0 = time.time()
        solution = cls(commitments, capacity, models, history=rows)
        if p_requests is None:
            p_requests = solution.p_show(requests)  # same model for all strategies: predict once
        accepted = [solution.accept(b, p) for b, p in zip(requests.to_dict("records"), p_requests)]
        profit = net_profit(pd.concat([commitments, requests[accepted]]), capacity)
        results.append((profit, cls.__name__, sum(accepted), time.time() - t0))
        print(f"{cls.__name__:<45} accepted {sum(accepted):>6} | profit {profit:>14,.0f} | {time.time() - t0:5.1f}s")

    print(f"\n{MODEL} together, {len(requests)} requests -- ranked:")
    for profit, name, n, _ in sorted(results, reverse=True):
        print(f"  {profit:>14,.0f}  {name} ({n} accepted)")
