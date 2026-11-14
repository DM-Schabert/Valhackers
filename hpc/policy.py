"""The accept rule and the replay that scores it.

A request is accepted if it pays and, on every night of its stay,

    expected guests + p + z * spread  <=  free rooms * (alpha + beta * price_tilt)

p = P(guest shows up), spread = standard deviation of the number of guests who show up,
price_tilt = nightly rate / normal rate of that hotel and month - 1.
alpha = 1, z = 0, beta = 0 is the rule in train-template-kit/solution.py.

Ledger applies one rule to one booking at a time (the Solution).
Replay applies every rule of GRID in one pass over a time window (the search).
"""
import itertools
import math
from collections import defaultdict

import numpy as np
import pandas as pd

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
WALK_PENALTY = 2.5
NEUTRAL = {"alpha": 1.0, "z": 0.0, "beta": 0.0}
GRID = {
    "alpha": [round(0.85 + 0.025 * i, 3) for i in range(21)],
    "z": [0.0, 0.5, 1.0],
    "beta": [0.0, 0.1, 0.2, 0.4],
}


def rules():
    return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def price_tilt(rate_rel):
    return np.clip(np.nan_to_num(np.asarray(rate_rel, dtype=float), nan=1.0) - 1.0, -0.5, 1.0)


class Ledger:
    """Guests expected per night in one hotel, under one rule."""

    def __init__(self, capacity, rule):
        self.capacity, self.rule = capacity, rule
        self.mean, self.var = defaultdict(float), defaultdict(float)

    def fits(self, nights, p, rate_rel):
        factor = self.rule["alpha"] + self.rule["beta"] * float(price_tilt(rate_rel))
        spread = p * (1 - p)
        return all(self.mean[n] + p + self.rule["z"] * math.sqrt(self.var[n] + spread)
                   <= self.capacity.get(n, math.inf) * factor for n in nights)

    def add(self, nights, p):
        for n in nights:
            self.mean[n] += p
            self.var[n] += p * (1 - p)


def split_period(period, start):
    """Booked before the period starts = confirmed commitments; the rest = requests in booking order."""
    commitments = period[period["booking_date"] < start]
    requests = period[period["booking_date"] >= start].sort_values("booking_date", kind="stable")
    return commitments, requests


def capacity(history, period, start):
    """Free rooms per night as in solution.py: rooms minus earlier guests who are still in house."""
    stayed = history[(history["arrival_date"] < start) & (history["cancelled"] == 0)
                     & (history["arrival_date"] + pd.to_timedelta(history["nights"], unit="D") > start)]
    free_rooms = {}
    for hotel, rooms in ROOMS.items():
        free = pd.Series(rooms, index=pd.date_range(start, period["arrival_date"].max()))
        guests = stayed[stayed["hotel"] == hotel]
        for arrival, nights in zip(guests["arrival_date"], guests["nights"]):
            free.loc[start: arrival + pd.Timedelta(days=int(nights) - 1)] -= 1
        free_rooms[hotel] = free.clip(lower=0)
    return free_rooms


def net_profit(confirmed, capacity):
    """The scoring rule, copied from train-template-kit/solution.py (slow, exact)."""
    shows = confirmed[confirmed["cancelled"] == 0]
    total = 0.0
    for hotel, free in capacity.items():
        taken = defaultdict(int)
        for day, guests in shows[shows["hotel"] == hotel].groupby("arrival_date"):
            values = (guests["nightly_rate"] * guests["nights"]).sort_values(ascending=False)
            rooms = max(int(free.get(day, ROOMS[hotel])) - taken[day], 0)
            for nights in guests.loc[values.index[:rooms], "nights"]:
                for night in pd.date_range(day, periods=nights):
                    taken[night] += 1
            total += values.iloc[:rooms].sum() - WALK_PENALTY * values.iloc[rooms:].sum()
    return total


class Replay:
    """One time window as arrays. rows = commitments, then requests in booking order."""

    def __init__(self, history, period, start):
        self.commitments, self.requests = split_period(period, start)
        self.capacity = capacity(history, period, start)
        self.rows = pd.concat([self.commitments, self.requests])
        hotel = self.rows["hotel"].astype(str).to_numpy()
        is_request = np.arange(len(self.rows)) >= len(self.commitments)
        self.where = {name: np.flatnonzero(hotel == name) for name in ROOMS}
        self.hotels = {name: _Hotel(self.rows.iloc[at], is_request[at], self.capacity[name], start)
                       for name, at in self.where.items()}

    def score(self, p_show, rule_list):
        """Net profit per hotel for every rule in rule_list: {hotel: array}."""
        alpha, z, beta = (np.array([rule[k] for rule in rule_list]) for k in GRID)
        return {name: part.profit(part.decide(p_show[self.where[name]], alpha, z, beta))
                for name, part in self.hotels.items()}

    def references(self):
        """Per hotel: reject all, accept all, and two strategies that know who shows up."""
        out = {}
        for name, part in self.hotels.items():
            n = len(part.day) - part.n_commitments
            masks = np.array([np.zeros(n, bool), np.ones(n, bool), part.oracle(False), part.oracle(True)])
            profit = part.profit(masks)
            out[name] = {"reject_all": float(profit[0]), "accept_all": float(profit[1]),
                         "first_come": float(profit[2]), "hindsight": float(profit[3]),
                         "requests": n, "first_come_accepted": int(masks[2].sum()),
                         "hindsight_accepted": int(masks[3].sum())}
        return out

    def confirmed(self, accepted):
        """Commitments plus the accepted requests ({hotel: mask over that hotel's requests}) as rows."""
        keep = np.ones(len(self.rows), bool)
        for name, part in self.hotels.items():
            keep[self.where[name][part.n_commitments:]] = accepted[name]
        return self.rows[keep]


class _Hotel:
    def __init__(self, rows, is_request, free, start):
        self.day = (rows["arrival_date"] - start).dt.days.to_numpy()
        self.nights = rows["nights"].to_numpy(dtype=int)
        self.rate = rows["nightly_rate"].to_numpy(dtype=float)
        self.shows = rows["cancelled"].to_numpy() == 0
        self.tilt = price_tilt(rows["rate_rel"])
        self.n_commitments = int((~is_request).sum())
        self.rooms = free.to_numpy().astype(int)
        self.length = max(int((self.day + self.nights).max()) + 1, len(self.rooms))
        self.cap = np.full(self.length, np.inf)
        self.cap[:len(self.rooms)] = self.rooms

        value = self.rate * self.nights
        shows = np.flatnonzero(self.shows)
        order = shows[np.lexsort((-value[shows], self.day[shows]))]
        self.s_value, self.s_nights = value[order], self.nights[order]
        self.s_from_request = order >= self.n_commitments
        self.s_request = order[self.s_from_request] - self.n_commitments
        days, first = np.unique(self.day[order], return_index=True)
        last = np.append(first[1:], len(order))
        self.arrivals = []
        for day, lo, hi in zip(days.tolist(), first.tolist(), last.tolist()):
            stays = [(int(n), np.flatnonzero(self.s_nights[lo:hi] == n))
                     for n in np.unique(self.s_nights[lo:hi]) if n > 0]
            self.arrivals.append((day, lo, hi, stays))

    def decide(self, p, alpha, z, beta):
        """accepted[rule, request]: the requests go through in booking order, all rules side by side."""
        c = self.n_commitments
        spread = p * (1 - p)
        mean, var = np.zeros(self.length), np.zeros(self.length)
        for day, nights, q, s in zip(self.day[:c].tolist(), self.nights[:c].tolist(),
                                     p[:c].tolist(), spread[:c].tolist()):
            mean[day:day + nights] += q
            var[day:day + nights] += s
        mean, var = np.tile(mean, (len(alpha), 1)), np.tile(var, (len(alpha), 1))
        factor = alpha[None, :] + self.tilt[c:, None] * beta[None, :]
        z = z[:, None]
        accepted = np.zeros((len(alpha), len(p) - c), bool)
        requests = zip(self.day[c:].tolist(), self.nights[c:].tolist(), p[c:].tolist(),
                       spread[c:].tolist(), (self.rate[c:] > 0).tolist())
        for i, (day, nights, q, s, pays) in enumerate(requests):
            if not pays:
                continue
            stay = slice(day, day + nights)
            need = mean[:, stay] + q + z * np.sqrt(var[:, stay] + s)
            ok = (need <= self.cap[stay] * factor[i][:, None]).all(axis=1)
            mean[:, stay] += ok[:, None] * q
            var[:, stay] += ok[:, None] * s
            accepted[:, i] = ok
        return accepted

    def profit(self, accepted):
        """Net profit per rule: check-in day by day, most valuable guests first, the rest are walked."""
        n_rules = len(accepted)
        confirmed = np.ones((n_rules, len(self.s_value)), bool)
        confirmed[:, self.s_from_request] = accepted[:, self.s_request]
        total, taken = np.zeros(n_rules), np.zeros(n_rules)
        leaving = np.zeros((n_rules, self.length + 1))
        previous = -1
        for day, lo, hi, stays in self.arrivals:
            taken -= leaving[:, previous + 1:day + 1].sum(axis=1)
            previous = day
            here = confirmed[:, lo:hi]
            rooms = np.maximum(self.rooms[day] - taken, 0)
            seated = here & (np.cumsum(here, axis=1) <= rooms[:, None])
            value = self.s_value[lo:hi]
            total += (1 + WALK_PENALTY) * (seated @ value) - WALK_PENALTY * (here @ value)
            for nights, columns in stays:
                count = seated[:, columns].sum(axis=1)
                taken += count
                leaving[:, day + nights] += count
        return total

    def oracle(self, by_rate):
        """Requests picked knowing who shows up: highest rates first, or first come first served."""
        c = self.n_commitments
        used = np.zeros(self.length)
        for j in np.flatnonzero(self.shows[:c]):
            used[self.day[j]:self.day[j] + self.nights[j]] += 1
        candidates = np.flatnonzero(self.shows[c:] & (self.rate[c:] > 0))
        if by_rate:
            candidates = candidates[np.argsort(-self.rate[c:][candidates], kind="stable")]
        picked = np.zeros(len(self.day) - c, bool)
        for i in candidates:
            stay = slice(self.day[c + i], self.day[c + i] + self.nights[c + i])
            if (used[stay] + 1 <= self.cap[stay]).all():
                used[stay] += 1
                picked[i] = True
        return picked
