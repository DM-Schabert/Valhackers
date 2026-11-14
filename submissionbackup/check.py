"""Local end-to-end check of the submission, run the way evaluate.py runs it.

    python submissionbackup/check.py

Learns from arrivals before 2016-10-01 (calibration on Jul-Sep 2016) and replays Oct-Dec 2016 from train.csv:
commitments (booked before 1 Oct, cancelled column removed) go into Solution(...), every later request goes
through decide() as a raw row in booking order. Prints the net profit; the same setup in
train-template-kit/compare_strategies.py on splits_v2 gave 823,4k for CatBoost isotonic + BidPrice.
"""
import time
from collections import defaultdict

import pandas as pd

from solution import TRAIN_CSV, Solution

START, END = pd.Timestamp("2016-10-01"), pd.Timestamp("2016-12-31")
ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}


def free_rooms(raw):
    """Rooms minus guests who arrived before START, showed up and are still in house."""
    stayed = raw[(raw["arrival_date"] < START) & (raw["cancelled"] == 0)
                 & (raw["arrival_date"] + pd.to_timedelta(raw["nights"], unit="D") > START)]
    capacity = {}
    for hotel, rooms in ROOMS.items():
        free = pd.Series(rooms, index=pd.date_range(START, END))
        h = stayed[stayed["hotel"] == hotel]
        for arrival, nights in zip(h["arrival_date"], h["nights"]):
            free.loc[START: arrival + pd.Timedelta(days=int(nights) - 1)] -= 1
        capacity[hotel] = free.clip(lower=0)
    return capacity


def net_profit(confirmed, capacity):
    """Check in day by day: the most valuable guests get a room, the cheapest are walked (-2.5x)."""
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
            total += values.iloc[:rooms].sum() - 2.5 * values.iloc[rooms:].sum()
    return total


if __name__ == "__main__":
    raw = pd.read_csv(TRAIN_CSV, parse_dates=["arrival_date", "booking_date"])
    window = raw[(raw["arrival_date"] >= START) & (raw["arrival_date"] <= END)]
    commitments = window[window["booking_date"] < START]
    requests = window[window["booking_date"] >= START].sort_values("booking_date", kind="stable")
    capacity = free_rooms(raw)

    t0 = time.time()
    solution = Solution(commitments.drop(columns="cancelled"), capacity, train_end=START)
    print(f"Solution ready in {time.time() - t0:.0f}s, {len(commitments):,} commitments")
    t0 = time.time()
    accepted = [solution.decide(b) for b in requests.drop(columns="cancelled").to_dict("records")]
    print(f"decide() on {len(requests):,} requests in {time.time() - t0:.0f}s, accepted {sum(accepted):,}")

    print(f"net profit Oct-Dec 2016: {net_profit(pd.concat([commitments, requests[accepted]]), capacity):,.0f}")
    print(f"reference reject all:    {net_profit(commitments, capacity):,.0f}")
