"""End-to-end check of the submission, run the way evaluate.py runs it. Trains a model: run it on the HPC.

    bash hpc/hpc.sh check          (or here: python submission/check.py)

Learns from arrivals before 2016-10-01 and replays Oct-Dec 2016 from train.csv: commitments (booked before
1 Oct, cancelled column removed) go into Solution(...), every later request goes through decide() as a raw
booking dict, in booking order. The net profit should equal what the HPC search computed for the same
window (hpc/results/best.json, "oct-dec"): 860,521 with 2,610 of 5,457 requests accepted. The check runs
with the pinned versions of submission/requirements.txt; if those differ from the search's, the models
can come out very slightly different, so a small difference is possible, a large one is a bug.
This check always trains; model.joblib is tested by export_model.py.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

import prep
import solution
from solution import Solution

START, END = pd.Timestamp("2016-10-01"), pd.Timestamp("2017-01-01")
ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
BEST = Path(__file__).resolve().parent.parent / "hpc" / "results" / "best.json"
SEARCH_RESULT = 860521


def free_rooms(history, period):
    """Rooms minus guests who arrived before START, showed up and are still in house (as in the search)."""
    stayed = history[(history["arrival_date"] < START) & (history["cancelled"] == 0)
                     & (history["arrival_date"] + pd.to_timedelta(history["nights"], unit="D") > START)]
    capacity = {}
    for hotel, rooms in ROOMS.items():
        free = pd.Series(rooms, index=pd.date_range(START, period["arrival_date"].max()))
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


def settings_match():
    """The settings written into solution.py against the search's best.json, if it is there."""
    if not BEST.exists():
        return "hpc/results/best.json not found, settings not compared"
    best = json.loads(BEST.read_text())
    model = {"model": "blend", "weight": solution.LOGREG_WEIGHT, "per_hotel": False, "features": "no_calendar",
             "half_life": 0, "calibration": "none"}
    rule = {"alpha": solution.ALPHA, "z": solution.Z, "beta": solution.BETA}
    if best["model"] == model and all(r == rule for r in best["rules"].values()):
        return "settings in solution.py match hpc/results/best.json"
    return f"MISMATCH: best.json has {best['model']} {best['rules']}"


if __name__ == "__main__":
    print(settings_match())
    print(f"package versions: {solution.fingerprint()['versions']}")
    raw = prep.load_raw(solution.train_csv())
    window = raw[(raw["arrival_date"] >= START) & (raw["arrival_date"] < END)]
    commitments = window[window["booking_date"] < START]
    requests = window[window["booking_date"] >= START].sort_values("booking_date", kind="stable")
    history = raw[(raw["arrival_date"] < START) & ~prep._bad_row_mask(raw)]
    capacity = free_rooms(history, window)

    t0 = time.time()
    s = Solution(commitments.drop(columns="cancelled"), capacity, train_end=START)
    print(f"Solution ready in {time.time() - t0:.0f}s, {len(commitments):,} commitments")
    t0 = time.time()
    accepted = [s.decide(b) for b in requests.drop(columns="cancelled").to_dict("records")]
    seconds = time.time() - t0
    print(f"decide() on {len(requests):,} requests in {seconds:.0f}s ({1000 * seconds / len(requests):.1f} ms each), "
          f"accepted {sum(accepted):,}")
    profit = net_profit(pd.concat([commitments, requests[accepted]]), capacity)
    print(f"net profit Oct-Dec 2016: {profit:,.0f}")
    verdict = "MATCH" if round(profit) == SEARCH_RESULT else f"DIFFERENT by {profit - SEARCH_RESULT:+,.0f}"
    print(f"HPC search, same window: {SEARCH_RESULT:,}  ->  {verdict}")
    print(f"reference, reject all:   {net_profit(commitments, capacity):,.0f}")
