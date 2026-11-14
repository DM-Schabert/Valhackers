"""The Solution with the settings found by the search (hpc/results/best.json).

    evaluate(Solution)                        trains on all of train.csv, decides one raw booking at a time
    python hpc/tuned_solution.py --check      replays the last search window through decide() and
                                              compares the net profit with what the search computed
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy  # noqa: E402
import search  # noqa: E402
from search import prep  # noqa: E402

BEST = search.RESULTS / "best.json"


def stay(row):
    return pd.date_range(row["arrival_date"], periods=int(row["nights"]))


class Solution:
    def __init__(self, commitments, capacity, until=None):
        """until: only train on arrivals before this date (for the check; None = all of train.csv)."""
        best = json.loads(BEST.read_text())
        self.params, rows = search.prepared(prep.load_raw(search.RAW), until)
        start = rows["arrival_date"].max() + pd.Timedelta(days=1) if until is None else pd.Timestamp(until)
        self.model = search.Predictor(best["model"], search.history(rows, start), start)
        self.ledgers = {hotel: policy.Ledger(capacity[hotel], best["rules"].get(hotel, policy.NEUTRAL))
                        for hotel in capacity}
        self.counter = prep.RequestCounter()
        booked = prep.prepare(commitments, self.params)
        for row, p in zip(booked.to_dict("records"), 1 - self.model.cancel(booked)):
            self.ledgers[str(row["hotel"])].add(stay(row), p)

    def decide(self, booking):
        x = prep.prepare(booking, self.params, counter=self.counter)
        p = 1 - self.model.cancel(x)[0]
        row = x.iloc[0]
        ledger, nights = self.ledgers[str(row["hotel"])], stay(row)
        if row["nightly_rate"] > 0 and ledger.fits(nights, p, row["rate_rel"]):
            ledger.add(nights, p)
            return True
        return False


def check():
    best = json.loads(BEST.read_text())
    name = list(search.WINDOWS)[-1]
    raw = prep.load_raw(search.RAW)
    window = search.Window(raw, name)
    replay = window.replay
    commitments, requests = raw.loc[replay.commitments.index], raw.loc[replay.requests.index]

    solution = Solution(commitments, replay.capacity, until=window.start)
    started = time.time()
    accepted = [solution.decide(booking) for booking in requests.to_dict("records")]
    seconds = time.time() - started
    online = policy.net_profit(pd.concat([commitments, requests[accepted]]), replay.capacity)
    searched = best["profit"][name]
    lines = [
        "## 8. Check: the Solution class reproduces the search\n",
        f"Window `{name}` replayed through `hpc/tuned_solution.py`, one raw booking at a time, "
        "scored with the original `net_profit`:\n",
        f"- net profit through `decide()`: **{online:,.0f}** ({sum(accepted):,} of {len(requests):,} requests accepted)",
        f"- net profit the search computed for the same settings: **{searched:,.0f}**",
        f"- difference: {abs(online - searched):,.0f}",
        f"- `decide()` takes {1000 * seconds / max(len(requests), 1):.1f} ms per booking\n",
    ]
    print("\n".join(lines))
    with open(search.RESULTS / "RESULTS.md", "a") as report:
        report.write("\n" + "\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    check()
