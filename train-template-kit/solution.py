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

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # train.py is next to this file, prep.py in the repo root
import prep  # noqa: E402
from train import train  # noqa: E402

# =============================================================================================
# INPUT FILES (preprocessed .parquet or .csv with the train.csv columns + features)
# =============================================================================================
DATA = HERE.parent / "data"
TRAIN_FILE = DATA / "splits" / "train.parquet"
VALID_FILE = DATA / "splits" / "valid.parquet"
TEST_FILE = DATA / "splits" / "test.parquet"
PARAMS_FILE = DATA / "prep_params_dev.json"  # optional: feature list; evaluate.py needs it for raw bookings
# =============================================================================================

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
NOT_FEATURES = ["cancelled", "booking_id", "arrival_year", "arrival_date", "booking_date",
                "split", "is_commitment", "use_for_occupancy", "stay_value"]
PARAMS = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else None


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


class Solution:
    def __init__(self, commitments, capacity, model=None):
        """evaluate.py calls Solution(commitments, capacity): the model then trains on all three files."""
        if model is None:
            rows = pd.concat([load(TRAIN_FILE), load(VALID_FILE), load(TEST_FILE)])
            model = train(rows, features(rows))
        self.model = model
        self.capacity = capacity
        self.expected = {hotel: defaultdict(float) for hotel in capacity}  # night -> expected guests
        self.counter = prep.RequestCounter()

        if not set(model.feature_name_) <= set(commitments.columns):  # raw rows (evaluate.py)
            commitments = prep.prepare(commitments, PARAMS)
        for booking, p in zip(commitments.to_dict("records"), self.p_show(commitments)):
            self.book(booking, p)

    def decide(self, booking):
        """evaluate.py: one raw booking -> preprocess (needs PARAMS_FILE) -> accept or reject."""
        x = prep.prepare(booking, PARAMS, counter=self.counter)
        return self.accept(booking, self.p_show(x)[0])

    def p_show(self, x):
        """Probability that the guest shows up."""
        return 1 - self.model.predict_proba(x[self.model.feature_name_])[:, 1]

    def accept(self, booking, p):
        """Accept if it pays and the expected guests still fit into the free rooms on every night."""
        hotel = str(booking["hotel"])
        nights = pd.date_range(booking["arrival_date"], periods=booking["nights"])
        fits = all(self.expected[hotel][n] + p <= self.capacity[hotel].get(n, float("inf")) for n in nights)
        if booking["nightly_rate"] > 0 and fits:
            self.book(booking, p)
            return True
        return False

    def book(self, booking, p):
        for night in pd.date_range(booking["arrival_date"], periods=booking["nights"]):
            self.expected[str(booking["hotel"])][night] += p


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
    model = train(rows, features(rows))

    test = load(TEST_FILE)
    start = test["arrival_date"].min()
    commitments = test[test["booking_date"] < start]  # booked before the test period: already confirmed
    requests = test[test["booking_date"] >= start].sort_values("booking_date", kind="stable")
    capacity = {hotel: pd.Series(rooms, index=pd.date_range(start, test["arrival_date"].max()))
                for hotel, rooms in ROOMS.items()}

    solution = Solution(commitments, capacity, model)
    accepted = [solution.accept(b, p) for b, p in zip(requests.to_dict("records"), solution.p_show(requests))]
 
    profit = net_profit(pd.concat([commitments, requests[accepted]]), capacity)
    print(f"accepted {sum(accepted)} of {len(requests)} requests | net profit on test: {profit:,.0f}")
