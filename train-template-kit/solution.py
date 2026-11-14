"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.
"""
import sys
from pathlib import Path

import pandas as pd

# prep.py lives in the repo root; resolve it relative to this file so the cwd doesn't matter
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import prep  # noqa: E402  (shared cleaning + features, see cleaning_steps.md)

PARAMS_PATH = REPO_ROOT / "data" / "prep_params.json"


class Solution:
    def __init__(self, commitments: pd.DataFrame, capacity: dict[str, pd.Series]) -> None:
        """Called once, before 2017 starts.

        commitments  2017 arrivals booked before 2017-01-01. The hotel already confirmed them, so
                     decide() is never called for them. Same columns as train.csv, except:
                     - no cancelled column;
                     - arrival_date and booking_date are already pandas Timestamps, not strings.

        capacity     hotel -> rooms free on each night of 2017 (indexed by date): the rooms minus
                     guests who arrived in 2016 and are still in house.
        """
        if not PARAMS_PATH.exists():
            raise FileNotFoundError(f"{PARAMS_PATH} missing: run preprocessing.ipynb (section 3) first")
        self.params = prep.load_params(PARAMS_PATH)
        self.counter = prep.RequestCounter()  # running state for the same_request_count feature
        self.capacity = capacity
        self.commitments = prep.prepare(commitments, self.params)  # cleaned + features, for later use

    def decide(self, booking: dict) -> bool:
        """Called once per 2017 booking request, in booking-date order. `booking` is one row with
        the same columns as commitments, as {column: value}. Return True to confirm the booking,
        False to reject it."""
        # Must run for EVERY request (also ones we reject quickly), otherwise the counter drifts.
        x = prep.prepare(booking, self.params, counter=self.counter)[prep.FEATURES]

        # TODO: p_show = model on x; accept if expected value beats expected walk cost.
        return True  # placeholder = "accept everything" baseline
