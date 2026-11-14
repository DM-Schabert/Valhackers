"""Grid test: accept strategies x probability sources, net profit on TEST_FILE.

Probability sources: each model in train.py, raw (fit on train+valid) and isotonic-calibrated
(fit on train, calibrated on valid), plus an oracle that knows who cancels (diagnostic ceiling only).

Run: python train-template-kit/compare_strategies.py   -> prints tables, writes strategy_results.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import accept_strategies as A
import train as T
from solution import TRAIN_FILE, VALID_FILE, features, load, net_profit

MODELS = ["lightgbm", "catboost", "xgboost", "logreg"]
STRATEGIES = [
    A.Baseline,
    A.Threshold,
    A.ExactPB,
    A.MonteCarlo,
    A.BidPrice,
    A.variant(A.BidPrice, THETA=0.5),
]


class Fixed:
    """Model stand-in that returns a precomputed P(cancel) stored in column _pc."""
    used_features = ["_pc"]

    def predict_proba(self, X):
        c = X["_pc"].to_numpy(dtype=float)
        return np.c_[1 - c, c]


def probability_sources(rows, tr, va, com, req):
    feats = features(rows)
    out = {}
    for name in MODELS:
        t0 = time.time()
        T.MODEL = name
        try:
            full = T.fit(rows, feats)
        except Exception as e:
            print(f"  skipped {name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        out[f"{name} raw"] = (full.predict_proba(com[feats])[:, 1], full.predict_proba(req[feats])[:, 1])
        part = T.fit(tr, feats)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(part.predict_proba(va[feats])[:, 1], va["cancelled"])
        out[f"{name} isotonic"] = (iso.predict(part.predict_proba(com[feats])[:, 1]),
                                   iso.predict(part.predict_proba(req[feats])[:, 1]))
        print(f"  trained {name} ({time.time() - t0:.0f}s)")
    out["oracle (diagnostic)"] = (com["cancelled"].to_numpy(float), req["cancelled"].to_numpy(float))
    return out


if __name__ == "__main__":
    rows, com, req, cap = A.setup()
    tr, va = load(TRAIN_FILE), load(VALID_FILE)
    print("training probability sources ...")
    sources = probability_sources(rows, tr, va, com, req)

    calib, results = [], []
    for src, (pc_com, pc_req) in sources.items():
        show = 1 - req["cancelled"].to_numpy()
        calib.append({"source": src, "pred show (requests)": 1 - pc_req.mean(), "actual": show.mean(),
                      "pred show (commitments)": 1 - pc_com.mean(), "actual ": 1 - com["cancelled"].mean(),
                      "brier (requests)": np.mean((pc_req - req["cancelled"].to_numpy()) ** 2)})
        com2, req2 = com.assign(_pc=pc_com), req.assign(_pc=pc_req)
        recs = req2.to_dict("records")
        for cls in STRATEGIES:
            t0 = time.time()
            s = cls(com2, cap, {"all": Fixed()})
            P = 1 - pc_req
            accepted = [s.accept(b, p) for b, p in zip(recs, P)]
            profit = net_profit(pd.concat([com, req[accepted]]), cap)
            results.append({"source": src, "strategy": cls.__name__, "accepted": sum(accepted), "profit": profit})
            print(f"  {src:<22} {cls.__name__:<22} accepted {sum(accepted):>5}  profit {profit:>12,.0f}  ({time.time() - t0:.1f}s)")

    pd.set_option("display.width", 200)
    print("\nCalibration on the test split:")
    print(pd.DataFrame(calib).set_index("source").round(3).to_string())

    res = pd.DataFrame(results)
    table = res.pivot(index="strategy", columns="source", values="profit")
    table = table[[s for s in sources]].reindex([c.__name__ for c in STRATEGIES])
    print("\nNet profit on test (rows = strategy, columns = probability source):")
    print((table / 1000).round(1).to_string(float_format=lambda x: f"{x:,.1f}k"))
    print("\nAccepted requests:")
    print(res.pivot(index="strategy", columns="source", values="accepted")[[s for s in sources]]
          .reindex([c.__name__ for c in STRATEGIES]).to_string())
    out = Path(__file__).with_name("strategy_results.csv")
    res.to_csv(out, index=False)
    print(f"\nsaved {out}")
