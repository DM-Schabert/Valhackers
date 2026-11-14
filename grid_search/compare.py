"""Numbers behind APPROACH.md, all scored with the same replay as solution.py.

    python grid_search/compare.py          train -> valid: reference strategies, preprocessing, model types
    python grid_search/compare.py --test   train + valid -> test: final check of the best grid settings
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_search as g  # noqa: E402
import train as template  # noqa: E402
from solution import NOT_FEATURES, net_profit  # noqa: E402

DATA = g.ROOT / "data"


def score(models, history, period):
    p = g.predict_cancel(models, period)
    profit, accepted, requests = g.replay(models, history, period)
    y = period[g.prep.TARGET]
    return {"net_profit": round(profit), "accepted": f"{accepted} / {requests}",
            "log_loss": round(log_loss(y, p), 3), "auc": round(roc_auc_score(y, p), 3),
            "predicted_cancel": f"{p.mean():.1%}", "actual_cancel": f"{y.mean():.1%}"}


def hindsight(commitments, requests, capacity):
    """Knows who shows up and fills the free rooms with the highest-rate guests first (greedy)."""
    used = {hotel: defaultdict(int) for hotel in capacity}
    shows = commitments[commitments[g.prep.TARGET] == 0]
    for hotel, arrival, nights in shows[["hotel", "arrival_date", "nights"]].itertuples(index=False):
        for night in pd.date_range(arrival, periods=nights):
            used[str(hotel)][night] += 1
    picked = []
    candidates = requests[requests[g.prep.TARGET] == 0].sort_values("nightly_rate", ascending=False)
    for index, hotel, arrival, nights in candidates[["hotel", "arrival_date", "nights"]].itertuples():
        stay = pd.date_range(arrival, periods=nights)
        hotel = str(hotel)
        if all(used[hotel][n] + 1 <= capacity[hotel].get(n, float("inf")) for n in stay):
            for n in stay:
                used[hotel][n] += 1
            picked.append(index)
    return net_profit(pd.concat([commitments, candidates.loc[picked]]), capacity), len(picked)


def references(history, period):
    commitments, requests, capacity = g.setup(history, period)
    best, picked = hindsight(commitments, requests, capacity)
    return [
        {"strategy": "reject every request", "net_profit": round(net_profit(commitments, capacity)),
         "accepted": f"0 / {len(requests)}"},
        {"strategy": "accept every request",
         "net_profit": round(net_profit(pd.concat([commitments, requests]), capacity)),
         "accepted": f"{len(requests)} / {len(requests)}"},
        {"strategy": "hindsight: knows who shows up, highest rates first",
         "net_profit": round(best), "accepted": f"{picked} / {len(requests)}"},
    ]


def template_models(name, rows, features, per_hotel):
    template.MODEL = name
    if not per_hotel:
        return {"all": template.fit(rows, features)}
    return {hotel: template.fit(rows[rows["hotel"] == hotel], features) for hotel in g.ROOMS}


def best_config(name):
    files = list((g.RESULTS / name).glob("*.json"))
    if not files:
        return None
    best = max((json.loads(f.read_text()) for f in files), key=lambda r: r["net_profit"])
    return {k: best[k] for k in g.GRIDS[name]}


def table(rows, title):
    df = pd.DataFrame(rows)
    print(f"\n### {title}\n")
    print("| " + " | ".join(df.columns) + " |")
    print("|" + "---|" * len(df.columns))
    for row in df.itertuples(index=False):
        print("| " + " | ".join(f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)
                                for v in row) + " |")


def develop():
    train, valid = g.prep.load_split("train"), g.prep.load_split("valid")
    table(references(train, valid), "Reference strategies (valid: Jul-Sep 2016)")

    minimal = [pd.read_parquet(DATA / "splits_minimal" / f"{s}.parquet") for s in ("train", "valid")]
    minimal_features = [c for c in minimal[0].select_dtypes(["number", "category"]).columns
                        if c not in NOT_FEATURES]
    pca = [pd.read_parquet(DATA / "splits_pca" / f"{s}.parquet") for s in ("train", "valid")]
    pca_features = json.loads((DATA / "splits_pca" / "features.json").read_text())["features"]
    rows = []
    for label, (tr, va), features in [
        ("minimal (raw columns, IDs as categories)", minimal, minimal_features),
        ("PCA (31 components)", pca, pca_features),
        ("full pipeline (claude-preprocessing)", (train, valid), g.FEATURE_SETS["baseline"]),
        ("full pipeline without country", (train, valid), g.FEATURE_SETS["no_country"]),
    ]:
        rows.append({"preprocessing": label, **score(template_models("logreg", tr, features, False), tr, va)})
    table(rows, "Preprocessing, same model (logistic regression from train.py)")

    rows = []
    for name in ("logreg", "lightgbm", "xgboost", "catboost"):
        for per_hotel in (False, True):
            models = template_models(name, train, g.FEATURE_SETS["baseline"], per_hotel)
            rows.append({"model": name, "per_hotel": per_hotel, **score(models, train, valid)})
    table(rows, "Model types with train.py settings (full pipeline, all features)")


def final():
    train, valid, test = (g.prep.load_split(s) for s in ("train", "valid", "test"))
    history = pd.concat([train, valid])
    table(references(history, test), "Reference strategies (test: Oct-Dec 2016)")

    rows = [{"model": "logreg, train.py settings (current submission)",
             **score(template_models("logreg", history, g.FEATURE_SETS["baseline"], False), history, test)}]
    for name in g.GRIDS:
        cfg = best_config(name)
        if cfg:
            rows.append({"model": f"{name}, best on valid: {cfg}", **score(g.train_models(name, cfg, history), history, test)})
    table(rows, "Final check on test (trained on train + valid)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    final() if parser.parse_args().test else develop()
