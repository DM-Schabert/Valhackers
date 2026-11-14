"""Trains the final model on all of train.csv and saves it as submission/model.joblib, so evaluate() doesn't
have to train. This trains a model: run it on the HPC, with the versions of submission/requirements.txt.

    bash hpc/hpc.sh export                       train + save on the cluster (log: hpc/logs/model.out)
    bash hpc/hpc.sh fetch                        copies model.joblib back into submission/
    python submission/export_model.py --status   no training: can this machine use model.joblib?
"""
import argparse
import hashlib
import time
from datetime import datetime, timezone

import joblib
import numpy as np

import prep
import solution


def status():
    params, model, problem = solution.load_saved()
    if problem:
        print(f"NOT usable here: {problem}. Solution() would train from train.csv instead.")
        return
    saved = joblib.load(solution.MODEL_FILE)
    print(f"usable here: {solution.MODEL_FILE} (trained {saved['trained']} on {saved['rows']:,} bookings, "
          f"train.csv md5 {saved['train_csv_md5'][:12]}, versions {saved['fingerprint']['versions']})")


def export():
    started = time.time()
    params, model = solution.fit_model()
    print(f"trained in {time.time() - started:.0f}s")
    csv = solution.train_csv()
    rows = prep.prepare(prep.load_raw(csv), params)
    joblib.dump({
        "params": params, "logreg": model.logreg, "lightgbm": model.lightgbm,
        "fingerprint": solution.fingerprint(),
        "rows": int((~prep._bad_row_mask(rows)).sum()),
        "train_csv_md5": hashlib.md5(csv.read_bytes()).hexdigest(),
        "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, solution.MODEL_FILE, compress=3)

    loaded_params, loaded, problem = solution.load_saved()
    if problem:
        raise SystemExit(f"FAILED: {problem}")
    if loaded_params != params:
        raise SystemExit("FAILED: the saved preprocessing parameters differ")
    difference = float(np.abs(loaded.cancel(rows) - model.cancel(rows)).max())
    if difference != 0.0:
        raise SystemExit(f"FAILED: the saved model predicts differently (largest difference {difference:g})")
    size = solution.MODEL_FILE.stat().st_size / 1e6
    print(f"saved {solution.MODEL_FILE} ({size:.1f} MB), versions {solution.fingerprint()['versions']}")
    print(f"reloaded and checked: identical P(cancel) for all {len(rows):,} bookings of train.csv -> OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    status() if parser.parse_args().status else export()
