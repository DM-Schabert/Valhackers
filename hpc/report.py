"""Turns the search results into the settings to use: hpc/results/best.json and RESULTS.md.

    python hpc/report.py

Share = (net profit - reject all) / (hindsight - reject all), per window: 0% = accepting nothing,
100% = knowing who shows up. Settings are compared by their average share over the windows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy  # noqa: E402
import search  # noqa: E402

SCHEMES = {
    "current": "Current submission: logistic regression, original rule",
    "model_only": "Best model, original rule",
    "one_rule": "Best model + one tuned rule for both hotels",
    "rule_per_hotel": "Best model + one tuned rule per hotel",
}
MODEL_SETTINGS = ["model", "features", "calibration", "half_life", "per_hotel", "C", "bins",
                  "n_estimators", "num_leaves", "min_child_samples", "weight"]


def smooth_alpha(gain):
    """Average each rule with its two alpha neighbours: prefer a plateau over a single lucky value."""
    grid = gain.reshape(gain.shape[:-1] + tuple(len(values) for values in policy.GRID.values()))
    padded = np.concatenate([grid[..., :1, :, :], grid, grid[..., -1:, :, :]], axis=-3)
    return ((padded[..., :-2, :, :] + padded[..., 1:-1, :, :] + padded[..., 2:, :, :]) / 3).reshape(gain.shape)


def pick(scheme, plain, smooth, current, neutral):
    """plain / smooth: share per (setting, hotel, rule), averaged over the windows used for choosing.
    Returns the setting and one rule per hotel."""
    hotels = plain.shape[1]
    if scheme == "current":
        return current, [neutral] * hotels
    if scheme == "model_only":
        return int(plain[:, :, neutral].sum(axis=1).argmax()), [neutral] * hotels
    if scheme == "one_rule":
        both = smooth.sum(axis=1)
        setting, rule = np.unravel_index(both.argmax(), both.shape)
        return int(setting), [int(rule)] * hotels
    setting = int(smooth.max(axis=2).sum(axis=1).argmax())
    return setting, [int(rule) for rule in smooth[setting].argmax(axis=1)]


def share(gain, setting, rules, window):
    return float(sum(gain[setting, window, hotel, rule] for hotel, rule in enumerate(rules)))


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def pct(value):
    return f"{100 * value:.1f}%"


def eur(value):
    return f"{value:,.0f}"


def show(value):
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def main():
    refs = json.loads((search.RESULTS / "references.json").read_text())
    windows, hotels, rule_list = list(refs), list(policy.ROOMS), policy.rules()
    runs = [json.loads(path.read_text()) for path in sorted(search.RUNS.glob("*.json"))]
    runs = [run for run in runs if set(run["windows"]) == set(windows)]
    keys = [search.key(run["cfg"]) for run in runs]
    neutral = rule_list.index(policy.NEUTRAL)
    current = keys.index(search.key(search.CURRENT)) if search.key(search.CURRENT) in keys else None
    schemes = [name for name in SCHEMES if name != "current" or current is not None]

    profit = np.array([[[run["windows"][w]["profit"][h] for h in hotels] for w in windows] for run in runs], dtype=float)
    floor = np.array([[refs[w][h]["reject_all"] for h in hotels] for w in windows])
    ceiling = np.array([[refs[w][h]["hindsight"] for h in hotels] for w in windows])
    room = np.maximum((ceiling - floor).sum(axis=1), 1.0)
    gain = (profit - floor[None, :, :, None]) / room[None, :, None, None]
    smooth = smooth_alpha(gain)

    held_out = {}
    for scheme in schemes:
        held_out[scheme] = []
        for left_out in range(len(windows)):
            others = [w for w in range(len(windows)) if w != left_out]
            setting, rules = pick(scheme, gain[:, others].mean(axis=1), smooth[:, others].mean(axis=1), current, neutral)
            held_out[scheme].append(share(gain, setting, rules, left_out))
    chosen = max(schemes, key=lambda name: np.mean(held_out[name]))
    plain_all, smooth_all = gain.mean(axis=1), smooth.mean(axis=1)
    setting, rules = pick(chosen, plain_all, smooth_all, current, neutral)

    def total(setting, rules, window):
        return float(sum(profit[setting, window, hotel, rule] for hotel, rule in enumerate(rules)))

    best = {
        "scheme": chosen,
        "model": runs[setting]["cfg"],
        "rules": {hotel: rule_list[rule] for hotel, rule in zip(hotels, rules)},
        "share": {w: share(gain, setting, rules, i) for i, w in enumerate(windows)},
        "profit": {w: total(setting, rules, i) for i, w in enumerate(windows)},
        "held_out_share": float(np.mean(held_out[chosen])),
        "settings_searched": len(runs),
    }
    (search.RESULTS / "best.json").write_text(json.dumps(best, indent=2))

    out = ["# Search results\n",
           f"Generated by `hpc/report.py` from {len(runs)} of {len(search.configs())} model settings, "
           f"each replayed under {len(rule_list)} accept rules on {len(windows)} time windows.\n",
           "**Share** = (net profit − reject all) / (hindsight − reject all): 0% = accept nothing, "
           "100% = knowing who shows up. Settings are compared by the average share over the windows.\n"]

    out.append("## 1. Settings to use (`best.json`)\n")
    out.append(f"Chosen scheme: **{SCHEMES[chosen]}** (the scheme with the best held-out share, section 2).\n")
    out.append(table(["Model setting", "Value"], [[k, f"`{v}`"] for k, v in best["model"].items()]))
    out.append(table(["Accept rule", *hotels], [[name, *[best["rules"][h][name] for h in hotels]] for name in policy.GRID]))

    out.append("## 2. Does the choice hold on a window it has not seen?\n")
    out.append("For each window: choose the settings on the other windows only, then score them on this one.\n")
    out.append(table(["Scheme", *[f"left out: {w}" for w in windows], "average"],
                     [[SCHEMES[s] + (" ← chosen" if s == chosen else ""), *map(pct, held_out[s]), pct(np.mean(held_out[s]))]
                      for s in schemes]))

    out.append("## 3. Net profit per window\n")
    header = ["Window", "Reject all", "Accept all", "First come, knowing who shows", "Hindsight"]
    if current is not None:
        header += ["Current submission", "share"]
    header += ["Chosen settings", "share"]
    rows = []
    for i, w in enumerate(windows):
        row = [w] + [eur(sum(refs[w][h][name] for h in hotels)) for name in ("reject_all", "accept_all", "first_come", "hindsight")]
        if current is not None:
            row += [eur(total(current, [neutral] * len(hotels), i)), pct(share(gain, current, [neutral] * len(hotels), i))]
        rows.append(row + [eur(best["profit"][w]), pct(best["share"][w])])
    out.append(table(header, rows))
    out.append("The chosen settings were picked on these same windows, so this table is optimistic; "
               "section 2 is the honest estimate.\n")

    frame = pd.DataFrame([run["cfg"] for run in runs])
    frame["original"] = plain_all[:, :, neutral].sum(axis=1)
    frame["tuned"] = smooth_all.max(axis=2).sum(axis=1)
    for name in ("log_loss", "auc"):
        frame[name] = [np.mean([run["windows"][w][name] for w in windows]) for run in runs]
    frame["gap"] = [np.mean([run["windows"][w]["predicted_cancel"] - run["windows"][w]["actual_cancel"] for w in windows])
                    for run in runs]
    in_grid = frame[[k in {search.key(cfg) for cfg in search.grid()} for k in keys]]

    out.append("## 4. Which model settings matter\n")
    out.append("Average over all grid settings that share the value. *Original rule* = alpha 1, z 0, beta 0; "
               "*tuned rule* = best rule per hotel. *Gap* = predicted − actual cancellation rate.\n")
    for name in MODEL_SETTINGS:
        if name not in in_grid or in_grid[name].nunique(dropna=True) < 2:
            continue
        groups = in_grid.dropna(subset=[name]).groupby(name, sort=True)
        rows = [[show(value), len(part), pct(part["original"].mean()), pct(part["tuned"].mean()), pct(part["tuned"].max()),
                 f"{part['log_loss'].mean():.3f}", f"{100 * part['gap'].mean():+.1f} pp"] for value, part in groups]
        out.append(table([f"`{name}`", "settings", "share, original rule", "share, tuned rule", "best", "log-loss", "gap"], rows))

    out.append("## 5. Feature groups: what happens when one is removed\n")
    out.append("Change in share (percentage points) against all features, same model settings. Negative = the group helps.\n")
    for base in search.ABLATION_BASES:
        if search.key(base) not in keys:
            continue
        reference = frame.iloc[keys.index(search.key(base))]
        rows = []
        for name in list(search.FEATURE_SETS)[1:] + list(search.ABLATIONS):
            k = search.key({**base, "features": name})
            if k in keys:
                row = frame.iloc[keys.index(k)]
                rows.append([name, ", ".join(search.DROPPED[name]), f"{100 * (row['original'] - reference['original']):+.1f}",
                             f"{100 * (row['tuned'] - reference['tuned']):+.1f}", f"{row['log_loss'] - reference['log_loss']:+.3f}"])
        out.append(f"**{base['model']}** (all features: {pct(reference['original'])} original rule, "
                   f"{pct(reference['tuned'])} tuned rule, log-loss {reference['log_loss']:.3f})\n")
        out.append(table(["Removed", "Features", "original rule", "tuned rule", "log-loss"], rows))

    out.append("## 6. The accept rule for the chosen model\n")
    out.append("Best share each hotel contributes for every value of one rule setting (the other two at their best). "
               "A flat row of similar values means the choice is not sensitive.\n")
    for name, values in policy.GRID.items():
        rows = []
        for value in values:
            members = [i for i, rule in enumerate(rule_list) if rule[name] == value]
            rows.append([value, *[pct(plain_all[setting, h, members].max()) for h in range(len(hotels))]])
        out.append(table([f"`{name}`", *hotels], rows))

    out.append("## 7. Top 15 model settings\n")
    top = frame.sort_values("tuned", ascending=False).head(15)
    names = [name for name in MODEL_SETTINGS if name in frame]
    rows = [[*["" if pd.isna(row[n]) else show(row[n]) for n in names], pct(row["original"]), pct(row["tuned"]),
             f"{row['log_loss']:.3f}", f"{row['auc']:.3f}", f"{100 * row['gap']:+.1f} pp"] for _, row in top.iterrows()]
    out.append(table([*names, "original rule", "tuned rule", "log-loss", "AUC", "gap"], rows))

    (search.RESULTS / "RESULTS.md").write_text("\n".join(out))
    print(f"{len(runs)} settings -> {search.RESULTS / 'best.json'}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
