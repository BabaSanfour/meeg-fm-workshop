"""Repeat the fine-tuning runs with other random seeds, to see how much a result depends on chance.

Run after precompute_finetune, from the repository folder: python -m meeg_fm.precompute_finetune_seeds (about 35 minutes on an Apple GPU).
Adds rows to finetune_history.csv and finetune_predictions.csv, with a `seed` column.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import meeg_fm as mf
from meeg_fm.finetune import finetune_reve, predict_proba

OUT = Path(mf.__file__).parent / "precomputed"
TWO_STAGES = dict(head_epochs=3, epochs=10, head_lr=1e-4, lr=3e-5, batch_size=32)
ONE_STAGE = dict(head_epochs=0, epochs=10, lr=1e-4, batch_size=32)
FEW_LABELS = dict(head_epochs=10, epochs=40, head_lr=1e-4, lr=3e-5, batch_size=32)   # more passes: 10 times fewer trials
RIGHT = mf.CLASSES.index("right_hand")

mf.use_data_dir()
X, y, meta = mf.load_epochs(mf.SUBJECTS, sfreq=200, fmin=0.5, fmax=99.5)
subject = meta["subject"].to_numpy()
train = (meta["session"] == "0train").to_numpy()
test = np.flatnonzero(~train)

history = pd.read_csv(OUT / "finetune_history.csv")
predictions = pd.read_csv(OUT / "finetune_predictions.csv")
if "seed" not in history:                                   # the first script used seed 0
    history["seed"], predictions["seed"] = 0, 0
rename = {"all participants, large learning rate": "all participants, one stage"}
history["run"], predictions["run"] = history["run"].replace(rename), predictions["run"].replace(rename)
drop = "all participants, 10 % of the labels"               # redone below with more passes
history, predictions = history[history["run"] != drop], predictions[predictions["run"] != drop]


def run(name, seed, fit, held_out, **recipe):
    global history, predictions
    print(f"--- {name}, seed {seed}: {len(fit)} training trials", flush=True)
    model, rows = finetune_reve(X[fit], y[fit], X[held_out], y[held_out], seed=seed, verbose=False, **recipe)
    history = pd.concat([history, pd.DataFrame(rows).assign(run=name, training_trials=len(fit), seed=seed)])
    predictions = pd.concat([predictions, pd.DataFrame({
        "run": name, "trial": test, "subject": subject[test], "label": y[test],
        "p_right": predict_proba(model, X[test])[:, RIGHT], "seed": seed})])
    history.round(4).to_csv(OUT / "finetune_history.csv", index=False)          # save after every run
    predictions.round(4).to_csv(OUT / "finetune_predictions.csv", index=False)


strata = [f"{s}-{c}" for s, c in zip(subject[train], y[train])]
fit, held_out = train_test_split(np.flatnonzero(train), test_size=0.15, stratify=strata, random_state=0)
few = np.concatenate([train_test_split(np.flatnonzero(train & (subject == s)), train_size=14,
                                       stratify=y[train & (subject == s)], random_state=0)[0] for s in mf.SUBJECTS])
rest = np.random.default_rng(0).choice(np.setdiff1d(np.flatnonzero(train), few), 200, replace=False)

for seed in (0, 1):
    run("all participants, 10 % of the labels", seed, few, rest, **FEW_LABELS)
for seed in (1, 2, 3):
    run("all participants", seed, fit, held_out, **TWO_STAGES)
    run("all participants, one stage", seed, fit, held_out, **ONE_STAGE)
print("Saved to", OUT)
