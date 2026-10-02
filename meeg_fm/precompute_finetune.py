"""Fine-tune REVE once and save what notebook 4 shows when no GPU is available.

Run from the repository folder: python -m meeg_fm.precompute_finetune
Writes three small tables to meeg_fm/precomputed/ (about 20 minutes on an Apple GPU):
  finetune_history.csv      one row per run and pass: loss, accuracy on training and on held-out day-1 trials
  finetune_predictions.csv  one row per run and day-2 trial: probability of "right hand"
  finetune_embeddings.csv   every trial in the two main directions of the fine-tuned model's embeddings
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import meeg_fm as mf
from meeg_fm.finetune import finetune_reve, predict_proba

OUT = Path(mf.__file__).parent / "precomputed"
RECIPE = dict(head_epochs=3, epochs=10, head_lr=1e-4, lr=3e-5, batch_size=32)      # chosen on held-out day-1 trials
RIGHT = mf.CLASSES.index("right_hand")

mf.use_data_dir()
X, y, meta = mf.load_epochs(mf.SUBJECTS, sfreq=200, fmin=0.5, fmax=99.5)
subject = meta["subject"].to_numpy()
train = (meta["session"] == "0train").to_numpy()
test = ~train
histories, predictions = [], []


def run(name, fit, held_out, **recipe):
    """Fine-tune on the `fit` trials, follow the `held_out` day-1 trials, predict every day-2 trial."""
    print(f"--- {name}: {len(fit)} training trials")
    model, history = finetune_reve(X[fit], y[fit], X[held_out], y[held_out], **recipe)
    histories.append(pd.DataFrame(history).assign(run=name, training_trials=len(fit)))
    rows = np.flatnonzero(test) if name.startswith("all") else np.flatnonzero(test & (subject == subject[fit[0]]))
    predictions.append(pd.DataFrame({"run": name, "trial": rows, "subject": subject[rows], "label": y[rows],
                                     "p_right": predict_proba(model, X[rows])[:, RIGHT]}))
    return model


# 1. one model for everyone: 85 % of everyone's day 1 to learn from, 15 % held out to follow the training
strata = [f"{s}-{c}" for s, c in zip(subject[train], y[train])]
fit, held_out = train_test_split(np.flatnonzero(train), test_size=0.15, stratify=strata, random_state=0)
model = run("all participants", fit, held_out, **RECIPE)

# the embeddings of that model, reduced to two directions for a figure
import torch
pos = model.get_positions(mf.CHANNELS).to(next(model.parameters()).device)
x = torch.from_numpy(mf.reve.normalize(X))
features = []
model.eval()
with torch.no_grad():
    for i in range(0, len(x), 64):
        xb = x[i:i + 64].to(pos.device)
        tokens = model(xb, pos=pos.expand(len(xb), -1, -1), return_features=True)["features"]
        features.append(tokens.mean(dim=2).flatten(1).cpu().numpy())
plane = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(np.concatenate(features)))
pd.DataFrame({"subject": subject, "label": y, "session": meta["session"], "direction_1": plane[:, 0],
              "direction_2": plane[:, 1]}).round(3).to_csv(OUT / "finetune_embeddings.csv", index=False)
del model

# 2. the same data with a recipe that fails: no head-only stage and a learning rate three times larger
run("all participants, large learning rate", fit, held_out, head_epochs=0, epochs=10, lr=1e-4, batch_size=32)

# 3. one model for everyone with 10 % of the labels: 14 trials per participant
few = np.concatenate([train_test_split(np.flatnonzero(train & (subject == s)), train_size=14,
                                       stratify=y[train & (subject == s)], random_state=0)[0] for s in mf.SUBJECTS])
rest = np.setdiff1d(np.flatnonzero(train), few)
run("all participants, 10 % of the labels", few, np.random.default_rng(0).choice(rest, 200, replace=False), **RECIPE)

# 4. one model per participant, trained on that participant only
for s in mf.SUBJECTS:
    own = np.flatnonzero(train & (subject == s))
    fit_s, held_out_s = train_test_split(own, test_size=0.15, stratify=y[own], random_state=0)
    run(f"participant {s} alone", fit_s, held_out_s, **{**RECIPE, "batch_size": 16})

pd.concat(histories).round(4).to_csv(OUT / "finetune_history.csv", index=False)
pd.concat(predictions).round(4).to_csv(OUT / "finetune_predictions.csv", index=False)
print("Saved to", OUT)
