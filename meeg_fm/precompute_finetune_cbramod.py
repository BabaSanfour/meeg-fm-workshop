"""Fine-tune CBraMod with the recipe used for REVE, to compare the two models once their weights can change.

Run after precompute_finetune and precompute_finetune_seeds, from the repository folder:
python -m meeg_fm.precompute_finetune_cbramod (about 20 minutes on a laptop processor).
Adds rows to finetune_history.csv and finetune_predictions.csv, under the run "cbramod, all participants".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from braindecode.models import CBraMod
from sklearn.model_selection import train_test_split

import meeg_fm as mf
from meeg_fm.finetune import finetune, predict_proba

OUT = Path(mf.__file__).parent / "precomputed"
RECIPE = dict(head_epochs=3, epochs=10, head_lr=1e-4, lr=3e-5, batch_size=32)
RUN = "cbramod, all participants"

mf.use_data_dir()
X, y, meta = mf.load_epochs(mf.SUBJECTS, sfreq=200, fmin=0.5, fmax=99.5)
subject = meta["subject"].to_numpy()
train = (meta["session"] == "0train").to_numpy()
test = np.flatnonzero(~train)
strata = [f"{s}-{c}" for s, c in zip(subject[train], y[train])]
fit, held_out = train_test_split(np.flatnonzero(train), test_size=0.15, stratify=strata, random_state=0)

history = pd.read_csv(OUT / "finetune_history.csv")
predictions = pd.read_csv(OUT / "finetune_predictions.csv")
history, predictions = history[history["run"] != RUN], predictions[predictions["run"] != RUN]
for seed in (0, 1):
    print(f"--- {RUN}, seed {seed}", flush=True)
    model = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_outputs=2, n_chans=len(mf.CHANNELS),
                                    n_times=X.shape[-1], sfreq=200, chs_info=[{"ch_name": c} for c in mf.CHANNELS])
    model, rows = finetune(model, X[fit], y[fit], X[held_out], y[held_out], seed=seed, verbose=False,
                           device="cpu", **RECIPE)      # CBraMod's attention does not run on Apple GPUs
    history = pd.concat([history, pd.DataFrame(rows).assign(run=RUN, training_trials=len(fit), seed=seed)])
    predictions = pd.concat([predictions, pd.DataFrame({
        "run": RUN, "trial": test, "subject": subject[test], "label": y[test],
        "p_right": predict_proba(model, X[test])[:, mf.CLASSES.index("right_hand")], "seed": seed})])
history.round(4).to_csv(OUT / "finetune_history.csv", index=False)
predictions.round(4).to_csv(OUT / "finetune_predictions.csv", index=False)
print("Saved to", OUT)
