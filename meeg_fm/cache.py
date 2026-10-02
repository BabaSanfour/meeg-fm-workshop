"""Precomputed REVE embeddings of BCI IV 2a, so the frozen-model notebook runs on any laptop.

One file per participant (left vs right hand, both sessions), shipped inside the
package. Missing files can be rebuilt from the raw data with build_embedding_cache.
"""

import json
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

from .data import CHANNELS, CLASSES, SUBJECTS, data_dir, load_epochs
from .reve import CLIP, MODEL_ID, SFREQ, embed, load_reve

CACHE_DIR = Path(str(files("meeg_fm").joinpath("cache")))   # ships with the package
BAND_HZ = (0.5, 99.5)
WINDOW_S = (0.0, 4.0)


def cache_file(subject, folder=CACHE_DIR):
    return Path(folder) / f"reve_bnci2014_001_s{subject:02d}.npz"


def _find(subject):
    """The cached file of this participant: shipped with the package, or rebuilt in the data folder."""
    for folder in (CACHE_DIR, data_dir() / "reve_cache"):
        if cache_file(subject, folder).exists():
            return cache_file(subject, folder)
    return None


def missing_from_cache(subjects=SUBJECTS):
    """Participants whose embeddings are not on disk yet."""
    return [s for s in subjects if _find(s) is None]


def load_embedding_cache(subjects=SUBJECTS):
    """Cached embeddings of these participants.

    Returns a dict: X (trials x 22 channels x 512), y, subject, session, run, and
    info (how the embeddings were made). Trials are in the same order as
    load_epochs(subjects) returns them.
    """
    parts = []
    for subject in subjects:
        path = _find(subject)
        if path is None:
            raise FileNotFoundError(f"No cached embeddings for participant {subject}: run build_embedding_cache([{subject}]).")
        with np.load(path, allow_pickle=False) as f:
            parts.append({k: f[k] for k in f.files})
    cache = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "subject", "session", "run")}
    cache["X"] = cache["X"].astype(np.float32)
    cache["info"] = json.loads(str(parts[0]["info"]))
    return cache


def build_embedding_cache(subjects=SUBJECTS, folder=None, classes=CLASSES):
    """Compute the embeddings from the raw data and save one file per participant.

    About 15 s per participant on a 4-core laptop (about 30 s on Colab's CPU).
    Default folder: <data folder>/reve_cache, which every notebook also reads.
    """
    import braindecode

    folder = Path(folder) if folder is not None else data_dir() / "reve_cache"
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "model": MODEL_ID, "braindecode": braindecode.__version__, "dataset": "BNCI2014_001",
        "classes": list(classes), "sfreq": SFREQ, "window_s": list(WINDOW_S), "band_hz": list(BAND_HZ),
        "channels": CHANNELS, "pooling": "mean over time patches: one 512-d vector per channel",
        "normalization": f"z-score per channel and trial, clipped at {CLIP}",
    }
    start = time.time()
    model = None
    for subject in subjects:
        X, y, meta = load_epochs([subject], classes=classes, sfreq=SFREQ, fmin=BAND_HZ[0], fmax=BAND_HZ[1],
                                 tmin=WINDOW_S[0], tmax=WINDOW_S[1])
        model = model or load_reve(CHANNELS, X.shape[-1])
        Z = embed(X, CHANNELS, model=model, progress=False)
        np.savez_compressed(
            cache_file(subject, folder),
            X=Z.astype(np.float16),              # float16 halves the size; plenty for a linear probe
            y=y.astype(str),
            subject=meta["subject"].to_numpy().astype(np.int16),
            session=meta["session"].to_numpy().astype(str),
            run=meta["run"].to_numpy().astype(str),
            info=json.dumps(info),
        )
        print(f"participant {subject}: {len(y)} trials embedded ({time.time() - start:.0f} s so far)")
    return folder
