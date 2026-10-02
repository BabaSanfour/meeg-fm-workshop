"""BCI Competition IV 2a (BNCI 2014-001): where it lives, how to download it, how to epoch it."""

import contextlib
import io
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module=r"(mne|moabb|braindecode)")

SUBJECTS = list(range(1, 10))                   # 9 participants
CLASSES = ("left_hand", "right_hand")            # the tutorial's task; 4 classes are homework
ALL_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
CHANNELS = [                                     # 22 EEG channels, 10-20 names (the 3 EOG channels are dropped)
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
FILE_MB = 43                                     # each .mat file; 2 per participant (sessions T and E)


def in_colab():
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules


def data_dir():
    """Folder where every notebook reads and writes data.

    Order: the EEGFM_DATA environment variable, then Google Drive (Colab with Drive
    mounted), then /content (Colab without Drive), then MNE's usual folder
    (~/mne_data), which MOABB and MNE already use on your computer.
    """
    if os.environ.get("EEGFM_DATA"):
        return Path(os.environ["EEGFM_DATA"])
    if in_colab():
        drive = Path("/content/drive/MyDrive")
        return drive / "meeg-fm" if drive.exists() else Path("/content/meeg-fm")
    import mne
    return Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")


def use_data_dir(path=None):
    """Point MNE, MOABB and this package to one data folder and return it."""
    path = Path(path) if path is not None else data_dir()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["EEGFM_DATA"] = str(path)
    os.environ["MNE_DATA"] = str(path)            # read by MNE and MOABB; your saved MNE settings stay untouched
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(path)
    return path


def _dataset():
    from moabb.datasets import BNCI2014_001
    return BNCI2014_001()


def _files(subject):
    """Paths of the .mat files of one participant already on disk (sessions T and E)."""
    found = []
    for folder in ("NEMAR", "MNE-bnci-data"):   # MOABB's mirror first, then the original host
        root = data_dir() / folder
        if root.exists():
            found += [f for s in ("T", "E") for f in root.rglob(f"A{subject:02d}{s}.mat")]
    return found[:2]


def is_downloaded(subject):
    """True if both sessions of this participant are on disk."""
    files = _files(subject)
    return len(files) == 2 and all(f.stat().st_size > 1e6 for f in files)


def download(subjects=SUBJECTS, retries=2):
    """Download the raw files of these participants (skips what is already there).

    Returns a table with one row per participant: size on disk and seconds taken.
    """
    import pandas as pd
    import pooch
    pooch.get_logger().setLevel("WARNING")      # hide the checksum message printed for every file
    dataset = _dataset()
    rows = []
    for subject in subjects:
        start = time.time()
        for attempt in range(retries + 1):
            if is_downloaded(subject):
                break
            try:
                with contextlib.redirect_stdout(io.StringIO()):   # keep the progress bars, drop the chatter
                    dataset.download(subject_list=[subject], verbose=False)
                break
            except Exception:                    # flaky Wi-Fi: try again before giving up
                if attempt == retries:
                    raise
                time.sleep(2 * (attempt + 1))
        size = sum(f.stat().st_size for f in _files(subject)) / 1e6
        rows.append({"subject": subject, "MB": round(size), "seconds": round(time.time() - start, 1)})
    return pd.DataFrame(rows).set_index("subject")


def load_epochs(subjects=SUBJECTS, classes=CLASSES, sfreq=None, fmin=None, fmax=None, tmin=0.0, tmax=4.0,
                return_epochs=False):
    """Cut every trial of these participants into an epoch, the same way in every notebook.

    Each trial is the 4 s of motor imagery after the cue (tmin=0, tmax=4).
    Returns X (trials x 22 channels x samples, in microvolts), y (class names) and a
    metadata table (subject, session, run). Session "0train" is day 1 and
    "1test" is day 2: the tutorial trains on day 1 and tests on day 2.

    With return_epochs=True, returns the same trials as one MNE Epochs object instead
    (in volts, with electrode positions; `epochs["left_hand"]` selects a class and
    `epochs.metadata` holds subject, session, run and label).
    """
    from moabb.paradigms import MotorImagery
    paradigm = MotorImagery(events=list(classes), n_classes=len(classes), fmin=fmin, fmax=fmax,
                            tmin=tmin, tmax=tmax, resample=sfreq, channels=CHANNELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, meta = paradigm.get_data(_dataset(), subjects=list(subjects))
    n_samples = int(round((tmax - tmin) * (sfreq or 250)))
    X = X[:, :, :n_samples].astype(np.float32)   # MOABB keeps the last sample; drop it so 4 s = 4 x sfreq
    y, meta = np.asarray(y), meta.reset_index(drop=True)
    if return_epochs:
        return _to_mne(X, y, meta, classes, sfreq or 250, tmin)
    return X, y, meta


def _to_mne(X, y, meta, classes, sfreq, tmin):
    """Wrap the trials in an MNE Epochs object (volts, standard 10-20 electrode positions)."""
    import mne
    info = mne.create_info(CHANNELS, sfreq, "eeg").set_montage("standard_1020")
    event_id = {name: number for number, name in enumerate(classes, start=1)}
    events = np.column_stack([np.arange(len(y)) * X.shape[-1], np.zeros(len(y), int), [event_id[c] for c in y]])
    return mne.EpochsArray(X * 1e-6, info, events=events, event_id=event_id, tmin=tmin,
                           metadata=meta.assign(label=y), verbose=False)
