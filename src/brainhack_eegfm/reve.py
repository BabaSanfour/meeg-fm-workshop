"""REVE (El Ouahidi et al., 2025) through Braindecode: load the pretrained model, turn epochs into embeddings."""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module=r"(mne|moabb|braindecode)")

MODEL_ID = "brain-bzh/reve-base"   # 69 M parameters, 512-dimensional tokens
SFREQ = 200                        # REVE was pretrained on EEG sampled at 200 Hz
CLIP = 15                          # after z-scoring, values beyond 15 standard deviations are clipped


def load_reve(ch_names, n_times, model_id=MODEL_ID):
    """Download (once) and load the pretrained REVE encoder for these channels and epoch length."""
    from braindecode.models import REVE
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()              # no login needed: hide the "unauthenticated requests" notice
    model = REVE.from_pretrained(
        model_id,
        n_outputs=2,               # a small classification head we never use here
        n_chans=len(ch_names),
        n_times=n_times,
        sfreq=SFREQ,
        chs_info=[{"ch_name": c} for c in ch_names],
    )
    return model.eval()


def normalize(X):
    """Z-score each channel of each trial, then clip, as in REVE's pretraining."""
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
    return np.clip(X, -CLIP, CLIP).astype(np.float32)


def embed(X, ch_names, model=None, batch_size=32, progress=True):
    """Frozen REVE embeddings of epochs sampled at 200 Hz.

    X: trials x channels x samples. Returns trials x channels x 512: each channel's
    tokens averaged over time, so every electrode keeps its own vector.
    """
    import torch
    from tqdm.auto import tqdm

    model = model or load_reve(ch_names, X.shape[-1])
    pos = model.get_positions(list(ch_names))
    X = normalize(X)
    out = []
    batches = range(0, len(X), batch_size)
    with torch.no_grad():
        for i in tqdm(batches, disable=not progress, desc="REVE"):
            xb = torch.from_numpy(X[i:i + batch_size])
            feats = model(xb, pos=pos.expand(len(xb), -1, -1), return_features=True)["features"]
            out.append(feats.mean(dim=2).numpy())   # batch x channels x time patches x 512 -> mean over time
    return np.concatenate(out)
