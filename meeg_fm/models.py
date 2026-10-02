"""Other pretrained EEG models through Braindecode, each with the preparation it needs on our 22 channels.

REVE itself is loaded in the open in notebook 3. This module holds the recipes of the
other models, so that a notebook can swap one for another in a line:

    Z = mf.models.embed("cbramod", X200)      # trials x tokens x numbers

Every recipe takes the trials prepared as for REVE (200 Hz, microvolts, 4 s).
"""

import numpy as np

from .data import CHANNELS
from .reve import normalize

CHS_INFO = [{"ch_name": name} for name in CHANNELS]


def _load_cbramod(n_times):
    from braindecode.models import CBraMod
    return CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True, n_chans=len(CHANNELS),
                                   n_times=n_times, sfreq=200, chs_info=CHS_INFO)


def _load_labram(n_times):
    """LaBraM was saved for 15 s windows: build it for our length and keep the first time embeddings."""
    import torch
    from braindecode.models import Labram
    saved = Labram.from_pretrained("braindecode/labram-pretrained").state_dict()
    model = Labram(n_times=n_times, n_chans=len(CHANNELS), n_outputs=2, chs_info=CHS_INFO, sfreq=200)
    own = model.state_dict()
    saved["temporal_embedding"] = saved["temporal_embedding"][:, :own["temporal_embedding"].shape[1]]
    model.load_state_dict({k: v for k, v in saved.items() if k in own and v.shape == own[k].shape}, strict=False)
    model.final_layer = torch.nn.Identity()          # we want the tokens, not a class
    return model


def _load_luna(n_times):
    """LUNA expects 250 Hz, so the trials are resampled; its head is removed to get the tokens."""
    import torch
    from braindecode.models import LUNA
    model = LUNA.from_pretrained("PulpBio/LUNA", filename="LUNA_base.safetensors", n_outputs=2, n_chans=len(CHANNELS),
                                 n_times=int(n_times * 250 / 200), embed_dim=64, num_queries=4, depth=8, chs_info=CHS_INFO)
    model.final_layer = torch.nn.Identity()
    return model


RECIPES = {
    "cbramod": dict(load=_load_cbramod, scale="z-score", call={}, pool_patches=True,
                    adapted="nothing: it loads for any number of channels"),
    "labram": dict(load=_load_labram, scale="z-score", call={"ch_names": CHANNELS, "return_patch_tokens": True},
                   pool_patches=False, adapted="saved for 15 s windows: rebuilt for 4 s, time embeddings cut"),
    "luna": dict(load=_load_luna, scale="z-score", call={}, pool_patches=False, resample=250,
                 adapted="trials resampled to 250 Hz, classification head removed"),
}


def load(name, n_times=800):
    """The pretrained model `name` ("cbramod", "labram" or "luna"), ready to give tokens for our 22 channels."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # the libraries print notices that do not concern us here
        return RECIPES[name]["load"](n_times).eval()


def embed(name, X, batch_size=32):
    """Frozen embeddings of trials sampled at 200 Hz (trials x channels x samples, microvolts).

    Returns trials x tokens x numbers. For CBraMod the tokens are the 22 electrodes
    (patches averaged over time); LaBraM and LUNA return their own tokens.
    """
    import torch
    recipe = RECIPES[name]
    model = load(name, X.shape[-1])
    x = normalize(X)
    if recipe.get("resample"):
        from scipy.signal import resample
        x = resample(x, int(X.shape[-1] * recipe["resample"] / 200), axis=-1).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            tokens = model(torch.from_numpy(x[i:i + batch_size]), **recipe["call"])
            out.append((tokens.mean(dim=2) if recipe["pool_patches"] else tokens).numpy())
    return np.concatenate(out)


def n_weights(model):
    """Number of weights of a model, in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
