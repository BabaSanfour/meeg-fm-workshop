"""Helpers shared by the M/EEG foundation-model workshop notebooks.

The notebooks show the important code; this package holds what participants
do not need to read (data folders, downloads, the REVE embedding cache).
"""

__version__ = "0.1.0"

from .data import (CHANNELS, CLASSES, SUBJECTS, data_dir, download, head_info, is_downloaded, load_epochs,
                   use_data_dir)
from .reve import MODEL_ID, embed, load_reve
from .cache import CACHE_DIR, build_embedding_cache, load_embedding_cache, missing_from_cache
from .finetune import finetune_reve, load_precomputed, pick_device, predict_proba

__all__ = [
    "CHANNELS", "CLASSES", "SUBJECTS", "data_dir", "download", "head_info", "is_downloaded", "load_epochs", "use_data_dir",
    "MODEL_ID", "embed", "load_reve",
    "CACHE_DIR", "build_embedding_cache", "load_embedding_cache", "missing_from_cache",
    "finetune_reve", "load_precomputed", "pick_device", "predict_proba",
]
