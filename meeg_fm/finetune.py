"""Fine-tune a pretrained model on labelled trials: first the classification head alone, then every weight."""

import time
from pathlib import Path

import numpy as np

from .data import CHANNELS, CLASSES
from .reve import load_reve, normalize


def pick_device():
    """The fastest device available: an NVIDIA GPU (cuda), an Apple GPU (mps), or the processor (cpu)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _forward(model, xb):
    """The model's output for one batch; REVE also takes the electrode positions."""
    if hasattr(model, "get_positions"):
        pos = model.get_positions(list(CHANNELS)).to(xb.device)
        return model(xb, pos=pos.expand(len(xb), -1, -1))
    return model(xb)


def predict_proba(model, X, device=None, batch_size=64):
    """Probability of each class (trials x classes) for epochs sampled at 200 Hz, in microvolts."""
    import torch
    device = device or next(model.parameters()).device
    x = torch.from_numpy(normalize(X))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(torch.softmax(_forward(model, x[i:i + batch_size].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def finetune(model, X_train, y_train, X_eval=None, y_eval=None, head_epochs=3, epochs=10, head_lr=1e-4, lr=3e-5,
             batch_size=32, weight_decay=0.01, seed=0, device=None, verbose=True):
    """Fine-tune a pretrained model with a classification head (`final_layer`) and return (model, history).

    Two stages: `head_epochs` passes where only the head learns (the encoder stays
    frozen), then `epochs` passes where every weight learns, with a small learning
    rate. X_train: trials sampled at 200 Hz, in microvolts. history has one row per
    pass: stage, loss and accuracy on the training trials, and accuracy on the
    evaluation trials if given.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()
    model = model.to(device)
    x = torch.from_numpy(normalize(X_train))
    target = torch.tensor([CLASSES.index(c) for c in y_train])
    encoder = [p for name, p in model.named_parameters() if not name.startswith("final_layer")]

    history, start = [], time.time()
    stages = [("head only", head_epochs, head_lr, False), ("all weights", epochs, lr, True)]
    for stage, n_epochs, rate, train_encoder in stages:
        for p in encoder:
            p.requires_grad = train_encoder
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate,
                                      weight_decay=weight_decay)
        for _ in range(n_epochs):
            model.train()
            order = rng.permutation(len(x))
            losses, correct = [], 0
            for i in range(0, len(order), batch_size):
                rows = order[i:i + batch_size]
                xb, tb = x[rows].to(device), target[rows].to(device)
                logits = _forward(model, xb)
                loss = torch.nn.functional.cross_entropy(logits, tb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                correct += (logits.argmax(dim=1) == tb).sum().item()
            row = {"epoch": len(history) + 1, "stage": stage, "loss": float(np.mean(losses)),
                   "train_accuracy": correct / len(x), "seconds": time.time() - start}
            if X_eval is not None:
                predicted = predict_proba(model, X_eval, device).argmax(axis=1)
                row["eval_accuracy"] = float(np.mean(np.array(CLASSES)[predicted] == np.asarray(y_eval)))
            history.append(row)
            if verbose:
                extra = f", held-out {100 * row['eval_accuracy']:.0f} %" if X_eval is not None else ""
                print(f"pass {row['epoch']:2d} ({stage}): loss {row['loss']:.3f}, "
                      f"training {100 * row['train_accuracy']:.0f} %{extra}")
    return model, history


def finetune_reve(X_train, y_train, X_eval=None, y_eval=None, **recipe):
    """Fine-tune the pretrained REVE (see `finetune`)."""
    return finetune(load_reve(CHANNELS, X_train.shape[-1]), X_train, y_train, X_eval, y_eval, **recipe)


def load_precomputed(name):
    """A table saved by meeg_fm/precompute_finetune.py: "history", "predictions" or "embeddings"."""
    import pandas as pd
    return pd.read_csv(Path(__file__).parent / "precomputed" / f"finetune_{name}.csv")
