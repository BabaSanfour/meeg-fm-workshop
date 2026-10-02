"""Fine-tune REVE on labelled trials: first the classification head alone, then every weight."""

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


def predict_proba(model, X, device=None, batch_size=64):
    """Probability of each class (trials x classes) for epochs sampled at 200 Hz, in microvolts."""
    import torch
    device = device or next(model.parameters()).device
    pos = model.get_positions(list(CHANNELS)).to(device)
    x = torch.from_numpy(normalize(X))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            out.append(torch.softmax(model(xb, pos=pos.expand(len(xb), -1, -1)), dim=1).cpu().numpy())
    return np.concatenate(out)


def finetune_reve(X_train, y_train, X_eval=None, y_eval=None, head_epochs=5, epochs=15, head_lr=1e-3, lr=3e-5,
                  batch_size=16, weight_decay=0.01, seed=0, device=None, verbose=True):
    """Fine-tune the pretrained REVE on these trials (200 Hz, microvolts) and return (model, history).

    Two stages: `head_epochs` passes where only the classification
    head learns (the encoder stays frozen), then `epochs` passes where every weight
    learns, with a small learning rate. history has one row per pass: stage, loss and
    accuracy on the training trials, and accuracy on the evaluation trials if given.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()
    model = load_reve(CHANNELS, X_train.shape[-1]).to(device)
    pos = model.get_positions(list(CHANNELS)).to(device)
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
                logits = model(xb, pos=pos.expand(len(xb), -1, -1))
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
                extra = f", evaluation {100 * row['eval_accuracy']:.0f} %" if X_eval is not None else ""
                print(f"pass {row['epoch']:2d} ({stage}): loss {row['loss']:.3f}, "
                      f"training {100 * row['train_accuracy']:.0f} %{extra}")
    return model, history


def load_precomputed(name):
    """A table saved by meeg_fm/precompute_finetune.py: "history", "predictions" or "embeddings"."""
    import pandas as pd
    return pd.read_csv(Path(__file__).parent / "precomputed" / f"finetune_{name}.csv")
