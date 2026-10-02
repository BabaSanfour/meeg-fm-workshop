"""Concept figures embedded in the notebooks' markdown (run once: python figures/make_figures.py).

Needs the tutorial package (`pip install -e .`); downloads participant 1 (87 MB) if it is not on disk yet.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import brainhack_eegfm as be

NAVY, BLUE, ORANGE, GREY, LIGHT, TRACE = "#0B2545", "#0B5CAD", "#E8710A", "#5B6472", "#EEF3F9", "#9AA3AF"
OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})


def card(fig, x, w, number, title, purpose, package):
    """One numbered step: a light card with title, one-line purpose and the package used."""
    ax = fig.add_axes([x, 0.08, w, 0.80])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=LIGHT, edgecolor="none"))
    ax.text(0.08, 0.88, f"{number}  {title}", color=NAVY, fontsize=13, fontweight="bold", va="center")
    ax.text(0.08, 0.76, purpose, color=GREY, fontsize=9.5, va="center")
    ax.text(0.08, 0.08, package, color=BLUE, fontsize=10, family="monospace", va="center")
    return ax


def inset(fig, x, w):
    """Axes for the small drawing in the middle of a card."""
    ax = fig.add_axes([x + 0.03 * w / 0.22, 0.25, w * 0.86, 0.36])
    ax.set_axis_off()
    return ax


def electrode_xy(names):
    """Top view of the standard 10-20 positions of these electrodes (the head circle has radius 1)."""
    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    xy = {}
    for name in names:
        x, y, z = positions[name] / np.linalg.norm(positions[name])
        radius = np.arccos(z) / (np.pi / 2)        # angle from the top of the head; 1 at the Fpz-T7-Oz ring
        xy[name] = radius * np.array([x, y]) / np.hypot(x, y) if radius > 1e-6 else np.zeros(2)
    return xy


def setup_pipeline():
    fig = plt.figure(figsize=(11, 3.5), dpi=150, facecolor="white")
    w, gap, x0 = 0.22, 0.03, 0.02
    xs = [x0 + i * (w + gap) for i in range(4)]
    rng = np.random.default_rng(3)

    # 1. environment
    card(fig, xs[0], w, "1", "Check", "Python, CPU or GPU, memory", "sys · torch")
    ax = inset(fig, xs[0], w)
    for i, line in enumerate(["Python 3.11", "4 CPU cores", "GPU: none (fine)"]):
        ax.text(0.02, 0.85 - i * 0.33, "✓", color=BLUE, fontsize=12, fontweight="bold", va="center")
        ax.text(0.16, 0.85 - i * 0.33, line, color=NAVY, fontsize=10.5, va="center", family="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2. install
    card(fig, xs[1], w, "2", "Install", "pip install -e .", "pip or uv")
    ax = inset(fig, xs[1], w)
    for i, name in enumerate(["mne", "moabb", "pyriemann", "braindecode"]):
        ax.text(0.02, 0.9 - i * 0.27, name, color=NAVY, fontsize=10.5, va="center", family="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 3. data: 2 s of a real trial (participant 1, first left-hand trial, 1-40 Hz) on C3, Cz, C4
    card(fig, xs[2], w, "3", "Download", "BCI IV 2a, 780 MB", "moabb")
    be.use_data_dir()
    X, y, _ = be.load_epochs([1], fmin=1, fmax=40)
    trial = X[np.flatnonzero(y == "left_hand")[0], :, 250:750]   # 1 to 3 s after the cue, 250 Hz
    shown = ["C3", "Cz", "C4"]

    head = fig.add_axes([xs[2] + 0.010, 0.22, 0.062, 0.42])
    head.set_axis_off()
    head.set_aspect("equal")
    head.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor=GREY, lw=1))
    head.plot([-0.12, 0, 0.12], [0.99, 1.14, 0.99], color=GREY, lw=1)          # nose
    for name, (ex, ey) in electrode_xy(be.CHANNELS).items():
        on = name in shown
        head.plot(ex, ey, "o", ms=3.6 if on else 2.2, color=ORANGE if on else TRACE)
    head.set_xlim(-1.1, 1.1)
    head.set_ylim(-1.1, 1.2)
    head.text(0, -1.45, "22 electrodes", color=GREY, fontsize=8.5, ha="center", va="center")

    ax = fig.add_axes([xs[2] + 0.098, 0.24, w - 0.112, 0.38])
    ax.set_axis_off()
    t = np.arange(trial.shape[1]) / 250
    for i, name in enumerate(shown):
        ax.plot(t, trial[be.CHANNELS.index(name)] - 50 * i, color=NAVY, lw=0.55)
        ax.text(-0.06, -50 * i, name, color=GREY, fontsize=9, ha="right", va="center")
    ax.plot([0, 1], [-135, -135], color=GREY, lw=1.2)                              # time scale
    ax.text(0.5, -148, "1 s", color=GREY, fontsize=8.5, ha="center", va="center")
    ax.set_xlim(-0.42, 2.0)
    ax.set_ylim(-158, 30)

    # 4. embeddings: channel x time tokens averaged into one vector per channel
    card(fig, xs[3], w, "4", "Embeddings", "REVE, one vector per channel", "braindecode")
    ax = inset(fig, xs[3], w)
    tokens = rng.random((6, 4))
    ax.imshow(tokens, cmap="Blues", vmin=-0.3, vmax=1.2, extent=[0, 4, 0, 6], aspect="auto")
    ax.imshow(tokens.mean(1, keepdims=True), cmap="Oranges", vmin=-0.3, vmax=1.2, extent=[6, 7, 0, 6], aspect="auto")
    ax.annotate("", xy=(5.8, 3), xytext=(4.3, 3), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(2, -0.9, "channel × time", color=GREY, fontsize=9, ha="center")
    ax.text(6.5, -0.9, "mean", color=GREY, fontsize=9, ha="center")
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(-1.6, 6.2)

    for x in xs[:-1]:
        fig.add_artist(FancyArrowPatch((x + w + 0.004, 0.48), (x + w + gap - 0.004, 0.48),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12, color=GREY))
    fig.text(0.02, 0.95, "The four steps of this notebook",
             color=GREY, fontsize=11, va="center")
    fig.savefig(OUT / "setup_pipeline.png", facecolor="white")
    plt.close(fig)


def screen(ax, x, y, w, h, content, color=NAVY):
    """A small monitor showing a cross, an arrow, or nothing."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", facecolor="#111827",
                                edgecolor=GREY, lw=1.2))
    ax.plot([x + w / 2, x + w / 2], [y - 0.12, y], color=GREY, lw=2)
    ax.plot([x + w / 2 - 0.18, x + w / 2 + 0.18], [y - 0.12, y - 0.12], color=GREY, lw=2)
    cx, cy = x + w / 2, y + h / 2
    if content == "cross":
        ax.plot([cx - 0.14, cx + 0.14], [cy, cy], color="white", lw=1.4)
        ax.plot([cx, cx], [cy - 0.17, cy + 0.17], color="white", lw=1.4)
    if content in ARROWS:
        dx, dy = ARROWS[content]
        ax.annotate("", xy=(cx + 0.3 * w * dx, cy + 0.3 * h * dy), xytext=(cx - 0.3 * w * dx, cy - 0.3 * h * dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, mutation_scale=13))


ARROWS = {"left": (-1, 0), "right": (1, 0), "down": (0, -1), "up": (0, 1)}
HANDS = {"left": "#1baf7a", "right": "#4a3aa7"}              # the class colours of the notebooks


def trial_timeline():
    """The experiment of BCI IV 2a: what the screen shows during one trial, and the four cues."""
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=150, facecolor="white")
    ax.set_axis_off()

    # what the person sees, above the period it belongs to
    for centre, content in [(-1, "cross"), (0.625, "left"), (2.625, "cross"), (4.75, None)]:
        screen(ax, centre - 0.6, 1.6, 1.2, 0.95, content, color=HANDS["left"])

    blocks = [(-2, 0, "#F3F4F6", "fixation cross"), (0, 1.25, "#FDF1E7", "cue"),
              (1.25, 4, LIGHT, "imagine the movement"), (4, 5.5, "white", "break")]
    for start, stop, color, label in blocks:
        ax.add_patch(FancyBboxPatch((start, 0.55), stop - start, 0.65, boxstyle="square,pad=0", facecolor=color,
                                    edgecolor=GREY, lw=0.8, linestyle="--" if label == "break" else "-"))
        ax.text((start + stop) / 2, 0.875, label, ha="center", va="center", color=NAVY, fontsize=10)
    ax.text(-2, 1.3, "beep", ha="center", color=GREY, fontsize=9.5)
    for x in range(-2, 5):
        ax.plot([x, x], [0.45, 0.55], color=GREY, lw=0.8)
        ax.text(x, 0.28, f"{x} s", ha="center", va="center", color=GREY, fontsize=9.5)
    ax.annotate("", xy=(0, -0.05), xytext=(4, -0.05), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
    ax.text(2, -0.3, "one trial: 0 to 4 s after the cue", ha="center", va="center",
            color=BLUE, fontsize=10)
    ax.annotate("", xy=(-1.5, -0.05), xytext=(-0.5, -0.05), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
    ax.text(-1.0, -0.3, "baseline", ha="center", va="center", color=ORANGE, fontsize=10)

    # the four cues
    ax.text(6.3, 1.3, "The four cues", color=NAVY, fontsize=10.5, fontweight="bold")
    cues = [("left", "left hand", HANDS["left"]), ("right", "right hand", HANDS["right"]),
            ("down", "both feet", GREY), ("up", "tongue", GREY)]
    for k, (arrow, label, color) in enumerate(cues):
        x = 6.3 + 1.3 * k
        screen(ax, x, 0.45, 1.0, 0.62, arrow, color=color)
        ax.text(x + 0.5, 0.1, label, ha="center", va="center", fontsize=9.5,
                color=color if color != GREY else "#5B6472")
    ax.text(6.3, -0.3, "this tutorial uses left hand and right hand", color=GREY, fontsize=9.5, va="center")
    ax.set_xlim(-2.3, 11.6)
    ax.set_ylim(-0.55, 2.75)
    fig.savefig(OUT / "trial_timeline.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, color=LIGHT, edge="none", fontsize=10, text_color=NAVY, weight="normal"):
    """A rounded box with centred text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", facecolor=color,
                                edgecolor=edge, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color,
            fontweight=weight, linespacing=1.4)


def arrow(ax, start, stop, color=GREY):
    ax.annotate("", xy=stop, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=12))


def decoding_pipelines():
    """From one trial to an answer: the two classical pipelines of notebook 2, step by step."""
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150, facecolor="white")
    ax.set_axis_off()
    for x, title in [(0.1, "1  One trial"), (3.2, "2  Summarise the trial"), (6.6, "3  Features"),
                     (9.0, "4  Classifier"), (11.4, "5  Answer")]:
        ax.text(x, 3.75, title, color=NAVY, fontsize=11, fontweight="bold", va="center")

    # a real trial (participant 1, first left-hand trial, 8-30 Hz), six of its 22 channels
    be.use_data_dir()
    X, y, _ = be.load_epochs([1], fmin=8, fmax=30)
    trial = X[np.flatnonzero(y == "left_hand")[0]]
    box(ax, 0.1, 0.55, 2.5, 2.8, "", color="white", edge=TRACE)
    t = np.linspace(0.25, 2.45, 500)
    for k, name in enumerate(["FC3", "C3", "Cz", "C4", "CP4", "Pz"]):
        ax.plot(t, 3.05 - 0.42 * k + trial[be.CHANNELS.index(name), 250:750] / 55, color=NAVY, lw=0.5)
    ax.text(1.35, 0.3, "22 channels × 1000 samples", ha="center", color=GREY, fontsize=9.5)

    rows = [(2.25, BLUE, "CSP\n4 spatial filters", "4 numbers\n(log power)", "LDA"),
            (0.75, ORANGE, "Covariance matrix\n22 × 22", "253 numbers\ntangent space", "Logistic\nregression")]
    for y0, color, step, features, classifier in rows:
        arrow(ax, (2.65, 1.95), (3.15, y0 + 0.5), color)
        box(ax, 3.2, y0, 2.7, 1.0, step, edge=color)
        arrow(ax, (5.95, y0 + 0.5), (6.55, y0 + 0.5), color)
        box(ax, 6.6, y0, 1.9, 1.0, features, edge=color, fontsize=9.5)
        arrow(ax, (8.55, y0 + 0.5), (8.95, y0 + 0.5), color)
        box(ax, 9.0, y0, 1.8, 1.0, classifier, edge=color)
        arrow(ax, (10.85, y0 + 0.5), (11.35, 1.95), color)
    box(ax, 11.4, 1.45, 1.7, 1.0, "left hand\nor right hand", color="white", edge=NAVY, weight="bold")
    ax.text(3.2, 0.2, "Every step in a coloured frame is learned from the trials of day 1 (fit), then applied unchanged "
            "to day 2 (predict).", color=GREY, fontsize=9.5, va="center")
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4)
    fig.savefig(OUT / "decoding_pipelines.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def tangent_space():
    """A curved space of covariance matrices, flattened around the mean matrix."""
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor="white")
    ax.set_axis_off()
    curve = lambda x: 1.0 - 0.28 * x ** 2                                    # the curved space, seen from the side
    x = np.linspace(-2.3, 2.3, 200)
    ax.plot(x, curve(x), color=GREY, lw=2)
    ax.text(0, -0.85, "space of covariance matrices (curved)", color=GREY, fontsize=9.5, ha="center")
    ax.plot([-2.6, 2.6], [1.0, 1.0], color=NAVY, lw=1.5)
    ax.text(2.65, 1.0, "tangent space\n(flat)", color=NAVY, fontsize=9.5, va="center")
    ax.plot(0, 1.0, "o", color=NAVY, ms=7)
    ax.text(0, 1.2, "mean matrix", color=NAVY, fontsize=9.5, ha="center")
    left, right = np.array([-1.9, -1.5, -1.2, -0.8]), np.array([0.7, 1.1, 1.5, 1.9])
    for points, color in [(left, HANDS["left"]), (right, HANDS["right"])]:
        ax.plot(points, curve(points), "o", color=color, ms=7)
        for px in points:                                                    # each trial moves onto the flat space
            arc = 1.12 * px                                                  # roughly its distance along the curve
            ax.annotate("", xy=(arc, 1.0), xytext=(px, curve(px)),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls="--", mutation_scale=8))
            ax.plot(arc, 1.0, "o", color=color, ms=5, mfc="white")
    ax.text(-1.6, 1.2, "left-hand trials", color=HANDS["left"], fontsize=9.5, ha="center")
    ax.text(1.6, 1.2, "right-hand trials", color=HANDS["right"], fontsize=9.5, ha="center")
    ax.text(0, 1.75, "one covariance matrix per trial (filled) becomes one ordinary vector (open)", color=GREY,
            fontsize=9.5, ha="center")
    ax.set_xlim(-3.0, 3.6)
    ax.set_ylim(-1.0, 1.95)
    fig.savefig(OUT / "tangent_space.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def foundation_model():
    """Pretraining (done once, without labels) and our use of the frozen model (a small classifier on top)."""
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150, facecolor="white")
    ax.set_axis_off()
    ax.text(0.1, 4.55, "Pretraining: done once by the authors, without labels", color=NAVY, fontsize=11, fontweight="bold")
    box(ax, 0.1, 3.1, 3.3, 1.15, "60,000 hours of EEG\n92 datasets, 25,000 people", edge=GREY)
    arrow(ax, (3.45, 3.67), (4.05, 3.67))
    box(ax, 4.1, 3.1, 3.6, 1.15, "hide parts of each recording,\nlearn to fill them in", edge=GREY)
    arrow(ax, (7.75, 3.67), (8.35, 3.67))
    box(ax, 8.4, 3.1, 2.6, 1.15, "REVE\n69 million weights", color="#FDF1E7", edge=ORANGE, weight="bold")
    arrow(ax, (9.7, 3.05), (5.0, 1.95), ORANGE)
    ax.text(8.2, 2.42, "the weights are downloaded, then left unchanged: frozen", color=ORANGE, fontsize=9.5)

    ax.text(0.1, 2.15, "This notebook: only the last box learns from our labels", color=NAVY, fontsize=11,
            fontweight="bold")
    box(ax, 0.1, 0.55, 2.3, 1.15, "one trial\n22 channels × 4 s", edge=GREY)
    arrow(ax, (2.45, 1.12), (3.05, 1.12))
    box(ax, 3.1, 0.55, 2.5, 1.15, "REVE, frozen", color="#FDF1E7", edge=ORANGE, weight="bold")
    arrow(ax, (5.65, 1.12), (6.25, 1.12))
    box(ax, 6.3, 0.55, 2.4, 1.15, "embedding\n22 × 512 numbers", edge=GREY)
    arrow(ax, (8.75, 1.12), (9.35, 1.12))
    box(ax, 9.4, 0.55, 2.6, 1.15, "logistic regression\n(the linear probe)", edge=BLUE)
    arrow(ax, (12.05, 1.12), (12.55, 1.12))
    box(ax, 12.6, 0.55, 1.6, 1.15, "left or\nright hand", color="white", edge=NAVY, weight="bold")
    ax.text(9.4, 0.25, "trained on day 1, tested on day 2", color=BLUE, fontsize=9.5)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 4.8)
    fig.savefig(OUT / "foundation_model.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def reve_tokens():
    """How REVE reads a trial: patches of about 1 s per channel, each with its electrode position."""
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150, facecolor="white")
    ax.set_axis_off()
    for x, title in [(0.1, "1  Cut each channel into patches"), (5.0, "2  One token per patch"),
                     (8.5, "3  Transformer"), (10.9, "4  Average over time")]:
        ax.text(x, 3.8, title, color=NAVY, fontsize=11, fontweight="bold", va="center")

    # a real trial (participant 1, first left-hand trial), five of its 22 channels, with the patch borders
    be.use_data_dir()
    X, y, _ = be.load_epochs([1], fmin=1, fmax=40)
    trial = X[np.flatnonzero(y == "left_hand")[0]]
    names = ["FC3", "C3", "Cz", "C4", "CP4"]
    t = np.linspace(0.6, 4.2, trial.shape[1])
    for k, name in enumerate(names):
        ax.plot(t, 3.0 - 0.55 * k + trial[be.CHANNELS.index(name)] / 75, color=NAVY, lw=0.4)
        ax.text(0.5, 3.0 - 0.55 * k, name, ha="right", va="center", color=GREY, fontsize=9)
    for k in range(5):
        ax.plot([0.6 + 0.9 * k] * 2, [0.45, 3.35], color=ORANGE, lw=1, ls="--")
    for k in range(4):
        ax.text(1.05 + 0.9 * k, 0.25, f"{k}-{k + 1} s", ha="center", color=ORANGE, fontsize=9)

    arrow(ax, (4.35, 1.9), (4.95, 1.9))
    for row in range(5):                                                   # the grid of tokens: channel x time
        for col in range(4):
            ax.add_patch(FancyBboxPatch((5.1 + 0.62 * col, 2.8 - 0.55 * row), 0.5, 0.4, boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=LIGHT, edgecolor=BLUE, lw=0.8))
    ax.text(6.3, -0.05, "22 electrodes × 4 patches = 88 tokens\neach one knows its electrode\nposition (x, y, z) and its time",
            ha="center", va="center", color=GREY, fontsize=9, linespacing=1.4)
    arrow(ax, (7.75, 1.9), (8.45, 1.9))
    box(ax, 8.5, 1.2, 1.8, 1.4, "every token\nlooks at all\nthe others", color="#FDF1E7", edge=ORANGE, fontsize=9.5)
    arrow(ax, (10.35, 1.9), (10.95, 1.9))
    for row in range(5):                                                   # one vector per channel
        ax.add_patch(FancyBboxPatch((11.1, 2.8 - 0.55 * row), 1.9, 0.4, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor="#FDF1E7", edgecolor=ORANGE, lw=0.8))
        ax.text(12.05, 3.0 - 0.55 * row, "512 numbers", ha="center", va="center", color=NAVY, fontsize=8.5)
    ax.text(12.05, 0.05, "one vector per electrode:\n22 × 512 numbers per trial", ha="center", va="center",
            color=GREY, fontsize=9, linespacing=1.4)
    ax.text(0.1, -0.45, "Five of the 22 channels are drawn.", color=GREY, fontsize=9)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.6, 4.0)
    fig.savefig(OUT / "reve_tokens.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def finetune_stages():
    """Three ways to use the pretrained model: frozen with a probe, head only, every weight."""
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150, facecolor="white")
    ax.set_axis_off()
    frozen, learns = dict(color="#F3F4F6", edge=GREY), dict(color="#FDF1E7", edge=ORANGE, weight="bold")
    rows = [(3.2, "Notebook 3: frozen model and linear probe", frozen, "logistic\nregression learns", learns),
            (1.7, "Notebook 4, stage 1: only the head learns", frozen, "head\nlearns", learns),
            (0.2, "Notebook 4, stage 2: every weight learns, in small steps", learns, "head\nlearns", learns)]
    for y0, title, encoder_style, head_text, head_style in rows:
        ax.text(0.1, y0 + 1.12, title, color=NAVY, fontsize=10.5, fontweight="bold")
        box(ax, 0.1, y0, 1.9, 0.9, "trial", edge=GREY)
        arrow(ax, (2.05, y0 + 0.45), (2.55, y0 + 0.45))
        encoder_text = "REVE encoder: 69 million weights, " + ("unchanged" if encoder_style is frozen else "all adjusted")
        box(ax, 2.6, y0, 5.6, 0.9, encoder_text, fontsize=9.5, **encoder_style)
        arrow(ax, (8.25, y0 + 0.45), (8.75, y0 + 0.45))
        box(ax, 8.8, y0, 2.3, 0.9, head_text, fontsize=9, **head_style)
        arrow(ax, (11.15, y0 + 0.45), (11.65, y0 + 0.45))
        box(ax, 11.7, y0, 1.6, 0.9, "left or\nright hand", color="white", edge=NAVY, fontsize=9.5)
    ax.text(13.5, 4.5, "grey: fixed", color=GREY, fontsize=9.5)
    ax.text(13.5, 4.2, "orange: learns from\nour labels", color=ORANGE, fontsize=9.5, va="top")
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 4.8)
    fig.savefig(OUT / "finetune_stages.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def gradient_descent():
    """Training as walking downhill on the loss: step size is the learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=150, facecolor="white", sharey=True)
    loss = lambda w: 0.35 * (w - 1.5) ** 2 + 0.4
    slope = lambda w: 0.7 * (w - 1.5)
    w = np.linspace(-2.2, 5.2, 200)
    for ax, rate, title in [(axes[0], 0.6, "A small learning rate: steady progress"),
                            (axes[1], 2.75, "A large learning rate: the steps overshoot")]:
        ax.plot(w, loss(w), color=GREY, lw=2)
        point = -1.8
        for _ in range(6):
            step = point - rate * slope(point)
            ax.annotate("", xy=(step, loss(step)), xytext=(point, loss(point)),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, mutation_scale=10))
            ax.plot(point, loss(point), "o", color=ORANGE, ms=5)
            point = step
        ax.plot(point, loss(point), "o", color=ORANGE, ms=5)
        ax.set_title(title, loc="left", fontsize=10, color=NAVY)
        ax.set_xlabel("One weight of the model", color=GREY)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Loss (error on the training trials)", color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_descent.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def classical_pipeline():
    """Classical decoding in three steps, and which steps a foundation model replaces."""
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=150, facecolor="white")
    ax.set_axis_off()
    ax.text(0.1, 4.25, "Classical decoding", color=NAVY, fontsize=11, fontweight="bold")
    steps = [(0.1, 2.0, "one trial\n22 channels × 4 s", "white", GREY),
             (2.7, 2.6, "1  Preprocess\nfilter, re-reference", LIGHT, BLUE),
             (5.9, 2.6, "2  Extract features\nband power at C3 and C4", LIGHT, BLUE),
             (9.1, 2.3, "3  Train a model\nlogistic regression", "#FDF1E7", ORANGE),
             (12.0, 1.6, "left or\nright hand", "white", NAVY)]
    for x, w, text, color, edge in steps:
        box(ax, x, 2.9, w, 1.0, text, color=color, edge=edge, fontsize=9.5)
    for x in (2.15, 5.35, 8.55, 11.45):
        arrow(ax, (x, 3.4), (x + 0.5, 3.4))
    ax.plot([2.7, 8.5], [2.7, 2.7], color=BLUE, lw=1.5)
    ax.text(5.6, 2.45, "designed by a person who knows the physiology", color=BLUE, fontsize=9.5, ha="center")
    ax.plot([9.1, 11.4], [2.7, 2.7], color=ORANGE, lw=1.5)
    ax.text(10.25, 2.45, "learned from labelled trials", color=ORANGE, fontsize=9.5, ha="center")

    ax.text(0.1, 1.65, "With a foundation model (notebooks 3 and 4)", color=NAVY, fontsize=11, fontweight="bold")
    box(ax, 0.1, 0.3, 2.0, 1.0, "one trial\n22 channels × 4 s", color="white", edge=GREY, fontsize=9.5)
    arrow(ax, (2.15, 0.8), (2.65, 0.8))
    box(ax, 2.7, 0.3, 5.8, 1.0, "a pretrained network makes the features", color="#F3F4F6", edge=GREY, fontsize=9.5)
    arrow(ax, (8.55, 0.8), (9.05, 0.8))
    box(ax, 9.1, 0.3, 2.3, 1.0, "3  Train a model", color="#FDF1E7", edge=ORANGE, fontsize=9.5)
    arrow(ax, (11.45, 0.8), (11.95, 0.8))
    box(ax, 12.0, 0.3, 1.6, 1.0, "left or\nright hand", color="white", edge=NAVY, fontsize=9.5)
    ax.text(5.6, 0.05, "learned beforehand from a large amount of unlabelled EEG", color=GREY, fontsize=9.5, ha="center")
    ax.set_xlim(0, 13.8)
    ax.set_ylim(-0.1, 4.5)
    fig.savefig(OUT / "classical_pipeline.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    setup_pipeline()
    trial_timeline()
    classical_pipeline()
    decoding_pipelines()
    tangent_space()
    foundation_model()
    reve_tokens()
    finetune_stages()
    gradient_descent()
