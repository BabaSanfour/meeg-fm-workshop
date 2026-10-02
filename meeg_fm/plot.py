"""Figures used by the notebooks, kept here so that the notebooks show the decoding code and not matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

HANDS = {"left_hand": "#1baf7a", "right_hand": "#4a3aa7"}
DECODERS = {"CSP + LDA": "#0B5CAD", "Tangent space + LR": "#E8710A", "REVE frozen + LR": "#7C3AED",
            "REVE fine-tuned": "#BE185D"}
GREY, DARK = "#9AA3AF", "#0B2545"


def style():
    """The look shared by every figure of the tutorial."""
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "legend.frameon": False})


def embeddings(Z, y, title=""):
    """The embedding of the first left-hand and the first right-hand trial: electrodes x numbers."""
    from .data import CHANNELS
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, layout="constrained")
    ticks = [CHANNELS.index(name) for name in ["Fz", "C3", "Cz", "C4", "Pz"]]
    for ax, cls in zip(axes, HANDS):
        trial = np.flatnonzero(y == cls)[0]
        image = ax.imshow(Z[trial], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
        ax.set_title(f"{title}first {cls.replace('_', '-')} trial", loc="left", fontsize=10, color=HANDS[cls])
        ax.set_xlabel(f"The {Z.shape[-1]} numbers of each electrode")
    axes[0].set_yticks(ticks, ["Fz", "C3", "Cz", "C4", "Pz"])
    axes[0].set_ylabel("Electrode")
    fig.colorbar(image, ax=axes, label="Value", shrink=0.85)
    plt.show()


def plane(points, subject, y, name=""):
    """Every trial in a plane, coloured by participant (left) and by imagined hand (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    dots = axes[0].scatter(points[:, 0], points[:, 1], c=subject, cmap="tab10", s=4)
    axes[0].legend(*dots.legend_elements(), title="participant", ncols=2, fontsize=8, loc="upper left",
                   bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title(f"{name}coloured by participant", loc="left", fontsize=10)
    for cls, color in HANDS.items():
        axes[1].scatter(points[y == cls, 0], points[y == cls, 1], s=4, color=color, alpha=0.6, label=cls.replace("_", " "))
    axes[1].legend(title="imagined hand", markerscale=3, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[1].set_title(f"{name}coloured by imagined hand", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Direction 1")
    axes[0].set_ylabel("Direction 2")
    plt.tight_layout()
    plt.show()


def decision_values(decision, y, days, scores):
    """Histogram of a classifier's decision values for each hand, on each day. days: {name: mask}."""
    fig, axes = plt.subplots(1, len(days), figsize=(10, 3.4), sharey=True)
    for ax, (name, day) in zip(axes, days.items()):
        bins = np.linspace(decision[day].min(), decision[day].max(), 30)
        for cls, color in HANDS.items():
            ax.hist(decision[day & (y == cls)], bins=bins, color=color, alpha=0.6, label=cls.replace("_", " "))
        ax.axvline(0, color=DARK, lw=1.2, ls="--")
        ax.set_title(f"{name}: {100 * scores[name]:.0f} % correct", loc="left", fontsize=10)
        ax.set_xlabel("Decision value of the probe")
    axes[0].set_ylabel("Number of trials")
    axes[-1].legend(title="imagined hand")
    plt.tight_layout()
    plt.show()


def head_scores(scores, title):
    """Scores of one probe per electrode (in %), drawn on the head with the electrode names."""
    import mne
    from .data import CHANNELS, head_info
    info, sphere = head_info()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image, _ = mne.viz.plot_topomap(scores, info, sphere=sphere, axes=ax, cmap="Purples", vlim=(50, scores.max()),
                                    names=CHANNELS, sensors="k,", contours=0, show=False)
    for text in ax.texts:
        text.set_fontsize(7)
        text.set_va("bottom")
    fig.colorbar(image, ax=ax, label="Balanced accuracy on day 2 (%)", shrink=0.75)
    ax.set_title(title, fontsize=10)
    plt.show()


def bars(table, colors=None):
    """Balanced accuracy on day 2 per participant, one bar per decoder. table: decoders x participants (+ mean)."""
    colors = colors or {name: DECODERS.get(name, GREY) for name in table.index}
    subjects = [c for c in table.columns if c != "mean"]
    width = 0.8 / len(colors)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, (name, color) in enumerate(colors.items()):
        offset = width * (k - (len(colors) - 1) / 2)
        ax.bar(np.arange(len(subjects)) + offset, 100 * table.loc[name, subjects], width=0.92 * width, color=color,
               label=f"{name} (mean {100 * table.loc[name, subjects].mean():.0f} %)")
    ax.axhline(50, color="#5B6472", lw=1, ls="--")
    ax.text(len(subjects) - 0.4, 51, "chance", color="#5B6472", fontsize=9, va="bottom")
    ax.set_xlim(-0.7, len(subjects) + 0.25)
    ax.set_xticks(range(len(subjects)), subjects)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Balanced accuracy on day 2 (%)")
    ax.set_ylim(40, 100)
    ax.legend(loc="upper left", ncols=min(3, len(colors)), bbox_to_anchor=(0, 1.22 if len(colors) > 3 else 1.15), fontsize=9)
    plt.show()


def label_curve(curve, colors=None):
    """Mean balanced accuracy on day 2 against the number of training trials. curve: sizes x decoders."""
    colors = colors or {name: DECODERS.get(name, GREY) for name in curve.columns}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, color in colors.items():
        ax.plot(curve.index, 100 * curve[name], color=color, marker="o", lw=2, label=name)
    ax.axhline(50, color="#5B6472", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(list(curve.index), list(curve.index))
    ax.minorticks_off()
    ax.set_xlabel("Training trials from day 1")
    ax.set_ylabel("Balanced accuracy on day 2 (%)")
    ax.legend()
    plt.show()


def before_after(before, after, labels, colors=None):
    """One panel per decoder: each participant's score in two settings, and the mean. before/after: decoders x participants."""
    colors = colors or {name: DECODERS.get(name, GREY) for name in before.index}
    fig, axes = plt.subplots(1, len(colors), figsize=(3.7 * len(colors), 3.8), sharey=True)
    for ax, (name, color) in zip(np.atleast_1d(axes), colors.items()):
        a, b = 100 * before.loc[name], 100 * after.loc[name]
        ax.plot([0, 1], [a, b], color=GREY, lw=1, marker="o", ms=4)
        ax.plot([0, 1], [a.mean(), b.mean()], color=color, lw=2.5, marker="o")
        ax.axhline(50, color="#5B6472", lw=1, ls="--")
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{name}: {a.mean():.0f} % → {b.mean():.0f} %", loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Balanced accuracy on day 2 (%)")
    plt.tight_layout()
    plt.show()
