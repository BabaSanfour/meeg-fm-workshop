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
        label_y = -np.inf                                        # participant numbers, moved apart when they overlap
        for participant, value in b.sort_values().items():
            label_y = max(value, label_y + 2.5)
            ax.text(1.06, label_y, str(participant), va="center", fontsize=8, color="#5B6472")
        ax.plot([0, 1], [a.mean(), b.mean()], color=color, lw=2.5, marker="o")
        ax.axhline(50, color="#5B6472", lw=1, ls="--")
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{name}: {a.mean():.0f} % → {b.mean():.0f} %", loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Balanced accuracy (%)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- notebook 1: the data
def electrodes(highlight=("C3", "C4")):
    """The 22 electrodes seen from above, with some of them in orange."""
    import mne
    from matplotlib.colors import ListedColormap
    from .data import CHANNELS, head_info
    info, sphere = head_info()
    chosen = np.isin(CHANNELS, list(highlight))
    fig, ax = plt.subplots(figsize=(4.6, 4.6), layout="constrained")
    mne.viz.plot_sensors(info, show_names=True, axes=ax, show=False, pointsize=60, linewidth=0, sphere=sphere,
                         ch_groups=[np.flatnonzero(~chosen), np.flatnonzero(chosen)],
                         cmap=ListedColormap([GREY, "#E8710A"]))
    for text in ax.texts:
        text.set_fontsize(8)
    ax.set_title("The 22 electrodes, seen from above (nose at the top)", fontsize=10)
    plt.show()


def time_frequency(maps, title=""):
    """Time-frequency maps (% change from the baseline) at C3 and C4, one row per movement."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharex=True, sharey=True, layout="constrained")
    for row, cls in zip(axes, HANDS):
        for ax, ch in zip(row, ["C3", "C4"]):
            maps[cls].plot(picks=ch, axes=ax, vlim=(-70, 70), cmap="RdBu_r", colorbar=False, show=False)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Time from the cue (s)" if cls == "right_hand" else "")
            ax.set_title(f"{title}{cls.replace('_', ' ')} imagery, {ch} ({'left' if ch == 'C3' else 'right'} side)",
                         loc="left", fontsize=10, color=HANDS[cls])
    fig.colorbar(axes[0, 0].images[0], ax=axes, label="Change of power (%)", shrink=0.8)
    plt.show()


def head_maps(maps, bands, window):
    """Mean change of power in each band during `window`, on the head: one row per movement, one column per band.

    maps: {movement: time-frequency map}. bands: {name: (low, high) in Hz}. C3 and C4 are circled.
    """
    import mne
    from .data import CHANNELS, head_info
    info, sphere = head_info()
    circled = np.isin(CHANNELS, ["C3", "C4"])
    one = len(bands) == 1
    fig, axes = plt.subplots(*((1, 2) if one else (2, len(bands))), layout="constrained",
                             figsize=(8, 3.8) if one else (2.6 * len(bands), 5.4))
    columns = [axes] if one else axes.T
    for column, (name, band) in zip(columns, bands.items()):
        for ax, cls in zip(column, HANDS):
            values = maps[cls].get_data(fmin=band[0], fmax=band[1], tmin=window[0], tmax=window[1]).mean(axis=(1, 2))
            image, _ = mne.viz.plot_topomap(values, info, sphere=sphere, axes=ax, vlim=(-60, 60), cmap="RdBu_r",
                                            contours=4, show=False, mask=circled,
                                            mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="k",
                                                             markersize=9))
            hand = cls.replace("_", " ")
            ax.set_title(f"{hand} imagery" if one else f"{name} ({band[0]}-{band[1]} Hz)\n{hand}",
                         fontsize=10 if one else 9, color=HANDS[cls])
    label = f"Change of {next(iter(bands))} power (%)" if one else "Change of power (%)"
    fig.colorbar(image, ax=axes, label=label, shrink=0.75 if one else 0.6)
    plt.show()


def sides(summary):
    """One line per participant: change of power on the same side as the hand and on the opposite side, per band."""
    bands = list(dict.fromkeys(summary.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(bands), figsize=(4.5 * len(bands), 4), sharey=True)
    for ax, band in zip(np.atleast_1d(axes), bands):
        table = summary.loc[band]
        stronger = int((table["opposite"] < table["same"]).sum())
        ax.plot([0, 1], table[["same", "opposite"]].T, color=GREY, lw=1, marker="o", ms=4)
        label_y = -np.inf                                        # participant numbers, moved apart when they overlap
        for subject, value in table["opposite"].sort_values().items():
            label_y = max(value, label_y + 4)
            ax.text(1.06, label_y, str(subject), va="center", fontsize=8, color="#5B6472")
        ax.plot([0, 1], table[["same", "opposite"]].mean(), color=DARK, lw=2.5, marker="o", label="mean")
        ax.axhline(0, color="#5B6472", lw=0.8)
        ax.set_xticks([0, 1], ["same side\nas the hand", "opposite side"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{band}: larger drop on the opposite side\nin {stronger} of {len(table)} participants",
                     loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Change of power during imagery (%)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    plt.show()


def feature_planes(features, y, bands, title=""):
    """The trials in the plane (feature at C3, feature at C4), one panel per band. features: table with '<band> C3' columns."""
    fig, axes = plt.subplots(1, len(bands), figsize=(4.75 * len(bands), 4), layout="constrained")
    for ax, band in zip(np.atleast_1d(axes), bands):
        for cls, color in HANDS.items():
            rows = np.asarray(y) == cls
            ax.scatter(features.loc[rows, f"{band} C3"], features.loc[rows, f"{band} C4"], s=12, color=color, alpha=0.6,
                       label=cls.replace("_", " "))
        ax.set_xlabel(f"log {band} power at C3")
        ax.set_ylabel(f"log {band} power at C4")
        ax.set_title(f"{title}{band}: one dot per trial", loc="left", fontsize=10)
        ax.locator_params(nbins=5)
    np.atleast_1d(axes)[-1].legend(title="imagined hand")
    plt.show()


def weights(values, title=""):
    """The weight a linear model gave to each feature (blue: towards left hand, orange: towards right hand)."""
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(values.index, values, color=["#0B5CAD" if value < 0 else "#E8710A" for value in values])
    ax.axhline(0, color="#5B6472", lw=0.8)
    ax.set_ylabel("Weight")
    ax.set_title(title, loc="left", fontsize=10)
    plt.show()


def scores(values, title=""):
    """One bar per participant: balanced accuracy on day 2, in %."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(range(len(values)), values, color="#0B5CAD")
    ax.axhline(50, color="#5B6472", lw=1, ls="--")
    ax.text(len(values) - 0.45, 51, "chance", color="#5B6472", fontsize=9, va="bottom")
    ax.set_xticks(range(len(values)), values.index)
    ax.set_xlim(-0.7, len(values) + 0.3)
    ax.set_ylim(40, 100)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Balanced accuracy on day 2 (%)")
    ax.set_title(title, loc="left", fontsize=10)
    plt.show()


# ---------------------------------------------------------------- notebook 2: classical decoders
def features_scatter(features, y, labels, title=""):
    """The trials in the plane of two features, coloured by imagined hand."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for cls, color in HANDS.items():
        rows = np.asarray(y) == cls
        ax.scatter(features[rows, 0], features[rows, 1], s=14, color=color, alpha=0.7, label=cls.replace("_", " "))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(title="imagined hand")
    plt.show()


def features_with_line(features, y, days, classifier, labels):
    """The trials of each day in the plane of two features; the line is where the classifier changes its answer.

    days: {name: mask of the trials of that day}. classifier: fitted on these two features.
    """
    from sklearn.metrics import balanced_accuracy_score
    fig, axes = plt.subplots(1, len(days), figsize=(4.75 * len(days), 4), sharex=True, sharey=True)
    low, high = features.min(axis=0) - 0.2, features.max(axis=0) + 0.2
    grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], 200), np.linspace(low[1], high[1], 200))
    side = classifier.decision_function(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    for ax, (name, day) in zip(axes, days.items()):
        for cls, color in HANDS.items():
            rows = day & (np.asarray(y) == cls)
            ax.scatter(features[rows, 0], features[rows, 1], s=14, color=color, alpha=0.7, label=cls.replace("_", " "))
        ax.contour(grid_x, grid_y, side, levels=[0], colors=DARK, linewidths=1.5)
        score = balanced_accuracy_score(np.asarray(y)[day], classifier.predict(features[day]))
        ax.set_title(f"{name}: {100 * score:.0f} % correct", loc="left", fontsize=10)
        ax.set_xlabel(labels[0])
    axes[0].set_ylabel(labels[1])
    axes[-1].legend(title="imagined hand")
    plt.show()


def confusion_and_roc(y_true, predicted, confidence, name):
    """Confusion matrix of the answers, and ROC curve of the confidence in "right hand"."""
    from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ConfusionMatrixDisplay.from_predictions(y_true, predicted, ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion matrix (number of trials)", loc="left", fontsize=10)
    is_right = np.asarray(y_true) == "right_hand"
    false_alarms, found, _ = roc_curve(is_right, confidence)
    axes[1].plot(false_alarms, found, color=DECODERS.get(name, DARK), lw=2,
                 label=f"{name} (AUC {roc_auc_score(is_right, confidence):.2f})")
    axes[1].plot([0, 1], [0, 1], color=GREY, ls="--", label="guessing")
    axes[1].set_xlabel("Left-hand trials called right (proportion)")
    axes[1].set_ylabel("Right-hand trials found (proportion)")
    axes[1].set_title("ROC curve", loc="left", fontsize=10)
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def covariance(mean_left, mean_right):
    """The mean covariance matrix of left-hand trials, and the difference right minus left."""
    from .data import CHANNELS
    ticks = [CHANNELS.index(name) for name in ["C3", "Cz", "C4", "Pz"]]
    difference = mean_right - mean_left
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), layout="constrained")
    image = axes[0].imshow(mean_left, cmap="viridis", vmin=0)
    axes[0].set_title("Left hand: mean covariance", loc="left", fontsize=10)
    fig.colorbar(image, ax=axes[0], label="µV²", shrink=0.85)
    image = axes[1].imshow(difference, cmap="RdBu_r", vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
    axes[1].set_title("Right hand minus left hand", loc="left", fontsize=10)
    fig.colorbar(image, ax=axes[1], label="µV²", shrink=0.85)
    for ax in axes:
        ax.set_xticks(ticks, ["C3", "Cz", "C4", "Pz"])
        ax.set_yticks(ticks, ["C3", "Cz", "C4", "Pz"])
    plt.show()


# ---------------------------------------------------------------- notebook 5: the comparison
def score_table(table, threshold):
    """Balanced accuracy on day 2 as a table of colours; scores that guessing could reach are in grey."""
    subjects = list(table.columns)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    image = ax.imshow(100 * table, cmap="YlGnBu", vmin=50, vmax=100, aspect="auto")
    for i, method in enumerate(table.index):
        for j, subject in enumerate(subjects):
            value = table.loc[method, subject]
            ax.text(j, i, f"{100 * value:.0f}", ha="center", va="center", fontsize=10,
                    color=GREY if value < threshold else ("white" if value > 0.8 else DARK))
        ax.text(len(subjects) - 0.3, i, f"mean {100 * table.loc[method].mean():.0f}", va="center", fontsize=10, color=DARK)
    ax.set_xticks(range(len(subjects)), subjects)
    ax.set_yticks(range(len(table)), table.index)
    ax.set_xlabel("Participant")
    ax.set_xlim(-0.5, len(subjects) + 0.7)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, label="Balanced accuracy on day 2 (%)", pad=0.02)
    plt.show()


def differences(differences, reference):
    """For each decoder: every participant's difference with the reference decoder (points), and the mean."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rng = np.random.default_rng(0)
    for k, (method, row) in enumerate(differences.iterrows()):
        color = DECODERS.get(method, DARK)
        x = k + rng.uniform(-0.12, 0.12, len(row))
        ax.scatter(x, row, s=28, color=color, alpha=0.8)
        for xi, (subject, value) in zip(x, row.items()):
            ax.text(xi + 0.04, value, str(subject), fontsize=7, color="#5B6472", va="center")
        ax.plot([k - 0.25, k + 0.25], [row.mean(), row.mean()], color=color, lw=3)
        ax.text(k + 0.28, row.mean(), f"{row.mean():+.1f}", color=color, va="center", fontweight="bold")
    ax.axhline(0, color="#5B6472", lw=1, ls="--")
    ax.set_xticks(range(len(differences)), differences.index)
    ax.set_xlim(-0.5, len(differences) - 0.3)
    ax.set_ylabel(f"Difference with {reference}\n(points of balanced accuracy)")
    plt.show()


def bootstrap(means, interval, observed, reference, name="REVE fine-tuned"):
    """The bootstrap distribution of a mean difference, with its 95 % interval."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(means, bins=50, color=DECODERS.get(name, DARK), alpha=0.7)
    for edge in interval:
        ax.axvline(edge, color=DARK, lw=1.5)
    ax.axvline(0, color="#5B6472", lw=1, ls="--")
    ax.set_xlabel(f"Mean difference with {reference} (points)")
    ax.set_ylabel("Resamples")
    ax.set_title(f"Mean {observed:+.1f} points, 95 % interval {interval[0]:+.1f} to {interval[1]:+.1f}", loc="left", fontsize=10)
    plt.show()


def conditions(own, pooled, curve, finetuned_labels, seconds, accuracy):
    """Three views of the same decoders: whose trials they learn from, how many labels, and what they cost.

    own, pooled: mean score per decoder trained on one participant / on everyone. curve: sizes x decoders.
    finetuned_labels: {labels per participant: score} for the fine-tuned model. seconds, accuracy: per decoder.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method, color in DECODERS.items():
        axes[0].plot([0, 1], [100 * own[method], 100 * pooled[method]], color=color, marker="o", lw=2, label=method)
    axes[0].set_xticks([0, 1], ["own day 1", "everyone's day 1"])
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].set_title("Whose trials it learns from", loc="left", fontsize=10)
    axes[0].legend(fontsize=8)

    for method in curve.columns:
        axes[1].plot(curve.index, 100 * curve[method], color=DECODERS[method], marker="o", lw=2)
    axes[1].plot(list(finetuned_labels), [100 * v for v in finetuned_labels.values()], color=DECODERS["REVE fine-tuned"],
                 marker="o", lw=2)
    axes[1].set_xscale("log")
    axes[1].set_xticks(list(curve.index), list(curve.index))
    axes[1].minorticks_off()
    axes[1].set_xlabel("Labelled trials per participant")
    axes[1].set_title("How many labels it gets", loc="left", fontsize=10)

    for method, color in DECODERS.items():
        axes[2].scatter(seconds[method], accuracy[method], s=70, color=color)
        axes[2].annotate(method, (seconds[method], accuracy[method]), textcoords="offset points", xytext=(8, 4),
                         fontsize=8, color=color)
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.1, 20000)
    axes[2].set_xlabel("Training time for the 9 participants (s)")
    axes[2].set_title("What it costs", loc="left", fontsize=10)
    for ax in axes:
        ax.set_ylabel("Balanced accuracy on day 2 (%)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- notebook 4: fine-tuning
def history(history, title=""):
    """Loss on the training trials, and accuracy on the training and held-out trials, pass by pass."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    passes = history["epoch"]
    axes[0].plot(passes, history["loss"], color=DARK, marker="o", lw=2)
    axes[0].axhline(np.log(2), color="#5B6472", lw=1, ls="--")
    axes[0].text(passes.max(), np.log(2), "guessing", color="#5B6472", fontsize=9, va="bottom", ha="right")
    axes[0].set_ylabel("Loss on the training trials")
    axes[1].plot(passes, 100 * history["train_accuracy"], color=GREY, marker="o", lw=2, label="training trials")
    axes[1].plot(passes, 100 * history["eval_accuracy"], color=DECODERS["REVE fine-tuned"], marker="o", lw=2,
                 label="held-out trials (day 1)")
    axes[1].axhline(50, color="#5B6472", lw=1, ls="--")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="lower right")
    n_head = (history["stage"] == "head only").sum()
    for ax in axes:
        if 0 < n_head < len(history):
            ax.axvspan(0.5, n_head + 0.5, color="#F3F4F6", zorder=0)
            ax.text(0.6, 0.02, "head only", transform=ax.get_xaxis_transform(), color="#5B6472", fontsize=9)
        ax.set_xlabel("Pass over the training trials")
        ax.set_xticks(passes)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    plt.tight_layout()
    plt.show()


def seeds(saved_history, runs):
    """Held-out accuracy pass by pass, one line per random seed, one panel per run. runs: {run name: panel title}."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.25 * len(runs), 3.6), sharey=True)
    for ax, (run, title) in zip(np.atleast_1d(axes), runs.items()):
        for seed, rows in saved_history[saved_history["run"] == run].groupby("seed"):
            ax.plot(rows["epoch"], 100 * rows["eval_accuracy"], marker="o", ms=3, lw=1.5, label=f"seed {seed}")
        ax.axhline(50, color="#5B6472", lw=1, ls="--")
        ax.set_xlabel("Pass over the training trials")
        ax.set_title(title, loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy on held-out trials (%)")
    np.atleast_1d(axes)[-1].legend()
    plt.tight_layout()
    plt.show()
