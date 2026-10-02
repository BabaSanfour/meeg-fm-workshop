"""Scoring helpers shared by the notebooks: train on day 1, test on day 2, the same way for every decoder."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def day2(make, inputs, y, subject, train, test, subjects):
    """One decoder per participant, fitted on day 1 and scored on day 2.

    make: function that builds a new scikit-learn pipeline. inputs: one row per trial.
    Returns a table with one row per participant: balanced accuracy, ROC AUC, seconds.
    """
    rows = []
    for s in subjects:
        person = subject == s
        start = time.time()
        model = make().fit(inputs[person & train], y[person & train])
        confidence = model.predict_proba(inputs[person & test])[:, list(model.classes_).index("right_hand")]
        rows.append({"subject": s,
                     "balanced_accuracy": balanced_accuracy_score(y[person & test], model.predict(inputs[person & test])),
                     "auc": roc_auc_score(y[person & test] == "right_hand", confidence),
                     "seconds": time.time() - start})
    return pd.DataFrame(rows)


def with_fewer_labels(make, inputs, y, subject, train, test, subjects, sizes, n_draws=5):
    """Mean day-2 balanced accuracy when only `size` random day-1 trials are used for training, for each size."""
    means = []
    for size in sizes:
        scores = []
        for s in subjects:
            person = subject == s
            inputs_train, y_train = inputs[person & train], y[person & train]
            if size >= len(y_train):
                subsets = [np.arange(len(y_train))]
            else:
                draws = StratifiedShuffleSplit(n_splits=n_draws, train_size=size, random_state=0)
                subsets = [keep for keep, _ in draws.split(inputs_train, y_train)]
            scores += [balanced_accuracy_score(y[person & test], make().fit(inputs_train[keep], y_train[keep])
                                               .predict(inputs[person & test])) for keep in subsets]
        means.append(np.mean(scores))
    return pd.Series(means, index=sizes)


def pooled(make, inputs, y, subject, train, test, subjects):
    """One decoder fitted on the day 1 of all participants; balanced accuracy on each participant's day 2."""
    model = make().fit(inputs[train], y[train])
    return pd.Series({s: balanced_accuracy_score(y[(subject == s) & test], model.predict(inputs[(subject == s) & test]))
                      for s in subjects})


def left_out(make, inputs, y, subject, test, subjects):
    """For each participant: a decoder fitted on the other participants (both days), scored on their day 2."""
    scores = {}
    for s in subjects:
        others, held_out = subject != s, (subject == s) & test
        scores[s] = balanced_accuracy_score(y[held_out], make().fit(inputs[others], y[others]).predict(inputs[held_out]))
    return pd.Series(scores)
