import itertools

import matplotlib.pyplot as plt
import numpy as np

_xlabel_size = 15
_ylabel_size = 15


def get_all_combinations_of_predictors(predictors):
    combinations = []
    for n_predictors in range(1, len(predictors) + 1):
        combinations.extend(list(itertools.combinations(predictors, n_predictors)))

    return combinations


def _format_number(number, normalize, include_raw, raw=None):
    if normalize and not include_raw:
        return f"{number:.2f}"
    elif not normalize:
        return f"{number}"
    elif normalize and include_raw:
        return f"{number:.2f} ({raw})"


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues,
        include_raw=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        raw = cm
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(
        tick_marks, classes, rotation=45, fontsize=int(_xlabel_size * 0.8), ha="right"
    )
    plt.yticks(tick_marks, classes, fontsize=int(_xlabel_size * 0.8))

    fmt = ".2f" if normalize else "d"
    if include_raw and normalize:
        fmt = '.2f (d)'

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            _format_number(cm[i,j], normalize, include_raw, raw= raw[i,j] if include_raw else None),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    # plt.tight_layout()
    plt.ylim([3.5, -0.5])
    plt.ylabel("True label", fontsize=_xlabel_size)
    plt.xlabel("Predicted label", fontsize=_ylabel_size)
