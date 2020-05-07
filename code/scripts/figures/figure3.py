""" This figure shows

a) the performance of all of the different combinations of predictors, with error bars
b) the confusion matrix for the best-performing model, on the test-set

"""
import logging
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import constants
from modelling import fit_and_test_best_model, get_validation_results
from utils import plot_confusion_matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)

np.random.seed(1234)


def plot_graph(collated_scores, confusion_mtx):

    fig, axes = plt.subplots(
        ncols=2, figsize=(constants.full_figure_width, constants.figure_height)
    )

    flat = []
    model_map = {"+".join(k): len(k) for k in collated_scores}
    collated_scores = {"+".join(k): v for k, v in collated_scores.items()}

    for key, value in collated_scores.items():
        for item in value:
            flat.append({key: item})
    performance = pd.DataFrame.from_records(flat)
    performance["model"] = performance.apply(lambda x: x.idxmax(axis=1), axis=1)

    print(model_map)
    performance["test-accuracy"] = performance.apply(
        lambda x: x[collated_scores.keys()].max(), axis=1
    )
    print(performance)
    performance["number-of-variables"] = performance.model.map(model_map)
    performance = performance[["model", "test-accuracy", "number-of-variables"]]

    b = sns.barplot(
        x="test-accuracy",
        y="model",
        hue="number-of-variables",
        data=performance,
        dodge=False,
        ax=axes[0],
    )

    plt.sca(axes[0])
    plt.legend([], [])
    plt.xlabel("Validation accuracy", fontsize=constants._xlabel_size)
    plt.ylabel("Model", fontsize=constants._ylabel_size)
    b.set_yticklabels(
        performance.model.unique(), fontsize=int(constants._xlabel_size * 0.8)
    )

    plt.sca(axes[1])
    plot_confusion_matrix(
        confusion_mtx, classes=constants.OUTCOME_DICT.values(), normalize=True, include_raw=True,
    )
    plt.title("")
    plt.tight_layout(pad=0.4, w_pad=4, h_pad=1.0)

    with open("../../figures/fig3.png", "wb") as fileout:
        fig.savefig(fileout, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    collated_scores = get_validation_results()
    confusion_mtx_best_model, _, _, score = fit_and_test_best_model()

    print("Best model result on test set:")
    pprint(score)

    fig = plot_graph(collated_scores, confusion_mtx_best_model)
