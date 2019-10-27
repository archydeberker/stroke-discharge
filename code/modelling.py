from typing import List, Generator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_predict, RandomizedSearchCV

from constants import CROSS_VAL, OUTCOME_DICT
import numpy as np

import pandas as pd

from utils import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)


def random_search(X, y, search_grid):
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=search_grid, n_iter=100, cv=CROSS_VAL, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)

    return rf_random


def grid_search(
    X,
    y,
    n_estimators: List,
    max_depth: List,
    min_samples_split: List,
    cross_val=CROSS_VAL,
):
    """Perform a grid search and return parameters and the results of the best performing model"""
    grid = {}
    for i1, n_est in enumerate(n_estimators):
        for i2, depth in enumerate(max_depth):
            for i3, min_samples in enumerate(min_samples_split):
                rf = RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, min_samples_split=min_samples
                )
                scores = cross_validate(rf, X, y, scoring=("accuracy"), cv=cross_val)
                grid[n_est, depth, min_samples] = scores["test_score"]

    return grid


def get_features(df):

    return {
        "Age": df["Age"].values.reshape(-1, 1),
        "NIHSS": df["NIHSS"].values.reshape(-1, 1),
        "MRS": pd.get_dummies(df["MRS"]),
        "Gender": pd.get_dummies(df["Gender"]),
    }


def find_and_evaluate_best_random_forest(features: List, y, outcome_dict=OUTCOME_DICT, cross_val=CROSS_VAL):

    X = np.concatenate(features, axis=1)

    estimators = range(1, 200, 50)
    depth = range(1, 10, 2)
    min_samples = range(2, 10, 2)

    logger.info('Commencing grid search')
    grid = grid_search(
        X, y, list(estimators), list(depth), list(min_samples), cross_val=cross_val
    )

    logger.info('Grid search complete')
    scores = get_best_result_from_grid(grid)
    rf = get_best_rf_from_grid(grid)
    logger.info('Best model found')
    return scores, rf


def generate_confusion_matrix(model, X, y, outcome_dict=OUTCOME_DICT):
    y_pred = cross_val_predict(model, X, y)
    sns.countplot([outcome_dict[yy] for yy in y_pred])
    plt.show()

    plot_confusion_matrix(confusion_matrix(y, y_pred), classes=outcome_dict.values())


def get_best_result_from_grid(grid):
    return max(grid.values(), key=lambda x: np.mean(x))


def get_best_rf_from_grid(grid):
    best_params = max(grid.keys(), key=lambda x: np.mean(grid[x]))
    return RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2],
    )
