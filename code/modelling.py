import logging
import pickle
from typing import Generator, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (RandomizedSearchCV, cross_val_predict,
                                     cross_validate)
from sklearn.dummy import DummyClassifier


import constants
from constants import CROSS_VAL, N_ITER, OUTCOME_DICT, SEARCH_GRID
from data import load_data_with_gender, stratified_sample_df
from utils import plot_confusion_matrix
import os

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),  '..')


def get_results_dummy_model():
    df_train, _ = get_train_test_data()
    model = DummyClassifier(strategy='prior')
    X = pd.get_dummies(df_train['MRS'])
    y = df_train['Outcome'].map({v: k for k, v in constants.OUTCOME_DICT.items()}).values

    scores = cross_validate(model, X, y, scoring=("accuracy"), cv=constants.CROSS_VAL)
    return scores['test_score']


def random_search(X, y, search_grid, metric):
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=search_grid, n_iter=N_ITER, cv=CROSS_VAL, verbose=2,
                                   random_state=42, n_jobs=-1, scoring=metric)
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
        "Gender": pd.get_dummies(df["Gender"], drop_first=True  ),
    }


def find_and_evaluate_best_random_forest(features: List, y, metric):

    X = np.concatenate(features, axis=1)

    rf_random = random_search(X, y, SEARCH_GRID, metric)
    logger.info('Grid search complete')

    return rf_random


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


def fit_and_test_best_model():
    df_train, df_test = get_train_test_data()
    print('Row 3 of test df:')
    print(df_test.iloc[3])
    validation_models = pickle.load(open(os.path.join(BASE_DIR, 'code/results/all_models_random.p'), 'rb'))
    validation_results = pickle.load(open(os.path.join(BASE_DIR, 'code/results/all_scores_random.p'), 'rb'))

    best_predictors = max(validation_results, key=lambda x: validation_results[x])
    print(best_predictors)
    best_model = validation_models[best_predictors]

    features_train, features_test = get_features(df_train), get_features(df_test)
    y_train = df_train['Outcome'].map({v: k for k, v in constants.OUTCOME_DICT.items()}).values
    y_test = df_test['Outcome'].map({v: k for k, v in constants.OUTCOME_DICT.items()}).values

    X_train = np.concatenate([features_train[predictor] for predictor in best_predictors], axis=1)
    X_test = np.concatenate([features_test[predictor] for predictor in best_predictors], axis=1)

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    score = best_model.score(X_test, y_test)

    confusion_mtx = confusion_matrix(y_test, y_pred)

    return confusion_mtx, best_model, X_test, score


def get_train_test_data():
    df = load_data_with_gender()

    df_train = stratified_sample_df(
        df, col="Outcome", frac=constants.train_frac, random_state=1234
    )
    df_test = df.loc[df.index.difference(df_train.index)]

    return df_train, df_test


def get_validation_results(model_path='../../results/all_models_random.p'):
    validation_models = pickle.load(open(model_path, 'rb'))

    df_train, _ = get_train_test_data()
    features = get_features(df_train)
    y = df_train['Outcome'].map({v: k for k, v in constants.OUTCOME_DICT.items()}).values

    collated_scores = {}

    for predictors, model in validation_models.items():
        X = np.concatenate([features[predictor] for predictor in predictors], axis=1)

        logger.info(f"Starting for {' '.join(predictors)}")
        scores = cross_validate(model, X, y, scoring=("accuracy"), cv=constants.CROSS_VAL)
        collated_scores[predictors] = scores['test_score']
        logger.info(f"Finished for {' '.join(predictors)}")

    return collated_scores