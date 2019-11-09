""" This figure shows

a) the performance of all of the different combinations of predictors, with error bars
b) the confusion matrix for the best-performing model, on the test-set

"""
import logging
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

import constants
from data import load_data_with_gender, stratified_sample_df
from modelling import get_features
from utils import plot_confusion_matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_train_test_data():
    df = load_data_with_gender(
        new_data_path="../../../data/Discharge destinations including gender.xlsx",
        old_data_path="../../../data/Discharge destinations 2.xlsx",
    )

    df_train = stratified_sample_df(
        df, col="Outcome", frac=constants.train_frac, random_state=1234
    )
    df_test = df.loc[df.index.difference(df_train.index)]

    return df_train, df_test


def get_validation_results():
    validation_models = pickle.load(open('../../results/all_models_random.p', 'rb'))

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


def fit_and_test_best_model():
    df_train, df_test = get_train_test_data()
    validation_models = pickle.load(open('../../results/all_models_random.p', 'rb'))
    validation_results = pickle.load(open('../../results/all_scores_random.p', 'rb'))

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

    confusion_mtx = confusion_matrix(y_test, y_pred)

    return confusion_mtx


def plot_graph(collated_scores, confusion_mtx):

    fig, axes = plt.subplots(ncols=2, figsize=(constants.full_figure_width, constants.figure_height))

    flat = []
    model_map = {'+'.join(k): len(k) for k in collated_scores}
    collated_scores = {'+'.join(k): v for k, v in collated_scores.items()}

    for key, value in collated_scores.items():
        for item in value:
            flat.append({key: item})
    performance = pd.DataFrame.from_records(flat)
    performance['model'] = performance.apply(lambda x: x.idxmax(axis=1), axis=1)

    print(model_map)
    performance['test-accuracy'] = performance.apply(lambda x: x[collated_scores.keys()].max(), axis=1)
    print(performance)
    performance['number-of-variables'] = performance.model.map(model_map)
    performance = performance[['model', 'test-accuracy', 'number-of-variables']]

    b = sns.barplot(x='test-accuracy', y='model', hue='number-of-variables', data=performance, dodge=False,
                    ax=axes[0])

    plt.sca(axes[0])
    plt.legend([], [])
    plt.xlabel('Validation accuracy', fontsize=constants._xlabel_size)
    plt.ylabel('Model', fontsize=constants._ylabel_size)
    b.set_yticklabels(performance.model.unique(), fontsize=int(constants._xlabel_size * 0.8))

    plt.sca(axes[1])
    plot_confusion_matrix(confusion_mtx, classes=constants.OUTCOME_DICT.values(), normalize=True)
    plt.title('')
    plt.tight_layout(pad=0.4, w_pad=4, h_pad=1.0)

    with open('../../figures/fig3.png', 'wb') as fileout:
        fig.savefig(fileout, bbox_inches='tight')

    return fig


if __name__ == '__main__':
    collated_scores = get_validation_results()
    confusion_mtx_best_model = fit_and_test_best_model()

    fig = plot_graph(collated_scores, confusion_mtx_best_model)


