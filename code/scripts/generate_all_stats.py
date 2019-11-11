from pprint import pprint

from scipy.stats import stats

import statistics
import constants
import numpy as np
import itertools

from data import load_data_with_gender, stratified_sample_df
from modelling import get_validation_results, get_results_dummy_model


def _currier(func, df, x):
    return {x: lambda x: func(df, x)}


def run_univariate_stats(df):
    kruskal_results = list(map(lambda feature: {feature: statistics.compute_kruskal(df, feature)}, constants.features))
    return kruskal_results


def run_correlations_between_predictors(df):
    results = {}
    for combo in itertools.combinations(constants.features, 2):
        r, p = stats.pearsonr(df[combo[0]], df[combo[1]])
        results[combo] = dict(r=r, p=p)

    return results


def run_correlations_by_outcome(df):
    results = {}
    for combo in itertools.combinations(constants.features, 2):
        results[combo] = list(map(lambda outcome: {outcome: statistics.covariate_correlations_segregated_by_outcome(df, combo[0], combo[1], outcome)},
                         constants.OUTCOMES))

    return results


def demographics_of_split(df):

    df_train = stratified_sample_df(
        df, col="Outcome", frac=constants.train_frac, random_state=1234
    )
    df_test = df.loc[df.index.difference(df_train.index)]

    def _test_column_equality(feature):
        t, p = stats.ttest_ind(df_test[feature], df_train[feature])
        return {feature: dict(t=t,
                              p=p,
                              test_mean=df_train[feature].mean(),
                              train_mean=df_train[feature].mean(),
                              )}

    results = list(map(_test_column_equality, constants.features))
    return results


def stats_model_comparisons(model_path):
    """This performs t-tests based upon the results of cross-validating within the train set"""
    collated_scores = get_validation_results(model_path=model_path)
    dummy = get_results_dummy_model()

    # Test all of the single predictors
    results = {feature: stats.ttest_ind(collated_scores[(feature,)], dummy) for feature in constants.features}
    best_predictors = max(collated_scores, key=lambda x: np.mean(collated_scores[x]))

    results[best_predictors] = stats.ttest_ind(collated_scores[best_predictors], dummy)

    return results


if __name__ == '__main__':

    np.random.seed(1234)

    df = load_data_with_gender(
        new_data_path="../../data/Discharge destinations including gender.xlsx",
        old_data_path="../../data/Discharge destinations 2.xlsx",
    )

    kruskal_results = run_univariate_stats(df)
    corrs_between_predictors = run_correlations_between_predictors(df)
    corrs_between_predictors_by_outcome = run_correlations_by_outcome(df)
    split_differences = demographics_of_split(df)
    model_comparison_results = stats_model_comparisons(model_path='../results/all_models_random.p')

    with open('../results/stats.txt', 'w') as logfile:
        pprint(kruskal_results, logfile)
        pprint(corrs_between_predictors, logfile)
        pprint(corrs_between_predictors_by_outcome, logfile)
        pprint(split_differences, logfile)
        pprint(model_comparison_results, logfile)
