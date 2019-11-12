import logging
import pickle

import numpy as np

import constants
from data import load_data_with_gender, stratified_sample_df
from modelling import find_and_evaluate_best_random_forest, get_features
from utils import get_all_combinations_of_predictors

np.random.seed(1234)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":

    df = load_data_with_gender(
        new_data_path="../../data/Discharge destinations including gender.xlsx",
        old_data_path="../../data/Discharge destinations 2.xlsx",
    )

    df_train = stratified_sample_df(
        df, col="Outcome", frac=constants.train_frac, random_state=1234
    )
    df_test = df.loc[df.index.difference(df_train.index)]

    logger.info(f"Train set is {len(df_train)}, test set is {len(df_test)}")

    features = get_features(df_train)
    y = (
        df_train["Outcome"]
        .map({v: k for k, v in constants.OUTCOME_DICT.items()})
        .values
    )

    all_combinations = get_all_combinations_of_predictors(
        ["Age", "MRS", "NIHSS", "Gender"]
    )
    all_scores = {}
    all_models = {}
    for combo in all_combinations:
        logger.info(f"Commencing for {' '.join(combo)}")

        rf_random = find_and_evaluate_best_random_forest(
            [features[x] for x in combo], y, metric="f1_micro"
        )

        all_scores[combo], all_models[combo] = (
            rf_random.best_score_,
            rf_random.best_estimator_,
        )

        logging.info(f"Mean accuracy for {' '.join(combo)} is {all_scores[combo]}")

        pickle.dump(all_scores, open("../results/all_f1_scores_random.p", "wb"))
        pickle.dump(all_models, open("../results/all_f1_models_random.p", "wb"))
