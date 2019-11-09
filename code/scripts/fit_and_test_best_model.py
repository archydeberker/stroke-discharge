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


    scores = cross_validate(rf, X, y, scoring=("accuracy"), cv=cross_val)
