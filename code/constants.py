train_frac = 0.9
test_frac = 0.1

CROSS_VAL = 20
N_ITER = 100

OUTCOME_DICT = {0: "Death", 1: "Inpatient", 2: "Comm. Hosp.", 3: "Home"}
OUTCOMES = ["Home", "Comm. Hosp.", "Inpatient", "Death"]
features = ["MRS", "NIHSS", "Age", "Gender"]
n_estimators = range(1, 500, 50)
max_features = ["auto", "sqrt"]
max_depth = range(2, 100, 10)
min_samples_split = range(2, 20, 2)


SEARCH_GRID = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
}

full_figure_width = 18
figure_height = 8
_xlabel_size = 15
_ylabel_size = 15
