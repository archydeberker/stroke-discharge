train_frac = 0.9
test_frac = 0.1
CROSS_VAL = 10
OUTCOME_DICT = {0: "Death", 1: "Inpatient", 2: "CH", 3: "Home"}

n_estimators = range(0, 2000, 50)
max_features = ['auto', 'sqrt']
max_depth = range(2, 100, 10)
min_samples_split = range(2, 20, 2)

SEARCH_GRID = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

