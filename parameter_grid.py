import numpy as np

# Grid of parameters to search across for gridsearches.
# Modify this grid if the parameter space needs to be changed.

param_grid_rf = {
    'n_estimators': [5, 10, 20, 50, 100, 150, 200],
    'max_depth': [None, 5, 10, 20, 50, 100, 200],
    'max_features': [None, 'sqrt', 'log2', ] + list(np.arange(0.5, 1, 0.1)),
    'min_samples_split': [2, 5, 10, 15, 20]
}