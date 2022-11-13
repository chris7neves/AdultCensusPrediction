import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def gridsearch(estimator, data, labels, parameter_space, scoring, num_folds=5):
    """
    Performs a grid search cross validation procedure using the estimator and data over the
    given parameter space. Can be used with any estimators from the sklearn library.

    Returns the most succesful model.
    """

    best_model = GridSearchCV(estimator, parameter_space, scoring=scoring, cv=num_folds, verbose=5)
    best_model.fit(data, labels)
    print("\nThe cross-validation {} of the chosen 'best model' is {}".format(best_model.scorer_, best_model.best_score_))

    return best_model.best_estimator_


def randomgridsearch(estimator, data, labels, parameter_space, r_seed, n_iter, scoring, num_folds=5):
    """
    Performs a grid search over a random subset of combinations from the parameter space.

    Returns the most succesful model.
    """

    best_model = RandomizedSearchCV(estimator, parameter_space, n_iter=n_iter, 
                                    scoring=scoring, cv=num_folds, n_jobs=-1, 
                                    random_state=r_seed, verbose=5)
    best_model.fit(data, labels)
    print("\nThe cross-validation {} of the chosen 'best model' is {}".format(best_model.scorer_, best_model.best_score_))

    return best_model.best_estimator_