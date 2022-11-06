from paths import DATA_DIR
from src.preprocessing import clean_adult, preprocess, oversample_classes
from src.gridsearch import gridsearch, randomgridsearch
from src.plotting_metrics import get_f1_score, plot_confusion_matrix_display, plot_roc_curve, get_feature_importance, get_accuracy
from src.util import save_params

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    
    # Define the seed for the run
    r_seed = 7

    # Import data
    data_file_path = DATA_DIR / "adult.data"
    adult_data = pd.read_csv(data_file_path, names=[
        "age", "workclass", "fnlwgt", "education",
        "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week", "native-country",
        "salary"
    ])

    test_file_path = DATA_DIR / "adult.test"
    adult_test = pd.read_csv(test_file_path, names=[
        "age", "workclass", "fnlwgt", "education",
        "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week", "native-country",
        "salary"
    ])

    # Clean the data
    clean_data = clean_adult(adult_data)
    clean_test = clean_adult(adult_test)

    # Combine them to form a single dset, get labels and data
    clean_combined = pd.concat([clean_data, clean_test], ignore_index=True, axis='index')

    # Preprocess both datasets
    full_dset = preprocess(clean_combined, drop_education=True, one_hot=["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"])
    labels = full_dset.pop("salary")

    # Create the estimator
    estimator = RandomForestClassifier(random_state=r_seed)

    # Reserve 10% of the data to be the final testing set (grid search already performs cross validation)
    X_train, X_test, y_train, y_test = train_test_split(full_dset, labels, test_size=0.1, 
                                                        stratify=labels, random_state=r_seed)

    # Oversample the data
    #X_train, y_train = oversample_classes(X_train, y_train, strategy="random")

    # Find the feature importances

    # Run the kfold cross validation to get the best model
    param_grid_rf = {
        'n_estimators': [5, 10, 20, 50, 100, 150, 200],
        'max_depth': [None, 5, 10, 20, 50, 100, 200],
        'max_features': [None, 'sqrt', 'log2', ] + list(np.arange(0.5, 1, 0.1)),
        'min_samples_split': [2, 5, 10, 15, 20]
    }

    # Use either random grid search (faster) of exhaustive grid search to search for optimal hypterparams

    #best_rf_model = randomgridsearch(estimator, X_train, y_train, param_grid_rf, r_seed, 3000, 'f1_macro', num_folds=5)
    #best_rf_model = gridsearch(estimator, X_train, y_train, param_grid_rf, 'f1_macro', num_folds=5)

    best_rf_model_params = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 0.6, 'max_leaf_nodes': None, 
    'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 20, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 17, 'n_jobs': None, 'oob_score': False, 
    'random_state':7, 'verbose':2, 'warm_start':False}
    best_rf_model = RandomForestClassifier(**best_rf_model_params)
    best_rf_model.fit(X_train, y_train)

    print("-------------------------------------------------------")
    print("Best model Parameters:\n")
    print(best_rf_model.get_params())
    print("\n\n")
    print("-------------------------------------------------------")

    # Save the parameters to a json
    save_params(best_rf_model.get_params())

    # Get the results on the held out 10% of data
    test_preds = best_rf_model.predict(X_test)

    f1 = get_f1_score(y_test, test_preds)
    print("F1 Score: {}".format(f1))

    acc = get_accuracy(y_test, test_preds)
    print("Acc: {}".format(acc))

    conf_mat = plot_confusion_matrix_display(y_test, test_preds)
    roc_curve = plot_roc_curve(y_test, test_preds, best_rf_model.predict_proba(X_test))
    roc_curve.plot()

    plt.show()
