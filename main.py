from paths import DATA_DIR
from src.preprocessing import clean_adult, preprocess
from src.gridsearch import gridsearch, randomgridsearch
from src.metrics import get_f1_score

import pandas as pd
import numpy as np

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

    # Run the kfold cross validation to get the best model
    param_grid_rf = {
        'n_estimators': [10, 20, 50, 100, 150, 200],
        'max_depth': [None, 5, 10, 20, 50, 100, 200],
        'max_features': [None, 'sqrt', 'log2', ] + list(np.arange(0.5, 1, 0.1)),
        'min_samples_split': [2, 4, 5, 10, 15, 20],
        'bootstrap': [True, False]
    }
    best_rf_model = randomgridsearch(estimator, full_dset, labels, param_grid_rf, r_seed, 1, 'f1_macro', num_folds=5)

    # Get the results on the held out 10% of data
    test_preds = best_rf_model.predict(X_test)

    f1 = get_f1_score(y_test, test_preds)

    print(f1)

    # Plot results using the unseen 10% of data