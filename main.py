import argparse

from paths import DATA_DIR
from src.preprocessing import clean_adult, preprocess, oversample_classes
from src.plotting_metrics import get_f1_score, plot_confusion_matrix_display, plot_roc_curve, get_accuracy, get_roc_score
from src.util import save_params, load_params, check_pos_int_arg, check_pos_float_arg
from src.gridsearch import randomgridsearch, gridsearch
from parameter_grid import param_grid_rf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main(args):
        
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
    full_dset = preprocess(clean_combined, to_drop=["education", "fnlwgt", "race", "native-country"], one_hot=["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"])
    labels = full_dset.pop("salary")

    # Reserve 10% of the data to be the final testing set (grid search already performs cross validation)
    X_train, X_test, y_train, y_test = train_test_split(full_dset, labels, test_size=0.1, 
                                                        stratify=labels, random_state=r_seed)

    # Oversample the data
    if args.oversample_type == "SMOTE":
        X_train, y_train = oversample_classes(X_train, y_train, strategy="SMOTE", ratio=args.sampling_strategy)
    elif args.oversample_type == "random":
        X_train, y_train = oversample_classes(X_train, y_train, strategy="random", ratio=args.sampling_strategy)
    else:
        print("No oversampling will be used.")

    # Parameter selection
    if args.model_param == "load":
        model_params = load_params(args.param_file)
        estimator = RandomForestClassifier(**model_params)

    elif args.model_param == "random_search":
        temp = RandomForestClassifier(random_state=r_seed)
        print("\nPerforming a random hyperparameter grid search with {} iterations and {} folds per iteration. This may take a minute...\n".format(
            args.random_search_iter, args.num_folds
        ))
        estimator = randomgridsearch(temp, X_train, y_train, param_grid_rf, r_seed, args.random_search_iter, 'f1_macro', num_folds=args.num_folds)

    elif args.model_param == "exhaustive":
        temp = RandomForestClassifier(random_state=r_seed)
        print("\nPerforming a random hyperparameter grid search with {} iterations and {} folds per iteration. This may take a minute...\n".format(
            args.random_search_iter, args.num_folds
        ))
        estimator = gridsearch(estimator, X_train, y_train, param_grid_rf, 'f1_macro', num_folds=5)

    # Run the kfold cross validation to get the best model


    # Use either random grid search (faster) of exhaustive grid search to search for optimal hypterparams

    #
    #best_rf_model = gridsearch(estimator, X_train, y_train, param_grid_rf, 'f1_macro', num_folds=5)

    best_rf_model_params = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 0.6, 'max_leaf_nodes': None, 
        'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 20, 'min_samples_split': 20, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 17, 'n_jobs': None, 
        'oob_score': False, 'random_state': 7, 'verbose': 2, 'warm_start': False}

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

    roc_auc = get_roc_score(y_test, best_rf_model.predict_proba(X_test))
    print("Area under ROC curve: {}".format(roc_auc))

    conf_mat = plot_confusion_matrix_display(y_test, test_preds)
    roc_curve = plot_roc_curve(y_test, test_preds, best_rf_model.predict_proba(X_test))
    roc_curve.plot()

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adult Census Prediction using Random Forest Classifier.")
    parser.add_argument("--model_param", action="store", choices=["exhaustive", "load", "random_search"])
    parser.add_argument("--random_search_iter", action="store", type=check_pos_int_arg, default=1000)
    parser.add_argument("--num_folds", action="store", type=check_pos_int_arg, default=5)
    parser.add_argument("--param_file", action="store", default="best_params.json")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--save_params", action="store_true")
    parser.add_argument("--oversample_type", action="store", default="SMOTE", choices=["SMOTE", "random", "none"])
    parser.add_argument("--sampling_strategy", action="store", type=check_pos_float_arg, default=0.5)

    args = parser.parse_args()

    if args.model_param == "load" and not args.param_file:
        print("\nNo path to parameter file specified with 'load' argument. Use --param_file <name of parameter file> to load the parameters.\n")
        raise FileNotFoundError("Parameter file not specified.")

    main(args)