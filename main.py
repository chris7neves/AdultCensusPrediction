import argparse
import pprint
from datetime import datetime

from paths import DATA_DIR
from src.preprocessing import clean_adult, preprocess, oversample_classes
from src.plotting_metrics import (get_f1_score, plot_confusion_matrix_display, 
    plot_roc_curve, get_accuracy, get_roc_score, get_precision_recall, get_feature_importance)
from src.util import save_params, load_params, check_pos_int_arg, check_pos_float_arg, save_plot
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
    r_seed = 10

    # Import data
    print("Importing data...")
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
    print("Importing done.")

    # Clean the data
    print("Cleaning dataset...")
    clean_data = clean_adult(adult_data)
    clean_test = clean_adult(adult_test)
    print("Cleaning done.")

    # Combine them to form a single dset, get labels and data
    print("Preprocessing dataset...")
    clean_combined = pd.concat([clean_data, clean_test], ignore_index=True, axis='index')

    # Preprocess both datasets
    full_dset = preprocess(clean_combined, to_drop=["education", "fnlwgt", "race", "native-country"], one_hot=["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"])
    labels = full_dset.pop("salary")
    print("Preprocessing done.")

    # Reserve 10% of the data to be the final testing set (grid search already performs cross validation)
    X_train, X_test, y_train, y_test = train_test_split(full_dset, labels, test_size=0.1, 
                                                        stratify=labels, random_state=r_seed)

    # Print feature importances
    if args.print_importances:
        fi = get_feature_importance(full_dset, labels, r_seed=r_seed)
        print(fi)

    # Oversample the data
    if args.oversample_type == "SMOTE":
        X_train, y_train = oversample_classes(X_train, y_train, strategy="SMOTE", ratio=args.sampling_strategy)
    elif args.oversample_type == "random":
        X_train, y_train = oversample_classes(X_train, y_train, strategy="random", ratio=args.sampling_strategy)
    else:
        print("No oversampling will be used.")

    # Parameter selection strategy
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
        print("\nPerforming an exhaustive hyperparameter grid search with {} folds per iteration. This may take a minute...\n".format(
            args.num_folds
        ))
        estimator = gridsearch(estimator, X_train, y_train, param_grid_rf, 'f1_macro', num_folds=args.num_folds)

    estimator.fit(X_train, y_train)

    print("\nModel Parameters:")
    print("-------------------------------------------------------")
    pprint.pprint(estimator.get_params())
    print("-------------------------------------------------------\n")

    # Save the parameters to a json
    if args.save_params:
        print("Saving parameters to params/")
        save_params(estimator.get_params())

    # Get the results on the held out 10% of data
    test_preds = estimator.predict(X_test)
    test_probs = estimator.predict_proba(X_test)

    print("\n Model Performance:")
    print("-------------------------------------------------------")
    f1 = get_f1_score(y_test, test_preds)
    print("F1 Macro Score: {}".format(f1))

    acc = get_accuracy(y_test, test_preds)
    print("Accuracy: {}".format(acc))

    roc_auc = get_roc_score(y_test, test_probs)
    print("Area under ROC curve: {}".format(roc_auc))

    precision, recall = get_precision_recall(y_test, test_preds)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("-------------------------------------------------------\n")

    conf_mat = plot_confusion_matrix_display(y_test, test_preds)
    roc_curve = plot_roc_curve(y_test, test_preds, test_probs)

    if args.show_plots:
        plt.show()

    if args.save_plots:
        # Get the date and time
        now = datetime.now()
        now_string = now.strftime("%d_%m_%y_%H_%M_%S")
        save_plot(conf_mat, "conf_mat_{}.png".format(now_string))
        save_plot(roc_curve, "roc_curve_{}.png".format(now_string))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adult Census Prediction using Random Forest Classifier.")
    parser.add_argument("--model_param", action="store", choices=["exhaustive", "load", "random_search"], default="load", 
                        help="Choose the parameters for the model. exhaustive performs GridSearchCV, load will load a parameter json and random_search performs a random grid search.")
    parser.add_argument("--random_search_iter", action="store", type=check_pos_int_arg, default=1000)
    parser.add_argument("--num_folds", action="store", type=check_pos_int_arg, default=5)
    parser.add_argument("--param_file", action="store", default="best_params.json")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--save_params", action="store_true")
    parser.add_argument("--oversample_type", action="store", default="SMOTE", choices=["SMOTE", "random", "none"])
    parser.add_argument("--sampling_strategy", action="store", type=check_pos_float_arg, default=0.5)
    parser.add_argument("--print_importances", action="store_true")


    args = parser.parse_args()

    if args.model_param == "load" and not args.param_file:
        print("\nNo path to parameter file specified with 'load' argument. Use --param_file <name of parameter file> to load the parameters.\n")
        raise FileNotFoundError("Parameter file not specified.")

    main(args)