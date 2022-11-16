# AdultCensusPrediction

Using a random forest classifier, predict whether an individual from the adult census dataset (https://archive.ics.uci.edu/ml/datasets/Adult) earns above or below 50K per annum.

**NOTE: PYTHON 3.9+ IS NEEDED TO RUN THE PROJECT**


## Directory Structure
---
```
│   .gitignore
│   main.py
│   parameter_grid.py
│   paths.py
│   README.md
│   requirements.txt
│   
├───data
│       adult.data
│       adult.names
│       adult.test
│       
├───notebooks
│       coen6321_asg2_notebook.ipynb
│       
├───params
│       best_params.json
│       
├───saved_plots
└───src
        gridsearch.py
        plotting_metrics.py
        preprocessing.py
        util.py
```

**main.py:** Entry point to the script, this is what is run.  
**parameter_grid.py:** The grid of parameters to look through when performing hyperparameter tuning.  
**paths.py:** File that allows easy access to all directories in the project.  
**src/gridsearch.py:** Functions used to perform hyperparameter gridsearch.  
**src/plotting_metrics.py:** Functions used to both plot results, and calculate metrics.
**src/preprocessing.py:** Functions used to prepare the data for the model.
**src/util.py**: Contains functions that help with various tasks.

**data/**: Directory that contains the data downloaded from https://archive.ics.uci.edu/ml/datasets/Adult  
**notebooks/**: Directory of jupyter notebooks used during the data exploration phase.  
**params/**: All saved parameter files are saved here. Parameter files that should be loaded into a model should also be placed here.

## Installation and Full Run

**NOTE: PYTHON 3.9+ IS NEEDED TO RUN THE PROJECT**

The following guide will instruct you how to install and run the project. The run that will be described will run the model with the best parameters that were found during the grid search.

KFold is performed while finding the optimal hyperparameters, and the model is tested on a completely unseen 10% of data to prevent data leakage.

1. Clone the repository to a local directory:

Navigate to the location that you would like to store this project and run the following command:

```
git clone https://github.com/chris7neves/AdultCensusPrediction.git
```
2. Make sure that the data/ directory exists. The repository contains the data needed for the project, but if it does not appear, then create a folder called "data" in the AdultCensusPrediction directory and add the following 3 files:

```
adult.data
adult.names
adult.test
```

Downloaded from this source: https://archive.ics.uci.edu/ml/datasets/Adult

3. CD into the project folder

```
cd AdultCensusPrediction
```

4. Create a virtual environment (preferably in the project root)

On Windows, this is dont by running:

```
python3 -m venv ./asg2env
```

Or depending on the python installation and path:

```
python -m venv asg2venv
```

5. Activate the venv

```
./asg2env/Scripts/activate
```

Or if using powershell:

```
./asg2env/Scripts/Activate.ps1
```

Or if on a MacOS/Unix environment:

```
source ./asg2env/bin/activate
```

6. Install the required dependecies from the included requirements.txt (make sure the asg2env is active)

```
python -m pip install -r requirements.txt
```

or

```
python3 -m pip install -r requirements.txt
```

7. Run the script using the following command in order to use the best parameters found for the problem:

```
python main.py --show_plots
```

This will load the best parameters into the RandomForestClassifier, prepare the data then test on the 10% test set.

## Additional Command Line Arguments

---

```
python main.py -h
```

Will yield a list of all the available arguments and options

---

```
python main.py --model_params
```

The --model_params argument lets you choose how the model parameters are found. The options are:

- load: Loads a parameter file for the RandomForest. Can be used with --param_file to specify the path to the parameter .json.
- exhaustive: Performs full kfold cross validation grid search over the parameter grid specified in parameter_grid.py. Will try every possible combination of parameters to find the best performing set.
- random_search: Performs a random kfold cross validation grid search. Will test a sample of parameter combinations to choose the best hyperparameters. Used in addition with --random_search_iter to define the number of parameter combinations to test out, and--num_folds to specify the number of folds to use in the kfold cross validation process.

---

```
python main.py --model_params load --param_file <filepath>
```

Used with --model_params load in order to specify the path to a parameter file. If this argument is left out, then params/best_params.json will be used instead. This file contains the parameters for our best performing model.

---

```
python main.py --save_plots
```

Saves the generated metrics plots to the AdultCenusPrediction/saved_plots/ directory. 

Plots that will be saved include the ROC curve, and the confusion matrix.

---

```
python main.py --show_plots
```

Show the plots after running the script. If not used, the plots will not be shown.

---

```
python main.py --save_params
```

Specify in order to save the parameters of the RandomForest mode to AdultCensusPrediction/params.

This should be used with the hyperparameter grid searches.

---

```
python main.py --oversample_type [SMOTE, random, none] --sampling_strategy <float>
```

Specifies the oversampling strategy for the training set.

The choices are:

- SMOTE: (Synthetic Minority Oversampling Technique) creates synthetic samples of the minority class through means of clustering. 
- random: randomly copies some samples of the minority class
- none: don't use an oversampling technique in model fitting

The --sampling_strategy argument is used to specify the target ratio for the minority class. For example, if 0.5 is used, then num_minorty_labels/num_majority_labels = 0.5.

---

```
python main.py --print_importances
```

Prints the importances for each feature as determined by an ExtraTreesClassifier, a tree ensemble model specifically used to determine robust feature importances that are not biased towards continuous features (like the importances generated by a random forest classifier).

---
