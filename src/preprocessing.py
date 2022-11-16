from paths import ROOT_DIR, DATA_DIR, SRC_DIR
from src.plotting_metrics import get_feature_importance

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler


def clean_adult(df):
    """
    Used to clean both the adult.data and the adult.test files.
    
    Columns must be renamed on import.
    """

    adf = df.copy()
    
    # Replace ? with np.nan
    adf.replace({"\?":np.nan}, regex=True, inplace=True)
    
    # Drop all rows with nans
    before = len(adf)
    adf.dropna(axis=0, inplace=True)
    after = len(adf)
    print("{} rows dropped due to missing data.".format(before - after))
    
    # Make sure all datatypes are properly set
    adf["age"] = adf["age"].astype(np.int64)
    adf["fnlwgt"] = adf["fnlwgt"].astype(np.int64)
    adf["education-num"] = adf["education-num"].astype(np.int64)
    adf["capital-gain"] = adf["capital-gain"].astype(np.int64)
    adf["capital-loss"] = adf["capital-loss"].astype(np.int64)
    adf["hours-per-week"] = adf["hours-per-week"].astype(np.int64)
    
    return adf


def preprocess(df, one_hot=[], to_drop=['education'], scale=[]):
    """
    Performs the relevant transforms on the data to prepare it for use with a model.
    """
    
    adf = df.copy()
    
    # Convert the labels for salary to 1 and 0
    adf["salary"].replace({"<=50K":0, ">50K":1, "<=50K.":0, ">50K.":1}, regex=True, inplace=True)
    
    # Drop the education column since it is already accounted for in education-num
    if to_drop:
        adf.drop(to_drop, axis=1, inplace=True)

    # One hot encode all columns in one_hot
    for col in one_hot:
        if col in adf.columns:
            temp = pd.get_dummies(adf[col])
            adf = adf.drop(col, axis=1)
            adf = adf.join(temp)
    
    # Scale the desired columns (center values around the column mean and std)
    if scale:
        scaler = StandardScaler()
        adf[scale] = scaler.fit_transform(adf[scale])
    
    return adf


def oversample_classes(train_data, train_labels, strategy='SMOTE', ratio=0.5):
    """
    Performs oversampling to help the classifier learn more about the minority class.
    Either the SMOTE or random oversampling techniques can be used.

    NOTE: Oversampling MUST be used on the training dataset AFTER splitting the data. 
    If not, the results will be overly optimistic.

    ratio: the goal proportion of the minority class compared to the majority class.
    """

    print("\nLabel value counts before oversampling:")
    print(train_labels.value_counts())

    if strategy == "SMOTE":
        oversampler = SMOTE(sampling_strategy=ratio)
    elif strategy == "random":
        oversampler = RandomOverSampler(sampling_strategy=ratio)
    else:
        print("{} is an invalid oversampling strategy. No oversampling will be performed.")
        return train_data, train_labels

    over_data, over_labels = oversampler.fit_resample(train_data, train_labels)

    print("Label value counts after oversampling:")
    print(over_labels.value_counts())
    print()

    return over_data, over_labels
