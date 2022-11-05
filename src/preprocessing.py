from paths import ROOT_DIR, DATA_DIR, SRC_DIR

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

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


def preprocess(df, one_hot=[], drop_education=True, scale=[]):
    """
    NOTE: ONE HOT ENCODING SHOULD NOT BE USED WITH RANDOM FORESTS
    https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769
    """
    
    adf = df.copy()
    
    # Convert the labels for salary to 1 and 0
    adf["salary"].replace({"<=50K":0, ">50K":1}, regex=True, inplace=True)
    
    # One hot encode all columns in one_hot
    for col in one_hot:
        temp = pd.get_dummies(adf[col])
        adf = adf.drop(col, axis=1)
        adf = adf.join(temp)
        
    # Drop the education column since it is already accounted for in education-num
    if drop_education:
        adf.drop("education", axis=1, inplace=True)
    
    # Scale the desired columns (center values around the column mean and std)
    if scale:
        scaler = StandardScaler()
        adf[scale] = scaler.fit_transform(adf[scale])
    
    return adf