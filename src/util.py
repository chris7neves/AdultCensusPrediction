import json
from datetime import datetime
import argparse
from pathlib import Path

from paths import PARAMS_DIR, CONFIG_PATH, PLOTS_PATH

import matplotlib.pyplot as plt


def save_params(params, filename=None):
    """
    Save a model's parameter json to file with a filename specified in 'filename'.
    Filename should not be a path. The file's path will be the root / params directory.
    """

    if type(params) is not dict:
        raise TypeError("The params to be saved are not in a dict. They are: {}".format(type(params)))
    else:

        if not filename:
            now = datetime.now()
            now_string = now.strftime("%d_%m_%y_%H_%M_%S")
            filename = ("{}_params.json".format(now_string))

        file_path = PARAMS_DIR / filename 
        with open(file_path, "w") as fout:
            json.dump(params, fout)

        print("Parameters were saved to {}".format(file_path))


def load_params(filename):
    """
    Handles finding the parameters file json, and loading it into a python dictionary to be used with
    scikit models.
    """
    filepath = PARAMS_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError("Parameter file {} does not exist.".format(filepath.resolve()))

    with open(filepath, "r") as fin:
        params = json.load(fin)

    return params


def check_pos_int_arg(val):
    """
    Used to validate and convert command line user input to int.
    """
    i = int(val)
    if i < 0:
        argparse.ArgumentTypeError("{} is not a positive integer value.".format(val))
    else:
        return i


def check_pos_float_arg(val):
    """
    Used to validate and convert command line user input to float.
    """
    i = float(val)
    if i < 0:
        argparse.ArgumentTypeError("{} is not a positive float value.".format(val))
    else:
        return i


def save_plot(p, filename):
    """
    Helper function used to save a plot as a png to the plot save directory.
    """

    # Make sure plot dir exists.
    p_path = Path(PLOTS_PATH)
    if not p_path.is_dir():
        p_path.mkdir(parents=False, exist_ok=False)
    
    # Save plot
    filepath = PLOTS_PATH / filename

    try:
        # Case where p is a matplotlib figure
        p.savefig(filepath)
    except AttributeError:
        pass
    try:
        # Case where p is a Display object from scikit
        p.figure_.savefig(filepath)
    except Exception as err:
        raise