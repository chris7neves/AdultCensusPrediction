import json
from datetime import datetime
import argparse

from paths import PARAMS_DIR, CONFIG_PATH

def save_params(params, filename=None):

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

    filepath = PARAMS_DIR / filename
    
    if not filepath.is_file():
        raise FileNotFoundError("Parameter file {} does not exist.".format(filepath.resolve()))

    with open(filepath, "r") as fin:
        params = json.load(fin)

    return params

def check_pos_int_arg(val):

    i = int(val)
    if i < 0:
        argparse.ArgumentTypeError("{} is not a positive integer value.".format(val))
    else:
        return i

def check_pos_float_arg(val):
    i = float(val)
    if i < 0:
        argparse.ArgumentTypeError("{} is not a positive float value.".format(val))
    else:
        return i