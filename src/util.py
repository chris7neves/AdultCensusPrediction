import json
from datetime import datetime

from paths import PARAMS_DIR

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