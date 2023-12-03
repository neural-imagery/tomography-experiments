import datetime
import os
import numpy as np


class Saver:
    def __init__(self, experiment_name, params):
        params_str = "_".join(
            [f"{k}={format(v, '.0e') if v > 1e3 else v}" for k, v in params.items()])
        params_folder = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{params_str}"
        self.path = f"data/{experiment_name}/{params_folder}"

        # create the data/experiment_name folder if it doesn't exist
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(f"data/{experiment_name}"):
            os.mkdir(f"data/{experiment_name}")
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def save(self, data, name=None):
        # if name is none get the name of the variable
        if name is None:
            name = var_name(data)

        np.save(f"{self.path}/{name}.npy", data)


def var_name(var):
    for k, v in globals().items():
        if v is var:
            return k
    return None
