import json
import os
import sys
import pandas as pd
import numpy as np

import dill

from src.utils.transforms import preprocess
from src.utils.transforms import postprocess
from src.utils.transforms import transform_query_to_final

CONFIGS_FOLDER = "configs"
MAIN_CONFIG = "main.json"

with open(os.path.join(CONFIGS_FOLDER, MAIN_CONFIG), "r") as stream:
    config = json.load(stream)
    model_folder = config["model_folder"]
    model_name = config["model_name"]
    target = config["target"]

with open(os.path.join(model_folder, model_name + ".dill"), "rb") as stream:
    model = dill.load(stream)

def predict(data):
    data = preprocess(data)
    data[target[0]] = model.predict_labels(data)
    data = data.apply(transform_query_to_final, axis=1)
    data = data[["id", target[0], target[2]]]
    data = postprocess(data)
    return data

if __name__ == "__main__":
    file = sys.argv[1]
    data = pd.read_csv(file)
    data = predict(data)
    data.to_csv("results/result.csv", index=False)
