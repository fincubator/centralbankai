import os
import json

import numpy as np
import dill

CONFIGS_FOLDER = "configs"
FEATURES = "features.json"
MAPPER = "mapper.json"
MAIN = "main.json"

OBJECTS_FOLDER = "objects"
ENCODERS_FOLDER = "encoders"
ENCODER_POSTFIX = "_enc"


with open(os.path.join(CONFIGS_FOLDER, FEATURES), "r") as stream:
    features = json.load(stream)

with open(os.path.join(CONFIGS_FOLDER, MAPPER), "r") as stream:
    mapper = json.load(stream)
    inv_mapper = {v:k for k, v in mapper.items()}

with open(os.path.join(CONFIGS_FOLDER, MAIN), "r") as stream:
    config = json.load(stream)
    target = config["target"]


def preprocess(data):
    data = rename_columns(data)
    feats_cat = features["feats_cat"]
    for feat in feats_cat:
        data[feat] = preprocess_cat_feature(data, feat)
    return data

def postprocess(data):
    data = rename_columns(data, mapper=inv_mapper)
    return data

def rename_columns(data, mapper=mapper):
    data = data.rename(mapper, axis=1)
    return data

def preprocess_cat_feature(data, feat):
    with open(os.path.join(OBJECTS_FOLDER, ENCODERS_FOLDER, feat + ENCODER_POSTFIX + ".dill"), "rb") as stream:
        encoder = dill.load(stream)
    data[feat] = encoder.transform(data[feat].values.reshape(-1, 1))
    return data[feat]

def transform_query_to_final(data):
    with open(os.path.join(OBJECTS_FOLDER, ENCODERS_FOLDER, target[1] + ENCODER_POSTFIX + ".dill"), "rb") as stream:
        encoder = dill.load(stream)
    if data[target[0]] == 0:
        data[target[2]] = encoder.inverse_transform(
            np.array([data[target[1]]]).reshape(1, -1)
        )
        data[target[2]] = data[target[2]][0, 0]
    if data[target[0]] == 1:
        data[target[2]] = "Инцидент"
    if data[target[0]] == 2:
        data[target[2]] = "Запрос"
    return data