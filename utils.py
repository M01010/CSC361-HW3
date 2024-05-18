import json
import pickle
import pandas as pd


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_prediction(form, pipelines, encoder=None):
    model_name = form['model']

    x = pd.DataFrame(form, index=[0], columns=list(form.keys()))
    x.drop('model', axis=1, inplace=True)

    y = pipelines[model_name].predict(x.iloc[0:1])

    if not encoder:
        return model_name, y[0]

    y_label = encoder.inverse_transform(y)[0]

    return model_name, y_label
