import os
import pickle


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def get_models(path):
    return os.listdir(path)
