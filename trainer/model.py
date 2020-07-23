import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from setup import *


def dump_model(model, name):
    PATH_PREDICT = os.path.join(PATH, CONFIG["PATH"]["PATH_PREDICT"])
    dump(model, os.path.join(PATH_PREDICT, name + ".pkl"))


def create_lr():
    model_lr = LogisticRegression(
        multi_class="ovr", penalty="l2", solver="liblinear", random_state=42, n_jobs=1,
    )
    dump_model(model_lr, "model_lr")
    return model_lr


def create_rf():
    model_rf = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    dump_model(model_rf, "model_rf")
    return model_rf


def create_gbm():
    model_gbm = GradientBoostingClassifier(
        n_estimators=100, loss="deviance", criterion="friedman_mse", random_state=42
    )
    dump_model(model_gbm, "model_gbm")
    return model_gbm


def create_ann():
    model_ann = MLPClassifier(
        activation="relu", hidden_layer_sizes=(100,), solver="adam"
    )
    dump_model(model_ann, "model_ann")
    return model_ann


def create_sve():
    models = zip(('LR', 'RF', 'GBM', 'ANN'), (create_lr(), create_rf(), create_gbm(), create_ann()))
    model_sve = VotingClassifier(list(models), voting="soft", n_jobs=-1)
    dump_model(model_sve, "model_sve")


if __name__ == "__main__":
    create_sve()
