import os
import numpy as np
import pandas as pd
import joblib
import trainerGUI
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
    cross_validate,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer

from trainer.setup import PATH, CONFIG


def select_model(path, model_name):
    model = (model_name, joblib.load(os.path.join(path, model_name)))
    return model


def scorer(y_true, y_pred):
    # print(classification_report(y_true, y_pred))
    return accuracy_score(y_true, y_pred)


def evaluate(model, X, y):
    cv = StratifiedShuffleSplit(n_splits=5, random_state=42)
    scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro", "roc_auc_ovr"]

    # 시각화
    ovr = OneVsRestClassifier(model, n_jobs=-1)
    score = cross_validate(
        model, X, y, scoring=scoring, cv=cv, n_jobs=-1,
    )  # score -> dic
    accuracy = cross_val_score(
        ovr, X, y, cv=cv, scoring=make_scorer(scorer)
    )  # accuracy -> array

    result_val = {}
    result_val["precision"] = score["test_precision_micro"]
    result_val["recall"] = score["test_recall_micro"]
    result_val["accuracy"] = accuracy

    return result_val

    ### 원래 코드 ###
    # for name, model in models:
    #     print(f"Running {name}...")
    #     ovr = OneVsRestClassifier(model, n_jobs=-1)
    #     score = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1,)
    #     accuracy = cross_val_score(ovr, X, y, cv=cv, scoring=make_scorer(scorer))
    #     for key, value in score.items():
    #         print(f"{key}: {np.mean(value):.3f}")
    #     print(f"Accuracy: {np.mean(accuracy):.3f}")
    #     print("============================")


if __name__ == "__main__":

    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])
    PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])
    PATH_PREDICT = os.path.join(PATH, CONFIG["PATH"]["PATH_PREDICT"])

    X = pd.read_csv(os.path.join(PATH_FEATURE, "joined.csv")).values
    y = pd.read_pickle(os.path.join(PATH_DATA, "sample.pkl"))
    y = y.reset_index()["failure_type"]

    model = select_model(PATH_PREDICT)
    evaluate(model, X, y)
