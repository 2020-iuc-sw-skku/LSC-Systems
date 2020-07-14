import os
import pandas as pd
import joblib
from setup import PATH, CONFIG


def predict_label(path, features):
    """
    Predict label with dumped scikit-learn models

    Args:
        path (str): path where dumped models are stored
        features (pd.DataFrame): joined features

    Returns:
        pd.DataFrame: predicted labels
    """

    models = os.listdir(path)
    models = [i for i in models if ".pkl" in i]

    predicted_labels = []
    for mod in models:
        # load dumped model
        model = joblib.load(os.path.join(path, mod))
        predict = model.predict(features)
        predicted_labels.append(predict)

    predicted_labels = pd.DataFrame(predicted_labels).T
    return predicted_labels


if __name__ == "__main__":
    PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])
    PATH_PREDICT = os.path.join(PATH, CONFIG["PATH"]["PATH_PREDICT"])

    joined_features = pd.read_csv(os.path.join(PATH_FEATURE, "joined.csv"))
    predicted_labels = predict_label(PATH_PREDICT, joined_features)
    predicted_labels.to_csv(os.path.join(PATH_PREDICT, "predicted.csv"))
