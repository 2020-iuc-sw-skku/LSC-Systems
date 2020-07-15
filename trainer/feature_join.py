import os
import sys
import pandas as pd
import numpy as np
from setup import PATH, CONFIG


def select_feature(path):
    """
    Join selected features

    Args:
        path (str): Directory where features are saved as .pkl.

    Returns:
        list: list of pickle files of selected features
    """

    features = os.listdir(path)
    features = [i for i in features if ".pkl" in i]

    if not features:
        print("No features found.")
        return []

    print(f"{len(features)} features found. Select features to join.")
    for i, fea in enumerate(features):
        print(f"{i+1}. {fea.split('.')[0]}")

    selection = input(": ")
    selection = map(int, selection.split(" "))

    selected_features = [features[i - 1] for i in selection]
    return selected_features


if __name__ == "__main__":
    PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])
    selected_features = select_feature(PATH_FEATURE)
    features = [
        pd.read_pickle(os.path.join(PATH_FEATURE, fea)) for fea in selected_features
    ]
    for fea in features:
        fea.reset_index(inplace=True)
        fea.fillna(0, inplace=True)

    joined_features = pd.concat(features, axis=1)
    joined_features.to_csv(os.path.join(PATH_FEATURE, "joined.csv"))
