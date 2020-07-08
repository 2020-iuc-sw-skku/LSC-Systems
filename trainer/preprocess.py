import os
import sys
import pandas as pd
import numpy as np
from setup import PATH, CONFIG


def preprocess(raw_data):
    """
    Preprocess raw data

    Args:
        raw_data (pd.DataFrame): raw data

    Returns:
        pd.DataFrame: preprocessesd data
    """

    # Drop unnecessary features
    data = raw_data.drop(["dieSize", "lotName", "waferIndex", "trianTestLabel"], axis=1)

    # Rename feature names (camelCase -> snake_case)
    data.rename(
        columns={"waferMap": "wafer_map", "failureType": "failure_type"}, inplace=True
    )

    # Add new feature (wafer_map_shape: shape of wafer_map)
    data["wafer_map_shape"] = data["wafer_map"].apply(lambda x: x.shape)

    # Change data type of failure_type (list -> str)
    data["failure_type"] = data["failure_type"].apply(
        lambda x: x[0][0] if len(x) > 0 else np.nan
    )

    # Drop rows with empty labels
    data = data[data["failure_type"].notnull()]

    # Change dtype of failure_type (object -> category)
    data["failure_type"] = data["failure_type"].astype("category")

    return data


if __name__ == "__main__":
    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])
    raw_data = pd.read_pickle(os.path.join(PATH_DATA, "LSWMD.pkl"))
    processed_data = preprocess(raw_data).reset_index()
    processed_data.to_pickle(os.path.join(PATH_DATA, "processed_data.pkl"))
