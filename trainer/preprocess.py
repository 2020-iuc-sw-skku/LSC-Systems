import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
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

    # Change data type of failure_type (list -> str)
    data["failure_type"] = data["failure_type"].apply(
        lambda x: x[0][0] if len(x) > 0 else np.nan
    )

    # Drop rows with empty labels
    data = data[data["failure_type"].notnull()]

    # Change dtype of failure_type (object -> category)
    data["failure_type"] = data["failure_type"].astype("category")

    return data

def denoise_spatial(x, t, L):
    img = x.copy()

    for i in range(t, img.shape[0] - t):
        for j in range(t, img.shape[1] - t):
            if img[i][j] == 2:
                frac = img[i-t:i+t+1, j-t:j+t+1]
                N = frac[frac != 0].sum() - 1
                R = (frac[frac == 2].sum() - 1) / N

                if R < L:
                    img[i][j] = 1
    return img

def denoise_OPTICS(x):
    img = x.copy()
    img[img == 1] = 0
    coor = np.argwhere(img == 2)
    optic = OPTICS(min_samples=2, eps=3).fit(coor)

    coor = coor[:, 0], coor[:, 1]
    img[coor] = optic.labels_ + 1

    cluster_num = np.unique(img)
    cluster_num = np.delete(cluster_num, 0)

    cnt_array = np.array([])
    for num in cluster_num:
        cnt = len(img[img == num])
        cnt_array = np.append(cnt_array, cnt)

    temp = cnt_array.copy()
    temp.sort()
    max_ = temp[-3 if len(temp) >= 3 else -1]

    for num in cluster_num:
        if cnt_array[num - 1] < max_:
            img[img == num] = 0

    return img

if __name__ == "__main__":
    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])
    raw_data = pd.read_pickle(os.path.join(PATH_DATA, "LSWMD.pkl"))
    processed_data = preprocess(raw_data).reset_index()

    processed_data["wm_denoised_sp"] = processed_data["wafer_map"].apply(denoise_spatial, args=(1, 1/8))
    processed_data["wm_denoised_OP"] = processed_data["wafer_map"].apply(denoise_OPTICS)
    processed_data = processed_data[["wafer_map", "wm_denoised_sp", "wm_denoised_OP", "failure_type", "wafer_map_shape"]]
    processed_data.to_pickle(os.path.join(PATH_DATA, "processed_data.pkl"))
