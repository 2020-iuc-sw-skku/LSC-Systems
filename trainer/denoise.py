import os
import sys
import pandas as pd
import numpy as np
import cv2
from skimage.measure import label
from scipy.ndimage import median_filter
from scipy.stats import mode
from setup import PATH, CONFIG


WAFER_SIZE = (100, 100)

def denoise(x):

    x = cv2.resize(x, dsize=WAFER_SIZE)
    label_x = label(median_filter(x==2, 2), connectivity=2, background=0).astype(np.uint8)

    if np.max(label_x) == 0:
        label_x[label_x == 0] = 1
    else:
        most_salient = mode(label_x[label_x > 0], axis=None)[0][0]
        label_x[label_x != most_salient] = 0

    return label_x


if __name__ == "__main__":

    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])

    data = pd.read_pickle(os.path.join(PATH_DATA, "sample.pkl"))
    data['wafer_map'] = data['wafer_map'].apply(denoise)

    data.to_pickle(os.path.join(PATH_DATA, "denoised_sample.pkl"))
