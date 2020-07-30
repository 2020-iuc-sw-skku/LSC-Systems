import os 
import pandas as pd 
import numpy as np
import cv2
from setup import PATH, CONFIG

if __name__ == "__main__":

    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])

    data = pd.read_pickle(os.path.join(PATH_DATA, "denoised_sample.pkl")).reset_index(drop=True)
    o_data = pd.read_pickle(os.path.join(PATH_DATA, "sample.pkl")).reset_index(drop=True)
    data.drop(columns=['index', 'wafer_map_shape'], inplace=True)
    o_data.drop(columns=['index', 'wafer_map_shape'], inplace=True)

    for i in range(data.shape[0]):
        cv2.imshow('original', cv2.resize(o_data.loc[i, 'wafer_map']*127, (500, 500)))
        cv2.imshow('denoised', cv2.resize(data.loc[i, 'wafer_map']*127, (500, 500)))

        cv2.waitKey(0)