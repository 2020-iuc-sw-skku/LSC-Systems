import os
import sys
import pandas as pd
import numpy as np
from math import log10
from skimage.transform import radon as radon_transform
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d
from scipy.stats import mode
from setup import PATH, CONFIG


def extract_density(x, num=6):
    """
    Extract density based feature

    Args:
        x (np.array): wafer map
        num (int): number of regions to divide wafer map. Defaults to 6.

    Returns:
        pd.DataFrame: density feature of each region
    """

    # 필요한 자릿수를 구한다 (특성 이름을 0으로 패딩하기 위함)
    max_digit = int(log10((num - 2) ** 2 + 4)) + 1
    # 해당 영역에서의 밀도를 구한다
    calculate_density = lambda x: 100 * (
        np.sum(x == 2) / np.sum(np.logical_or(x == 1, x == 2))
    )
    # 특성명을 생성한다. 남는 자릿수는 0으로 패딩한다
    # (특성명을 사전순으로 배열할 때 순서가 깨지는 것을 방지하기 위함)
    feature_name = lambda x: f"density_{str(x).zfill(max_digit)}"

    # 웨이퍼 맵을 분할하는 경계선을 구한다
    row, col = x.shape
    interval_row = np.linspace(0, row, num=num, endpoint=False, dtype=np.int32)
    interval_col = np.linspace(0, col, num=num, endpoint=False, dtype=np.int32)

    density = {}
    count = 1

    # 중간 부분의 밀도
    for i in range(1, num - 1):
        for j in range(1, num - 1):
            region = x[
                interval_row[i] : interval_row[i + 1],
                interval_col[j] : interval_col[j + 1],
            ]
            density[feature_name(count)] = calculate_density(region)
            count += 1

    # 바깥 부분의 밀도
    density[feature_name(count)] = calculate_density(
        x[interval_row[0] : interval_row[1], :]
    )
    count += 1
    density[feature_name(count)] = calculate_density(x[:, interval_col[num - 1] :])
    count += 1
    density[feature_name(count)] = calculate_density(x[interval_row[num - 1] :, :])
    count += 1
    density[feature_name(count)] = calculate_density(
        x[:, interval_col[0] : interval_col[1]]
    )

    return pd.Series(density)


def extract_radon(x):
    """
    Extract radon based feature

    Args:
        x (np.array): wafer map

    Returns:
        pd.DataFrame: radon feature of each region
    """

    x = x.copy()
    x[x == 1] = 0

    calculate_density = lambda x: 100 * (
        np.sum(x == 2) / np.sum(np.logical_or(x == 1, x == 2))
    )
    feature_name = lambda s, x: f"{s}_{str(x).zfill(2)}"

    theta = np.linspace(0, 180, max(x.shape), endpoint=False)
    sinogram = radon_transform(x, theta=theta, circle=False, preserve_range=False)
    radon = {}

    # mean of sinogram
    mean_y = np.mean(sinogram, axis=1)
    mean_x = np.linspace(1, mean_y.size, mean_y.size)

    mean_interpolate = interp1d(mean_x, mean_y, kind="cubic")
    new_mean_x = np.linspace(1, mean_y.size, 20)
    new_mean_y = mean_interpolate(new_mean_x) / 100

    for i in range(20):
        radon[feature_name("mean", i + 1)] = new_mean_y[i]

    # std of sinogram
    std_y = np.std(sinogram, axis=1)
    std_x = np.linspace(1, std_y.size, std_y.size)

    std_interpolate = interp1d(std_x, std_y, kind="cubic")
    new_std_x = np.linspace(1, std_y.size, 20)
    new_std_y = std_interpolate(new_std_x) / 100

    for i in range(20):
        radon[feature_name("std", i + 1)] = new_std_y[i]

    return pd.Series(radon)


def extract_geometry(x):
    """
    Extract geometry based feature

    Args:
        x (np.array): wafer map

    Returns:
        pd.DataFrame: geometry feature of each region
    """

    norm_area = np.prod(x.shape)
    norm_perimeter = np.linalg.norm(x.shape, ord=2)

    label_x = label(x, connectivity=1, background=0)

    if np.max(label_x) == 0:
        label_x[label_x == 0] = 1
    else:
        most_salient = mode(label_x[label_x > 0], axis=None)[0][0]
        label_x[label_x != most_salient] = 0

    prop_x = regionprops(label_x)[0]
    geometry = {}

    geometry["prop_area"] = prop_x.area / norm_area
    geometry["prop_perimeter"] = prop_x.perimeter / norm_perimeter
    geometry["prop_major_axis"] = prop_x.major_axis_length / norm_perimeter
    geometry["prop_minor_axis"] = prop_x.minor_axis_length / norm_perimeter
    geometry["prop_eccentricity  "] = prop_x.eccentricity
    geometry["prop_solidity"] = prop_x.solidity

    return pd.Series(geometry)


def extract_polar(x):
    pass


if __name__ == "__main__":
    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])
    PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])

    data = pd.read_pickle(os.path.join(PATH_DATA, "processed_data.pkl"))

    density_based_4x4 = data["wafer_map"].apply(extract_density)
    density_based_5x5 = data["wafer_map"].apply(extract_density, args=(7,))
    radon_based = data["wafer_map"].apply(extract_radon)
    geometry_based = data["wafer_map"].apply(extract_geometry)
    # polar_based = data['wafer_map'].apply(extract_polar)

    density_based_4x4.to_pickle(os.path.join(PATH_FEATURE, "density_based_4x4.pkl"))
    density_based_5x5.to_pickle(os.path.join(PATH_FEATURE, "density_based_5x5.pkl"))
    radon_based.to_pickle(os.path.join(PATH_FEATURE, "radon_based.pkl"))
    geometry_based.to_pickle(os.path.join(PATH_FEATURE, "geometry_based.pkl"))
    # polar_based.to_pickle(os.path.join(PATH_FEATURE, 'polar_based.pkl'))
