import os
import sys
import pandas as pd
import numpy as np
import cv2
from math import log10
from skimage.transform import radon as radon_transform
from skimage.measure import label, regionprops
from skimage.feature import greycomatrix, greycoprops
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
    calculate_density = lambda x: (
        np.sum(x == 2) / np.size(x)
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
    x[x != 2] = 0

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

    label_x = label(x == 2, connectivity=1, background=0)

    if np.max(label_x) == 0:
        label_x[label_x == 0] = 1
    else:
        most_salient = mode(label_x[label_x > 0], axis=None)[0][0]
        label_x[label_x != most_salient] = 0
        label_x[label_x == most_salient] = 1

    prop_x = regionprops(label_x)[0]
    geometry = {}

    geometry["prop_area"] = prop_x.area / norm_area
    geometry["prop_perimeter"] = prop_x.perimeter / norm_perimeter
    geometry["prop_major_axis"] = prop_x.major_axis_length / norm_perimeter
    geometry["prop_minor_axis"] = prop_x.minor_axis_length / norm_perimeter
    geometry["prop_eccentricity"] = prop_x.eccentricity
    geometry["prop_solidity"] = prop_x.solidity

    return pd.Series(geometry)


def extract_distance(x):

    feature_name = lambda s, x: f"{s}_{str(x).zfill(2)}"

    coor = np.argwhere(x == 2) - (np.array(x.shape) // 2)
    radius = np.linalg.norm(coor, ord=2, axis=1)

    dist = {}

    # polar accumulate
    dist_y, _ = np.histogram(radius)
    dist_x = np.linspace(1, dist_y.size, dist_y.size)

    dist_interpolate = interp1d(dist_x, dist_y, kind="cubic")
    new_dist_x = np.linspace(1, dist_y.size, 20)
    new_dist_y = dist_interpolate(new_dist_x) / np.linspace(
        1, new_dist_x.size, new_dist_x.size
    )

    for i in range(20):
        dist[feature_name("dist_value", i + 1)] = new_dist_y[i]

    dist[feature_name("dist", "mean")] = np.mean(new_dist_y)
    dist[feature_name("dist", "std")] = np.std(new_dist_y)
    dist[feature_name("dist", "max")] = np.max(new_dist_y)
    dist[feature_name("dist", "min")] = np.min(new_dist_y)
    dist[feature_name("dist", "argmax")] = np.argmax(new_dist_y)
    dist[feature_name("dist", "argmin")] = np.argmin(new_dist_y)

    return pd.Series(dist)


def extract_texture(x):
    feature_name = lambda s, x, y: f"{s}_{str(x).zfill(2)}_{str(y)}"
    text = {}
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']

    angles = [0, np.pi/4, np.pi/2, np.pi*3/4] #4 angles
    glcm = greycomatrix(x , [1], angles) 
    
    for f in props:
        for i in range(4):
            text[feature_name('text', f, i)] = greycoprops(glcm, f)[0][i]
    
    return pd.Series(text)


def get_polar_mask(N_a, N_c, size=32):
    """
    Make polar mask

    Args:
        N_a (int): Sampling parameter of radius for polar masks
        N_c (int): Sampling parameter of angle for polar masks
        size (int): size of wafer map image
    
    Returns:
        list: polar masks 
    """

    A = [[2 ** N_a - 1], [(1 << (N_a // 2)) - 1], []]
    C = [[] for i in range(N_c)]

    bit_mask = 2 ** N_a - 1
    #a_1,*
    for i in range(1, N_a):
        if i == N_a // 2:
            continue
        A[0].append((A[0][0] << i) & bit_mask)
    
    #a_2,*
    for i in range(1, N_a // 2):
        A[1].append((A[1][0] ^ (A[1][0] << i)) & bit_mask)

    A = [list(map(lambda x: list(map(int, bin(x)[2:].zfill(N_a))), arr)) for arr in A]
    
    #a_3,*
    A[2] = [list(map(lambda x: 2 * x - 1, arr)) for arr in A[1]]

    #c
    for i in range(N_c):
        C[i].append(2 ** (i + 1) - 1)
        for j in range(1, N_c - i):
            C[i].append(C[i][0] << j)

    C = [list(map(lambda x: list(map(int, bin(x)[2:].zfill(N_c))), arr)) for arr in C]
    
    #flatten
    A = [a for arr in A for a in arr]
    C = [c for arr in C for c in arr]

    weight_matrix = np.array([np.matmul(np.array([a]).T, np.array([c])) for a in A for c in C])

    #basic mask
    R = size // 2
    rad = np.linspace(0, R, num=N_c, endpoint=False)
    angle = np.linspace(0, 2*np.pi, num=N_a, endpoint=False)
    dist = lambda y, x: np.sqrt((x - R) ** 2 + (y - R) ** 2)

    mask = np.zeros((size, size))    
    for i in range(size):
        for j in range(size):
            d = dist(i, j)
            if d > R:
                continue
            theta = np.arctan2(R - i, j - R)
            if theta < 0:
                theta += 2 * np.pi
            mask[i][j] = np.argmax(angle[angle <= theta]) * N_c + np.argmax(rad[rad <= d]) + 1

    #weighted mask   
    masks = []
    for weight in weight_matrix:
        weighted_mask = mask.copy()
        for i in range(N_a):
            for j in range(N_c):
                weighted_mask[weighted_mask == (N_c * i + j + 1)] = weight[i][j]
        masks.append(weighted_mask)
    
    return masks


def get_line_mask(N_l, size=32):
    """
    Make line mask

    Args:
        N_l (int): Sampling parameter of radius for line masks
        size (int): size of wafer map image
    
    Returns:
        list: line masks 
    """
    
    R = size // 2
    dist = lambda y, x: np.sqrt((x - R) ** 2 + (y - R) ** 2)
    rad = np.linspace(0, R, num=N_l, endpoint=False)

    #basic mask
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            d = dist(i, j)
            if d > R:
                continue

            theta = np.arctan2(R - i, j - R)
            if -np.pi / 2 <= theta <= np.pi / 2:
                mask[i][j] = np.argmax(rad[rad <= (j - R)]) + 1

    #weighted mask
    masks = []
    for i in range(1, N_l + 1):
        weighted_mask = mask.copy()
        weighted_mask[weighted_mask != i] = 0
        weighted_mask[weighted_mask == i] = 1
        masks.append(weighted_mask)

    return masks


def get_arc_mask(N_r, N_o, R_l, R_h, R_c, size=32):
    """
    Make arc mask

    Args:
        N_r (int): Sampling parameter of annulus radius for arc masks
        N_o (int): Sampling parameter of annulus center location for arc masks
        R_l (float): Lower limit of annulus radius for arc masks
        R_h (float): Higher limit of annulus radius for arc masks
        R_c (float): Higher limit of annulus center location for arc masks
        size (int): size of wafer map image
    
    Returns:
        list: arc masks 
    """
       
    R_l, R_h, R_c = [(size / 2) * i for i in (R_l, R_h, R_c)]

    dist = lambda y, x, c: np.sqrt((x - c - size / 2) ** 2 + (y - size / 2) ** 2)
    
    rad = np.linspace(R_l, R_h, num=N_r, endpoint=False)
    center = np.linspace(R_c, 0, num=N_o, endpoint=False)

    masks = []
    for c in center:
        #basic mask
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                d = dist(i, j, c)
                if d < R_l or d > R_h:
                    continue
                
                mask[i][j] = np.argmax(rad[rad <= d]) + 1

        #weighted mask
        for i in range(1, N_r + 1):
            weighted_mask = mask.copy()
            weighted_mask[weighted_mask != i] = 0
            weighted_mask[weighted_mask == i] = 1
            masks.append(weighted_mask)

    return masks


def extract_mask(x, mask, N_t, size=32):
    """
    Extract feature using masks

    Args:
        x (np.array): wafer map
        mask (dict): masks
        N_t (int): Number of rotated copies of the master mask
        size (int): size of wafer map image
    
    Returns:
        pd.DataFrame: extracted features 
    """
    #resize img
    img = x.copy()
    if img.shape != (size, size):
        img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
    
    #preprocess before masking
    img[img != 2] = 0
    img[img == 2] = 1

    feature = {}
    for name, masks in mask.items():
        max_digit = len(str(len(masks)))
        feature_name = lambda x: f"{name}_{str(x).zfill(max_digit)}"

        for i in range(len(masks)):
            extracted = []

            #rotate master mask
            for angle in np.linspace(0, 360, num=N_t, endpoint=False):
                rotation_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
                rotated_mask = cv2.warpAffine(masks[i], rotation_matrix, (size, size))

                masked_img = img * rotated_mask
                extracted.append(masked_img.sum(dtype=np.int32))

            #save max value 
            feature[feature_name(i + 1)] = max(extracted)

    return pd.Series(feature)


if __name__ == "__main__":

    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])
    PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])

    data = pd.read_pickle(os.path.join(PATH_DATA, "denoised_sample.pkl"))

    type = {"1": "wafer_map", "2": "wm_denoised_sp", "3": "wm_denoised_OP"}
    print("1. Noised 2. Denoised(Spatial) 3. Denoised(OPTICS)\nChoose type of image before extraction: ")
    selected_img = type[input()]

    density_based = data[selected_img].apply(extract_density, args=(6,))
    radon_based = data[selected_img].apply(extract_radon)
    geometry_based = data[selected_img].apply(extract_geometry)
    distance_based = data[selected_img].apply(extract_distance)
    texture_based = data[selected_img].apply(extract_texture)
    mask_based = data[selected_img].apply(extract_mask, args=({
        "polar_mask": get_polar_mask(4, 5, 32),
        "line_mask": get_line_mask(7, 32),
        "arc_mask": get_arc_mask(6, 12, 0.5, 1.0, 1.2, 32)
    }, 16, 32))
    
    density_based.to_pickle(os.path.join(PATH_FEATURE, "sample_density_based_4x4.pkl"))
    radon_based.to_pickle(os.path.join(PATH_FEATURE, "sample_radon_based.pkl"))
    geometry_based.to_pickle(os.path.join(PATH_FEATURE, "sample_geometry_based.pkl"))
    distance_based.to_pickle(os.path.join(PATH_FEATURE, "sample_distance_based.pkl"))
    texture_based.to_pickle(os.path.join(PATH_FEATURE, "sample_texture_based.pkl"))
    mask_based.to_pickle(os.path.join(PATH_FEATURE, "sample_mask_based.pkl"))
