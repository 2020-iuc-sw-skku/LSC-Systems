import os
import numpy as np
import pandas as pd
import math
import cv2
from setup import PATH, CONFIG


def similarity(y, x, type):
    # 웨이퍼맵의 평균 크기는 40x40 => 정규화시 40x40에 맞춰서 보간법 진행
    resize = lambda x: cv2.resize(x, dsize=(40, 40), interpolation=cv2.INTER_AREA)
    x = resize(x)
    y = resize(y)

    # similarity 계산
    mean = lambda x: x.sum() / x.size
    queried = x[:, :] - mean(x)
    candidate = y[:, :] - mean(y)
    score = ((queried * candidate).sum()) / math.sqrt(
        (queried ** 2).sum() * (candidate ** 2).sum()
    )

    x[x == 1] = 0
    y[y == 1] = 0

    queried = x[:, :] - mean(x)
    candidate = y[:, :] - mean(y)
    score_2 = ((queried * candidate).sum()) / math.sqrt(
        (queried ** 2).sum() * (candidate ** 2).sum()
    )

    global count
    count += 1

    # 유사도, 검색한 웨이퍼맵 결함패턴, 검색한 웨이퍼맵 인덱스 반환
    return (((score * 0.5) + (score_2 * 0.5)) * 100), type[count], count


def wm_index(x, y):
    # 입력받은 결함패턴의 대표 웨이퍼맵 찾기
    for i in range(len(y["wafer_map"])):
        if y["failure_type"].to_numpy()[i] == x:
            return i


if __name__ == "__main__":

    PATH_DATA = os.path.join(PATH, CONFIG["PATH"]["PATH_DATA"])
    PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])

    # denoising(spatial)처리후 진행, denoising후 반환시 .astype(np.float32) 적용
    data = pd.read_pickle(os.path.join(PATH_DATA, "sample_denoised.pkl"))

    # 검색하고싶은 결함패턴 입력
    defect_pattern = input(
        "1.Center  2.Donut  3.Edge-Loc  4.Edge-Ring  5.Loc  6.Near-full  7.Random  8.Scratch\n=> What defect pattern ? : "
    )
    count = -1

    # 유사도 구하기
    similarity_ranking = data["wafer_map"].apply(
        similarity,
        x=data["wafer_map"].to_numpy()[wm_index(defect_pattern, data)],
        type=data["failure_type"].to_numpy(),
    )

    # 정확순으로 정렬후 출력
    print(similarity_ranking.sort_values(ascending=False))
