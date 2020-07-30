import os
import sys
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from PyQt5.QtWidgets import *
from PyQt5 import uic

from trainer import predict
from trainer.setup import PATH, CONFIG

# import threading
# import winsound

# 프레임 이름 뭐야?
# 그 AUC는 widget_auc, matrix는 widget_matrix
# UI파일 연결
# 아 내일 해야겠다
# 이거 파일 어디 올려둬
# 깃헙에 다 커밋 해놔도 돼
trainer_class = uic.loadUiType("trainerGUI.ui")[0]

# 화면을 띄우는데 사용되는 Class 선언
class TrainerGUI(QMainWindow, trainer_class):
    def __init__(self):
        super().__init__()

        self.path = {}

        self.data = None
        self.setupUi(self)

        self.text_data_path.setAcceptRichText(False)
        self.text_data_path.setReadOnly(True)

        self.text_model_path.setAcceptRichText(False)
        self.text_model_path.setReadOnly(True)

        self.button_load_data.clicked.connect(self.load_data_path)
        self.button_load_model.clicked.connect(self.load_data_model)
        self.button_start.clicked.connect(self.start_predict)

    def load_data_path(self):

        file_path = QFileDialog.getOpenFileNames(self, "Load Files", ".", "pkl(*.pkl)")[
            0
        ][0]

        self.text_data_path.clear()
        self.text_data_path.append(file_path)
        self.path["data"] = file_path

    def load_data_model(self):

        file_path = QFileDialog.getOpenFileNames(self, "Load Files", ".", "pkl(*.pkl)")[
            0
        ][0]

        self.text_model_path.clear()
        self.text_model_path.append(file_path)
        self.path["model"] = file_path

    # def show_matrix(self, matrix):

    def show_result(self, result):
        self.text_result.clear()
        accuracy_average = np.mean(result["accuracy"])
        self.text_result.append(
            "Accuracy: {} %".format(round((accuracy_average) * 100, 2))
        )
        precision_average = np.mean(result["precision"])
        self.text_result.append(
            "Precision: {} %".format(round((precision_average) * 100, 2))
        )
        recall_average = np.mean(result["recall"])
        self.text_result.append("Recall: {} %".format(round((recall_average) * 100, 2)))
        # self.text_result.append(str(round(precision_average) * 100, 2)))
        # self.text_result.append("%")
        # print(type(np.mean(result["precision"])))

    def start_predict(self):

        PATH_FEATURE = os.path.join(PATH, CONFIG["PATH"]["PATH_FEATURE"])

        data = pd.read_pickle(self.path["data"])
        data = data.reset_index()["failure_type"]
        model = joblib.load(self.path["model"])
        # 특성 추출한 것 joined.csv로 만들어서 feature 폴더안에 넣어야 함
        feature = pd.read_csv(os.path.join(PATH_FEATURE, "joined.csv")).values

        result = predict.evaluate(model, feature, data)
        self.show_result(result)


if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # TrainerGUI의 인스턴스 생성
    trainer = TrainerGUI()

    # 프로그램 화면을 보여주는 코드
    trainer.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
    # model = predict.select_model(PATH, 0)
