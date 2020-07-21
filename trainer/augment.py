import os
import warnings
import pandas as pd
import numpy as np
from torchvision import transforms

warnings.filterwarnings("ignore")


class Augmenter:
    def __init__(self, data):

        self.data = data.copy()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation((0, 360)),
            ]
        )

    def get(self, amount):

        df = pd.DataFrame()
        for ft, cnt in amount.items():
            sample = (
                self.data[self.data["failure_type"] == ft]
                .sample(cnt, replace=True)
                .reset_index()
            )
            sample.loc[:, "wafer_map"] = sample["wafer_map"].apply(
                lambda x: np.array(self.transform(x))
            )

            df = pd.concat([df, sample], axis=0)

        return df


if __name__ == "__main__":
    data = pd.read_pickle("data/sample.pkl")
    aug = Augmenter(data)

    amount = {
        "Center": 600,
        "Donut": 600,
        "Edge-Loc": 600,
        "Edge-Ring": 600,
        "Loc": 600,
        "Near-full": 600,
        "Random": 600,
        "Scratch": 600,
        "none": 600,
    }
    augmented = aug.get(amount)
    print(augmented.shape)
