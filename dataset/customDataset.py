from torch import nn
from torch.utils.data import Dataset
import cv2
import json
import pandas as pd
# from ..utils.load_json_file import open_json_file

class CustomDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.data = data
        self.config = config
        self.image_folder = config.image_folder

        self.image_paths = data.image_path

        self.captions = data.product_title
        self.labels = data.class_label
        self.brand = data.brand

    def __getitem__(self, index):
        img_path = self.image_folder + self.image_paths[index]
        img = cv2.imread(img_path)
        img_caption = self.captions[index] + ""
        return img, self.captions[index], self.labels[index]

    def __len__(self):
        return len(self.captions)

if __name__ == "__main__":
    def open_json_file(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    path = "./test_data.json"
    df = open_json_file(path)
    print(df.product_title[2])