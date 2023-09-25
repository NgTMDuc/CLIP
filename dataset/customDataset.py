from torch import nn
from torch.utils.data import Dataset
import cv2
import json
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.data = data
        self.config = config
        self.image_folder = config['global']["image_folder"]

        self.image_paths = data.image_path

        self.captions = data.product_title
        self.labels = data.class_label
        self.brands = data.brand
        self.prompt = config["global"]['PROMPT']

    def __getitem__(self, index):
        img_path = self.image_folder + self.image_paths[index]
        # print(img_path)
        img = cv2.imread(img_path)
        # img = None
        img_caption = self.prompt.format(self.captions[index], self.brands[index], self.labels[index])
        
        return img, img_caption

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