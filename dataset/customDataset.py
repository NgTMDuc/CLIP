from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import pandas as pd
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, config, preprocess = None):
        super().__init__()
        self.data = data
        self.config = config
        self.image_folder = config['global']["image_folder"]

        self.image_paths = data.image_path

        self.captions = data.product_title
        self.labels = data.class_label
        self.brands = data.brand
        self.prompt = config["global"]['PROMPT']
        self.preprocess = preprocess
        # self.tokenize = tokenize

    def __getitem__(self, index):
        img_path = self.image_folder + self.image_paths[index]
        # print(img_path)
        img = Image.open(img_path)
        # img = transforms.ToPILImage(img)
        
        # print(img.shape)
        # img = 1
        if self.preprocess is not None:
            img = self.preprocess(img)
        
        img_caption = self.prompt.format(self.captions[index], self.brands[index], self.labels[index])
        

        return img, img_caption

    def __len__(self):
        return len(self.captions)


def dataLoader(dataset, configs):
    batch_size = configs['global']['BATCH_SIZE']
    return DataLoader(dataset, batch_size = batch_size)

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