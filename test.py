from dataset import customDataset
from utils.basic_args import obtain_args
from utils.load_config import load_config
from utils.load_json_file import open_json_file
from dataset.customDataset import CustomDataset, dataLoader
import torch
from clip import clip
args = obtain_args()
# print(args)
configs = load_config(args.config_paths)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) 
# tokenize = clip.tokenize()
# print(configs)

train_path = configs['global']['train_path']
test_path = configs['global']['test_path']
valid_path = configs['global']['valid_path']

data = open_json_file(test_path)

customData = CustomDataset(data, configs, preprocess)
dataCheck = dataLoader(customData, configs)
# print(dataCheck[0])
# for img, cap in dataCheck:
#     print(1)
for batch in dataCheck:
    # print(batch)
    image, caption = batch
    print("-----------")
    print("Image: ")
    print("-----------")
    print(image.shape)
    caption = clip.tokenize(caption)
    print("Caption")
    print(caption.shape)
    print("-----------")
    break