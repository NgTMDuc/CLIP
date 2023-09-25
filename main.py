import os
import clip
import torch
import numpy as np
from tqdm import tqdm

from utils.basic_args import obtain_args
from utils.load_config import load_config
from utils.load_json_file import open_json_file
from dataset.customDataset import CustomDataset

def train(args):
    config = args.config_paths
    configs = load_config(config)
    # print(configs)
    # print(configs['global'])
    image_path = configs['global']['image_path']
    train_path_json = configs['global']['train_path']
    valid_path_json = configs['global']['valid_path']
    test_path_json = configs['global']['test_path']



if __name__ == "__main__":
    args = obtain_args()
    train(args)