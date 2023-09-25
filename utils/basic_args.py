import os
import sys
import time
import random
import argparse
# from load_config import load_config

def obtain_args():
    parser = argparse.ArgumentParser(
        description = "Fine tune CLIP model",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config_paths", type = str, default = "/kaggle/working/CLIP/config/clip.yaml")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = obtain_args()
    # print(args.config_paths)
    # config = load_config(args.config_paths)