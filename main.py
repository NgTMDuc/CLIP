import os
import clip
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch import optim
from utils.basic_args import obtain_args
from utils.load_config import load_config
from utils.load_json_file import open_json_file
from dataset.customDataset import CustomDataset, dataLoader
from clip.model import convert_weights
from clip.clip import tokenize
from utils.log_utils import setup_logger
from datetime import datetime
from utils.processText import process_text

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def train(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    
    # Configuration
    config = args.config_paths
    configs = load_config(config)
    EPOCH = configs['global']['EPOCHS']
    logger = setup_logger("CLIP", configs["global"]['log_path'], "CLIP_" + str(datetime.now()) + ".log")
    checkpoint_path = configs['global']["checkpoint_path"]
    # Load dataset
    train_path_json = configs['global']['train_path']
    valid_path_json = configs['global']['valid_path']
    test_path_json = configs['global']['test_path']
    
    train_json = open_json_file(train_path_json)
    valid_json = open_json_file(valid_path_json)
    test_json = open_json_file(test_path_json)
    
    train_data = dataLoader(CustomDataset(train_json, configs, preprocess), configs)
    test_data = dataLoader(CustomDataset(test_json, configs, preprocess), configs)
    valid_data = dataLoader(CustomDataset(valid_json, configs, preprocess), configs)

    logger.info("Number of train images: " + str(len(train_data)))
    logger.info("Number of validation images: " + str(len(valid_data)))

    
    loss_img = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    logger.info("Start training")

    for epoch in range(EPOCH):
        iter = 0
        for batch in train_data:
            optimizer.zero_grad()

            images, texts = batch
            # texts = process_text(texts)
            texts = clip.tokenize(texts)
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            
            image_loss = loss_img(logits_per_image,ground_truth)
            text_loss = loss_text(logits_per_text,ground_truth)

            total_loss = (image_loss + text_loss)/2
            
            print("Train - Iter: {} - {}, Epoch: {} , total loss: {}, image loss: {}, text loss: {}".format(iter*configs["global"]["BATCH_SIZE"], (iter+1)*configs["global"]["BATCH_SIZE"], epoch, total_loss, image_loss, text_loss))
            logger.info("Epoch %d Iter [%d:%d] Loss: %f" % (epoch, iter*configs["global"]["BATCH_SIZE"], (iter+1)*configs["global"]["BATCH_SIZE"], total_loss))

            total_loss.backward()

            if device == "cuda":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            iter += 1

        with torch.no_grad():
            loss_test = 0
            for batch in valid_data:
                images, texts = batch
                # texts = process_text(texts)
                texts = clip.tokenize(texts)
                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                image_loss = loss_img(logits_per_image,ground_truth)
                text_loss = loss_text(logits_per_text,ground_truth)
                batch_loss = (image_loss + text_loss)/2
                loss_test += batch_loss
            loss_test /= len(valid_data)
            print("Valid - Epoch: {} , total loss: {}, image loss: {}, text loss: {}".format(epoch, loss_test, image_loss, text_loss))

    if os.path.isfile(checkpoint_path) is False:
        with open(checkpoint_path, "r"):
            pass

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, checkpoint_path) 
    
if __name__ == "__main__":
    args = obtain_args()
    train(args)