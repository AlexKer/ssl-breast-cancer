import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import json
import torch.optim as optim
from data_utils import get_data_loader
from sklearn.model_selection import KFold, train_test_split
from train_ssl import train_spark_3d
import resnet
from model import Model
from train_downstream import train_downstream
from visualize import visualize

def main():
    torch.cuda.empty_cache()

    '''
    Make Data Loaders
    '''
    with open('ispy1.json', 'r') as f:
        data_list = json.load(f)
        print("Training on IPSY1 dataset")

    # with open('ispy2_four_timepoints.json', 'r') as f:
    #     data_list = json.load(f)
    #     print("Training on IPSY2 dataset")
    
    # Step 1: Split into train and test sets 
    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(train_val_data_indices, test_size=0.10, shuffle=True, random_state=42)
    train_loader = get_data_loader("train", train_indices, data_list = data_list, batch_size = 2)
    vis_loader = get_data_loader("test", val_indices, data_list = data_list, batch_size = 1)


    # print("----SPARK PRETRAINING-------")
    trained_spark_model = train_spark_3d(train_loader, vis_loader, epochs = 100)

    # load the pretrained resnet weights
    resnet50 = resnet.resnet50()
    checkpoint = torch.load("pretrained_resnet_epoch_381.pth")

    pretrained = False
    # Load the state_dict to the model
    if pretrained:
        print("Downstream Training with pretrained resnet")
        resnet50.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Downstream Training with UNTRAINED resnet")
    # print(resnet50)
    model = Model(resnet=resnet50).cuda()
    optimizer = optim.AdamW(params=model.parameters(), lr=0.0001)


    downstream_loader = get_data_loader("train", val_indices, data_list = data_list, batch_size = 3)
    best_epoch, best_accuracy, best_auc, best_f1, best_sensitivity, best_specificity = train_downstream(model, optimizer, train_loader, downstream_loader, downstream_loader, epochs=100)

            

if __name__ == "__main__":
    main()