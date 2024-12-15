import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import json
import torch.optim as optim
import nibabel as nib
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from shuffle_ssl import SSLModel
from tqdm import tqdm
from model import Model
import torch.nn as nn


def train_ssl_model(train_loader, epochs=50, lr=0.005, resnet_type=50):
    model = SSLModel(resnet_type=resnet_type).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for batch_index, batch in progress_bar:
            optimizer.zero_grad()
            images, labels = batch

            batch_size, num_samples, *rest = images.shape
            images = images.view(batch_size * num_samples, *rest)  # Now shape: (18, 4, 60, 230, 230)
            labels = labels.view(-1).cuda().float()

            # print(images.shape)
            # print(labels.shape)

            first_images = images[:, 0, :, :, :].cuda()
            second_images = images[:, 1, :, :, :].cuda()  
            third_images = images[:, 2, :, :, :].cuda()
            fourth_images = images[:, 3, :, :, :].cuda()  

            # Get features for both views
            logits = model(first_images, second_images, third_images, fourth_images)
            labels = labels.unsqueeze(1)

            # Contrastive loss (adjust to your SSL task)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# Example usage
# train_ssl_model(data_loader, epochs=20, resnet_type=50)