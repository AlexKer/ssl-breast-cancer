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
from spark import SparkModel
from tqdm import tqdm
import torch.nn as nn
from encoder import SparseEncoder
from decoder import LightDecoder
import resnet
from torch.cuda.amp import autocast, GradScaler 
torch.cuda.empty_cache() 
from visualize import visualize2


def train_spark_3d(train_loader, vis_loader, epochs=3, lr=0.0001, resnet_type=50):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the GPU you want to use
    device = torch.device("cuda")

    resnet50 : resnet.ResNet = resnet.resnet50()

    input_size = (80, 256, 256)
    downsample_ratio = (16, 32, 32)
    encoder : SparseEncoder = SparseEncoder(resnet50, input_size = input_size)
    decoder : LightDecoder = LightDecoder(downsample_ratio, channel_width=2048)

    spark_model = SparkModel(
                sparse_encoder = encoder, 
                dense_decoder = decoder,
                mask_ratio=0.4
            ).to(device)

    optimizer = optim.AdamW(spark_model.parameters(), lr=lr)
    
    checkpoint_path = "pretrained_resnet_epoch_281.pth"
    checkpoint = torch.load(checkpoint_path)

    spark_model.sparse_encoder.sp_cnn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the state_dict to the model
    # spark_model.sparse_encoder.sp_cnn.load_state_dict(checkpoint)


    scaler = GradScaler()

    losses = []
    for epoch in range(epochs):
        spark_model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        epoch_loss = []
        for batch_index, batch in progress_bar:
            (t0, t1, t2, t3), non_mri, labels = batch

            t0 = t0.cuda()
            # t1 = t1.cuda()
            t2 = t2.cuda()
            t3 = t3.cuda()

            with autocast():
                l0 = spark_model(t0)
                # l1 = spark_model(t1)
                l2 = spark_model(t2)
                l3 = spark_model(t3)
                loss = (l0 + l2 + l3) / 3
           
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        print(f"Loss: {np.mean(epoch_loss)}")
        losses.append(epoch_loss)
        # Save the encoder's state_dict
        checkpoint_path = f"pretrained_resnet_epoch_{epoch}.pth"
        torch.save(spark_model.sparse_encoder.sp_cnn.state_dict(), checkpoint_path)
        torch.save({
            'model_state_dict': spark_model.sparse_encoder.sp_cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        visualize2(spark_model, vis_loader)
