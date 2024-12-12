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
from plot import plot_losses
import torch.nn as nn
from encoder import SparseEncoder
from decoder import LightDecoder
import resnet

def visualize_slices(input_image, masked_image, hybrid_image, output_dir, batch_idx):
    """
    Visualize and save slices of input, masked, and hybrid images.

    Args:
        input_image (torch.Tensor): The original input image [1, 1, depth, height, width].
        masked_image (torch.Tensor): The masked image [1, 1, depth, height, width].
        hybrid_image (torch.Tensor): The reconstructed image [1, 1, depth, height, width].
        output_dir (str): Directory to save the visualized slices.
    """
    input_image = input_image.squeeze().cpu().numpy()  # Shape: [depth, height, width]
    masked_image = masked_image.squeeze().cpu().numpy()
    hybrid_image = hybrid_image.squeeze().detach().cpu().numpy()

    depth = input_image.shape[0]

    for slice_idx in range(depth):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original input image
        axs[0].imshow(input_image[slice_idx], cmap="gray")
        axs[0].set_title(f"Input Image - Slice {slice_idx}")
        axs[0].axis("off")

        # Masked image
        axs[1].imshow(masked_image[slice_idx], cmap="gray")
        axs[1].set_title(f"Masked Image - Slice {slice_idx}")
        axs[1].axis("off")

        # Hybrid image
        axs[2].imshow(hybrid_image[slice_idx], cmap="gray")
        axs[2].set_title(f"Hybrid Image - Slice {slice_idx}")
        axs[2].axis("off")

        # Save the plot
        slice_path = os.path.join(output_dir, f"{batch_idx}_slice_{slice_idx}.png")
        plt.savefig(slice_path, bbox_inches="tight")
        plt.close(fig)

        # print(f"Saved slice {slice_idx} visualization to {slice_path}")

def visualize(pretrained_resnet, data_loader):
    input_size = (80, 256, 256)
    downsample_ratio = (16, 32, 32)
    encoder : SparseEncoder = SparseEncoder(pretrained_resnet, input_size = input_size)
    decoder : LightDecoder = LightDecoder(downsample_ratio, channel_width=2048)

    spark_model = SparkModel(
                sparse_encoder = encoder, 
                dense_decoder = decoder
            ).cuda()
    
    spark_model.eval()
    for batch_idx, batch in enumerate(data_loader): 
        if batch_idx % 50 == 0:
            (t0, t1, t2, t3), non_mri, labels = batch
            t0 = t0.cuda()
            input_image, masked_image, hybrid_image = spark_model.forward(t0, vis=True)

            # print("shapes: ", input_image.shape, masked_image.shape, hybrid_image.shape)

            
            current_dir = os.getcwd()
            save_dir = os.path.join(os.path.dirname(current_dir), "spark_3d/visualization") 
            visualize_slices(input_image, masked_image, hybrid_image, save_dir, batch_idx)

def visualize2(spark_model, data_loader):
    spark_model.eval()
    for batch_idx, batch in enumerate(data_loader): 
        if batch_idx % 5 == 0:
            (t0, t1, t2, t3), non_mri, labels = batch
            t0 = t0.cuda()
            input_image, masked_image, hybrid_image = spark_model.forward(t0, vis=True)

            # print("shapes: ", input_image.shape, masked_image.shape, hybrid_image.shape)

            
            current_dir = os.getcwd()
            save_dir = os.path.join(os.path.dirname(current_dir), "spark_3d/val_vis") 
            visualize_slices(input_image, masked_image, hybrid_image, save_dir, batch_idx)