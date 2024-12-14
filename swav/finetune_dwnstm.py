import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from dataloader import get_data_loader, balance_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from combined_loss import combined_loss 

class PretrainedSWaVModel(nn.Module):
    """
    Pre-trained SWaV model for feature extraction.
    
    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        resnet = torchvision.models.resnet50(weights=None)

        # Exclude the average pooling and FC layers
        backbone = nn.Sequential(*list(resnet.children())[:-1])  

        # Initialize Projection Head
        projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        # Extract and load backbone weights
        backbone_keys = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
        backbone_state_dict = {k.replace("backbone.", ""): v for k, v in backbone_keys.items()}
        try:
            backbone.load_state_dict(backbone_state_dict, strict=False)
            print("Backbone loaded successfully.")
        except Exception as e:
            print("Error loading backbone state_dict:", e)

        # Extract and load projection head weights
        projection_keys = {k: v for k, v in state_dict.items() if k.startswith("projection_head.")}
        projection_state_dict = {k.replace("projection_head.", ""): v for k, v in projection_keys.items()}
        try:
            projection_head.load_state_dict(projection_state_dict, strict=False)
            print("Projection head loaded successfully.")
        except Exception as e:
            print("Error loading projection_head state_dict:", e)

        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x):
        """
        Forward pass for the pre-trained SWaV model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W]
        
        Returns:
            torch.Tensor: Normalized feature embeddings of shape [B, 128]
        """
        # Check input dimensions
        if x.dim() != 5:
            raise ValueError(f"Expected input to have 5 dimensions [B, C, T, H, W], but got {x.dim()} dimensions.")

        # Average over the temporal dimension (T)
        x = torch.mean(x, dim=2)  # Shape: [B, C, H, W]

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  
        elif x.size(1) == 3:
            pass  
        else:
            raise ValueError(f"Unexpected number of channels: {x.size(1)}. Expected 1 or 3.")

        with torch.no_grad():
            feats = self.backbone(x).flatten(start_dim=1)   # Shape: [B, 2048]
        z = self.projection_head(feats)                     # Shape: [B, 128]
        z = nn.functional.normalize(z, dim=1)               # Normalize the embeddings
        return z


class DownstreamModel(nn.Module):
    """
    Downstream model for binary classification.
    
    Args:
        swav_model (PretrainedSWaVModel): Pre-trained SWaV model for feature extraction.
        freeze_encoder (bool, optional): Whether to freeze the SWaV encoder. Default is True.
    """

    def __init__(self, swav_model: PretrainedSWaVModel, freeze_encoder=True):
        super().__init__()
        self.swav_model = swav_model
        if freeze_encoder:
            for param in self.swav_model.parameters():
                param.requires_grad = False  # Freeze the SWaV encoder

        self.classifier = nn.Linear(128, 1)  # Binary classification head

    def forward(self, t0, t1, non_mri_features=None):
        """
        Forward pass for the downstream classification model.
        
        Args:
            t0 (torch.Tensor): Input tensor for modality t0 of shape [B, 1, T, H, W]
            t1 (torch.Tensor): Input tensor for modality t1 of shape [B, 1, T, H, W]
            non_mri_features (Optional[torch.Tensor]): Additional non-MRI features (unused)
        
        Returns:
            torch.Tensor: Logits for binary classification of shape [B]
        """
        # Extract features from both modalities using the pre-trained SWaV model
        feat_t0 = self.swav_model(t0)  # Shape: [B, 128]
        feat_t1 = self.swav_model(t1)  # Shape: [B, 128]

        # Combine features using average
        combined_feats = (feat_t0 + feat_t1) / 2.0  # Shape: [B, 128]

        # Add non-MRI features
        #if non_mri_features is not None:
             #combined_feats = torch.cat([combined_feats, non_mri_features.to(combined_feats.device)], dim=1)  # Shape: [B, 128 + F]

        # Pass through the classification head
        logits = self.classifier(combined_feats).squeeze()  # Shape: [B]
        return logits

def save_checkpoint(state, filename):
    """
    Save the training checkpoint.
    
    Args:
        state (dict): State dictionary containing model and optimizer states.
        filename (str): Path to save the checkpoint.
    """
    torch.save(state, filename)

def test(model, test_loader):
    """
    Basically from Taishi's code.
    Evaluate the model on the test dataset.
    
    Args:
        model (nn.Module): The downstream classification model.
        test_loader (DataLoader): DataLoader for the test dataset.
    
    Returns:
        Tuple containing loss, accuracy, AUC, F1, precision, recall, and specificity.
    """
    model.eval()
    losses = []
    total_samples = 0
    total_positives, true_positives = 0, 0
    total_negatives, true_negatives = 0, 0
    false_positives, false_negatives = 0, 0

    all_logits, all_labels, all_probs, all_preds = [], [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri, labels = batch
            t1_first_images = t1_first_images.cuda()
            t0_first_images = t0_first_images.cuda()
            labels = labels.cuda().float()
            if len(labels) == 1:
                continue  # Skip batches with a single sample to avoid BatchNorm issues
            logits = model(t0_first_images, t1_first_images, non_mri).squeeze()
            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            loss = combined_loss(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            total_samples += labels.size(0)
            for gt, pred in zip(labels, preds):
                if gt:
                    total_positives += 1
                    if pred:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    total_negatives += 1
                    if pred:
                        false_positives += 1
                    else:
                        true_negatives += 1

    epoch_loss = np.mean(losses) if losses else 0
    epoch_accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    epoch_sensitivity = true_positives / total_positives if total_positives > 0 else 0
    epoch_specificity = true_negatives / total_negatives if total_negatives > 0 else 0
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("TEST - LOSS: {:.3f}".format(epoch_loss),
          " ACCURACY: {:.3f}".format(epoch_accuracy),
          " AUC: {:.3f}".format(epoch_auc),
          " F1: {:.3f}".format(epoch_f1),
          " PRECISION: {:.3f}".format(epoch_precision),
          " RECALL/SENSITIVITY: {:.3f}".format(epoch_recall),
          " SPECIFICITY: {:.3f}".format(epoch_specificity))

    return epoch_loss, epoch_accuracy, epoch_auc, epoch_f1, epoch_precision, epoch_recall, epoch_specificity

def train_downstream(model, optimizer, train_loader, test_loader, epochs: int):
    """
    Basically from Taishi's code.
    Train the downstream classification model.
    
    Args:
        model (nn.Module): The downstream classification model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        epochs (int): Number of training epochs.
    """
    model.train()
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        total_samples = 0
        total_positives, true_positives = 0, 0
        total_negatives, true_negatives = 0, 0
        false_positives, false_negatives = 0, 0
        all_logits, all_labels, all_probs, all_preds = [], [], [], []
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_index, batch in progress_bar:
            optimizer.zero_grad()
            (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri, labels = batch
            t1_first_images = t1_first_images.cuda()
            t0_first_images = t0_first_images.cuda()
            labels = labels.cuda().float()
            if len(labels) == 1:
                continue  # Skip batches with a single sample to avoid BatchNorm issues
            logits = model(t0_first_images, t1_first_images, non_mri).squeeze()
            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            loss = combined_loss(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            total_samples += labels.size(0)
            for gt, pred in zip(labels, preds):
                if gt:
                    total_positives += 1
                    if pred:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    total_negatives += 1
                    if pred:
                        false_positives += 1
                    else:
                        true_negatives += 1

            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(losses) if losses else 0
        epoch_accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        epoch_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
        epoch_sensitivity = true_positives / total_positives if total_positives > 0 else 0
        epoch_specificity = true_negatives / total_negatives if total_negatives > 0 else 0
        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

        print("TRAIN - LOSS: {:.3f}".format(epoch_loss),
              " ACCURACY: {:.3f}".format(epoch_accuracy),
              " AUC: {:.3f}".format(epoch_auc),
              " F1: {:.3f}".format(epoch_f1),
              " PRECISION: {:.3f}".format(epoch_precision),
              " RECALL/SENSITIVITY: {:.3f}".format(epoch_recall),
              " SPECIFICITY: {:.3f}".format(epoch_specificity))

        # Evaluate on test set each epoch
        test_loss, test_accuracy, test_auc, test_f1, test_precision, test_recall, test_specificity = test(model, test_loader)
        if test_auc > best_auc:
            best_auc = test_auc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"best_downstream_checkpoint.pth")
            print(f"New best AUC: {best_auc:.3f}. Checkpoint saved.")

    return

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load dataset and create train/test loaders
    with open('SSL_dataset.json', 'r') as f:
        data_list = json.load(f)

    # Use half of the samples randomly
    half_size = len(data_list) // 2
    data_list = random.sample(data_list, half_size)

    # Split into training and validation sets
    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(
        train_val_data_indices, 
        test_size=0.10, 
        shuffle=True, 
        random_state=42
    )

    # Create DataLoaders
    train_loader = get_data_loader(
        "train", 
        train_indices, 
        data_list=data_list, 
        batch_size=1, 
        num_workers=0
    )

    test_loader = get_data_loader(
        "test", 
        val_indices, 
        data_list=data_list, 
        batch_size=1, 
        num_workers=0
    )

    # Load the pre-trained SWaV model checkpoint
    swav_checkpoint_path = 'checkpoints/swav_ft.ckpt'  # Update this path if necessary
    if not os.path.exists(swav_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {swav_checkpoint_path}")

    swav_model = PretrainedSWaVModel(checkpoint_path=swav_checkpoint_path).cuda()
    print("Pre-trained SWaV model loaded successfully.")

    # Create the downstream classification model
    model = DownstreamModel(swav_model, freeze_encoder=True).cuda()
    print("Downstream classification model created.")

    # Define the optimizer (only parameters of the classifier will be updated if encoder is frozen)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    print("Optimizer initialized.")

    # Train the downstream classifier
    epochs = 50  # Set the desired number of epochs
    train_downstream(model, optimizer, train_loader, test_loader, epochs=epochs)
    print("Training completed.")