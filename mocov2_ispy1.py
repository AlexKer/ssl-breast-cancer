from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import math
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from preprocess_ispy1 import SAVE_DIR  # Import the save directory from preprocess_ispy1

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'save_dir'

class ISPY1Pair(torch.utils.data.Dataset):
    """ISPY1 Dataset that loads pre/first/second post-contrast volumes for MoCo training"""
    def __init__(self, json_path, transform=None, debug=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if debug:
            self.data = self.data[:4]  # Only use 4 samples for testing
        self.transform = transform
        self.save_dir = SAVE_DIR  # From preprocess_ispy1

    def __getitem__(self, index):
        item = self.data[index]
        patient_id = item['patient_id']
        
        # Load all phases for t0 and t1
        t0_pre = np.load(f"{self.save_dir}/{patient_id}_t0_pre.npy")
        t0_first = np.load(f"{self.save_dir}/{patient_id}_t0_first.npy")
        t0_second = np.load(f"{self.save_dir}/{patient_id}_t0_second.npy")
        t0_third = t0_second  # Using second post-contrast as fourth phase for now
        print(f"t0pre:",t0_pre.shape)
        print(f"t0first:",t0_first.shape)
        print(f"t0sec:",t0_second.shape)
        print(f"t0third:",t0_third.shape)

        t1_pre = np.load(f"{self.save_dir}/{patient_id}_t1_pre.npy")
        t1_first = np.load(f"{self.save_dir}/{patient_id}_t1_first.npy")
        t1_second = np.load(f"{self.save_dir}/{patient_id}_t1_second.npy")
        t1_third = t1_second  # Using second post-contrast as fourth phase for now

        # Stack all phases together - now 4 phases like shuffle_ssl
        vol_t0 = np.stack([t0_pre, t0_first, t0_second, t0_third], axis=0)  # Shape: (4, D, H, W)
        vol_t1 = np.stack([t1_pre, t1_first, t1_second, t1_third], axis=0)  # Shape: (4, D, H, W)
        
        # Convert to tensor
        vol_t0 = torch.FloatTensor(vol_t0)  
        vol_t1 = torch.FloatTensor(vol_t1)

        if self.transform is not None:
            vol_t0 = self.transform(vol_t0)
            vol_t1 = self.transform(vol_t1)

        # Stack the two timepoints together
        volumes = torch.stack([vol_t0, vol_t1], dim=0)  # Shape: (2, 4, D, H, W)
        
        return volumes

    def __len__(self):
        return len(self.data)

class MRITransform:
    """Transform for 3D MRI volumes"""
    def __init__(self, size=(64, 64, 64)):
        self.size = size

    def __call__(self, volume):
        # Normalize each phase independently
        for i in range(volume.shape[0]):
            mean = volume[i].mean()
            std = volume[i].std()
            volume[i] = (volume[i] - mean) / (std + 1e-7)
        
        # Resize the volume to a smaller size to prevent memory issues
        # Use adaptive pooling to get desired size
        volume = F.adaptive_avg_pool3d(volume.unsqueeze(0), self.size).squeeze(0)
        
        return volume

class SplitBatchNorm(nn.BatchNorm3d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, D, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, D, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps).view(N, C, D, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps)

class ModelBase(nn.Module):
    def __init__(self, feature_dim=128, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm3d
        
        self.net = nn.Sequential(
            # Initial conv layer for 3 input channels (pre, first, second post-contrast)
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            self._make_layer(32, 64, 2, norm_layer),
            self._make_layer(64, 128, 2, norm_layer, stride=2),
            self._make_layer(128, 256, 2, norm_layer, stride=2),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )

    def _make_layer(self, in_channels, out_channels, blocks, norm_layer, stride=1):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(norm_layer(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks-1):
            layers.append(nn.Conv3d(out_channels, out_channels, 3, padding=1))
            layers.append(norm_layer(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=2048, m=0.99, T=0.07, bn_splits=8, symmetric=False):
        super(ModelMoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # Create encoders for each phase
        self.encoder_q = ModelBase(feature_dim=dim, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, bn_splits=bn_splits)

        # Initialize key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, volumes):
        # volumes shape: (batch_size, 2, 4, D, H, W)
        batch_size, num_timepoints, *rest = volumes.shape
        
        # Reshape like train_shuffle
        volumes = volumes.view(batch_size * num_timepoints, *rest)  # Shape: (batch_size*2, 4, D, H, W)
        
        # Split phases like shuffle_ssl
        first = volumes[:, 0, :, :, :]     # Pre-contrast
        second = volumes[:, 1, :, :, :]    # First post-contrast
        third = volumes[:, 2, :, :, :]     # Second post-contrast
        fourth = volumes[:, 3, :, :, :]    # Third post-contrast
        
        # Get features for all phases
        f1 = self.encoder_q(first)    
        f2 = self.encoder_k(second)  
        f3 = self.encoder_k(third)
        f4 = self.encoder_k(fourth)
        
        # Normalize features
        f1 = nn.functional.normalize(f1, dim=1)
        f2 = nn.functional.normalize(f2, dim=1)
        f3 = nn.functional.normalize(f3, dim=1)
        f4 = nn.functional.normalize(f4, dim=1)

        # Compute logits for all phases
        l_pos1 = torch.einsum('nc,nc->n', [f1, f2]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [f1, f3]).unsqueeze(-1)
        l_pos3 = torch.einsum('nc,nc->n', [f1, f4]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [f1, self.queue.clone().detach()])

        # Combine all logits
        logits = torch.cat([l_pos1, l_pos2, l_pos3, l_neg], dim=1)
        logits /= self.T

        # Labels: first three positions are positive
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Update queue with all positive features
        self._dequeue_and_enqueue(torch.cat([f2, f3, f4], dim=0))

        return logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)  # Changed from optimizer to train_optimizer

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for vol_t0, vol_t1 in train_bar:
        vol_t0 = vol_t0.to(device)
        vol_t1 = vol_t1.to(device)

        loss = net(vol_t0, vol_t1)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(
            epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num
def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def test(model, memory_data_loader, test_data_loader, epoch, args):
    model.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # Generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = model(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # Loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

# kNN classifier
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

class ISPY1Dataset(torch.utils.data.Dataset):
    """ISPY1 Dataset for kNN evaluation"""
    def __init__(self, json_path, train=True, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.train = train
        self.save_dir = SAVE_DIR
        # PCR status from json is used as the target/label
        self.targets = [item['label'] for item in self.data]  
        self.classes = sorted(list(set(self.targets)))

    def __getitem__(self, index):
        item = self.data[index]
        patient_id = item['patient_id']
        
        # Load t0 data only
        t0_data = np.load(f"{self.save_dir}/{patient_id}/t0.npy")
        t0_pre = t0_data[0]
        t0_first = t0_data[1]
        t0_second = t0_data[2]
        
        volume = np.stack([t0_pre, t0_first, t0_second], axis=0)
        volume = torch.FloatTensor(volume)

        if self.transform is not None:
            volume = self.transform(volume)

        target = self.targets[index]
        return volume, target

    def __len__(self):
        return len(self.data)

# Move all the execution code into a main() function
def main():
    # Data loading
    train_transform = MRITransform(size=(32, 32, 32))  # Smaller size
    train_data = ISPY1Pair(json_path='non_mri_data.json', 
                          transform=train_transform,
                          debug=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True,
        drop_last=True
    )

    # Data loading for memory bank and testing
    memory_data = ISPY1Dataset(json_path='non_mri_data.json', 
                              train=True, 
                              transform=train_transform)
    memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True
    )

    test_data = ISPY1Dataset(json_path='non_mri_data.json', 
                            train=False, 
                            transform=train_transform)
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True
    )

    # Create model
    model = ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
    ).to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # Load model if resume
    epoch_start = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch'] + 1
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Logging setup
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Dump args
    with open(os.path.join(args.results_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # Training loop
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        
        # Test
        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        
        # Save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(os.path.join(args.results_dir, 'log.csv'), index_label='epoch')
        
        # Save model
        save_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # Save latest model
        torch.save(save_dict, os.path.join(args.results_dir, 'model_last.pth'))
        
        # Save best model based on test accuracy
        if test_acc_1 == max(results['test_acc@1']):
            torch.save(save_dict, os.path.join(args.results_dir, 'model_best.pth'))
        
        print(f'Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Test Acc@1: {test_acc_1:.2f}%')

    print("Training completed!")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MoCo on ISPY1')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--moco-dim', default=128, type=int)
    parser.add_argument('--moco-k', default=2048, type=int)
    parser.add_argument('--moco-m', default=0.99, type=float)
    parser.add_argument('--moco-t', default=0.07, type=float)
    parser.add_argument('--bn-splits', default=8, type=int)
    parser.add_argument('--symmetric', action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH',
                        help='path to cache (default: none)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='wd')
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                        help='number of warmup epochs (default: 10)')

    args = parser.parse_args()
    args.cos = True
    
    # For testing purposes, override some args
    args.epochs = 2
    args.batch_size = 2
    args.moco_k = 16
    args.moco_dim = 128
    
    # Setup results directory and logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Dump args
    with open(os.path.join(args.results_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # Run main function
    main()