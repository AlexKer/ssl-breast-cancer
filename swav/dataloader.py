import torch
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
from typing import List, Optional
import random
import torchvision.transforms as T

def balance_data(data_list, label_key='label'):
    """
    From TaiSHi's code.

    Balances the dataset by upsampling the minority class.

        :param data_list: List[dict] of data samples
        :param label_key: key of the label in the data sample 
        
        :return: List[dict] of balanced data samples
    """
    positive_samples = [data for data in data_list if data[label_key] == 1]
    negative_samples = [data for data in data_list if data[label_key] == 0]

    num_negatives = len(negative_samples)
    num_positives = len(positive_samples)
    if num_positives == 0:
        return data_list

    upsample_factor = max(1, num_negatives // num_positives)
    balanced_data_list = negative_samples + positive_samples * upsample_factor
    random.shuffle(balanced_data_list)
    return balanced_data_list

class MultiCropTransform:
    """
    - 2 global, 4 local total
    - Output size (224, 224)
    - Augmentations: horizontal flip, intensity scaling, random rotation, Gaussian blur
    """
    def __init__(self,
                 global_crops_scale=(0.7, 1.0),
                 local_crops_scale=(0.5, 0.7),
                 n_local_crops=4):
        self.n_local_crops = n_local_crops

        global_transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=global_crops_scale),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=5)
        ])

        local_transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=local_crops_scale),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=5)
        ])

        self.global_transform = tio.Compose([
            tio.transforms.CropOrPad((80, 256, 256)), 
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])
        self.local_transform = tio.Compose([
            tio.transforms.CropOrPad((80, 256, 256)),
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])

        self.global_torch_transforms = global_transforms
        self.local_torch_transforms = local_transforms

    def __call__(self, volume):
        # volume: [1, T, H, W]

        # 2 global crops
        global_crops = []
        for _ in range(2):
            g_vol = self.global_transform(volume)
            g_img = self.global_torch_transforms(g_vol)
            global_crops.append(g_img)

        # 4 local crops
        local_crops = []
        for _ in range(self.n_local_crops):
            l_vol = self.local_transform(volume)
            l_img = self.local_torch_transforms(l_vol)
            local_crops.append(l_img)

        return global_crops + local_crops

class NumpyDataset(Dataset):
    """
    From's Taishi's code. 
    Dataset for loading numpy arrays.
    """
    def __init__(self, data_list: List[dict], multi_crop_transform: MultiCropTransform):
        self.data_list: List[dict] = data_list
        self.multi_crop_transform = multi_crop_transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        t0_first = torch.tensor(np.load(self.data_list[idx]['image_t0_first']).astype(np.float32)).unsqueeze(0)
        t1_first = torch.tensor(np.load(self.data_list[idx]['image_t1_first']).astype(np.float32)).unsqueeze(0)
        t2_first = torch.tensor(np.load(self.data_list[idx]['image_t2_first']).astype(np.float32)).unsqueeze(0)
        t3_first = torch.tensor(np.load(self.data_list[idx]['image_t3_first']).astype(np.float32)).unsqueeze(0)

        non_mri_features = self.data_list[idx]['non_mri_features']
        label = self.data_list[idx]['label']

        t0_crops = self.multi_crop_transform(t0_first)
        t1_crops = self.multi_crop_transform(t1_first)
        t2_crops = self.multi_crop_transform(t2_first)
        t3_crops = self.multi_crop_transform(t3_first)

        transformed_images = list(zip(t0_crops, t1_crops, t2_crops, t3_crops))
        n_crops = len(transformed_images)  # 6 total (2 global + 4 local)

        return transformed_images, [non_mri_features] * n_crops, [label] * n_crops

def collate_fn(batch):
    """
    From's Taishi's code.
    Custom collate fn for dealing with batches of image data and labels.
    """
    images, features, labels = zip(*batch)
    t0_first_images = []
    t1_first_images = []
    t2_first_images = []
    t3_first_images = []
    non_mri_features = []
    flattened_labels = []

    for image_set, feature_set, label_set in zip(images, features, labels):
        for (t0_first, t1_first, t2_first, t3_first) in image_set:
            t0_first_images.append(t0_first)
            t1_first_images.append(t1_first)
            t2_first_images.append(t2_first)
            t3_first_images.append(t3_first)

        non_mri_features.extend(feature_set)
        flattened_labels.extend(label_set)

    t0_first_images = torch.stack(t0_first_images)
    t1_first_images = torch.stack(t1_first_images)
    t2_first_images = torch.stack(t2_first_images)
    t3_first_images = torch.stack(t3_first_images)
    flattened_labels = torch.tensor(flattened_labels, dtype=torch.long)

    return (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri_features, flattened_labels

def get_data_loader(type: str, indices, data_list, batch_size=1, num_workers=0):
    multi_crop_transform = MultiCropTransform(
        global_crops_scale=(0.7, 1.0),
        local_crops_scale=(0.5, 0.7),
        n_local_crops=4 
    )

    if type in ["train", "ft_train"]:
        data = [data_list[i] for i in indices]
        balanced_data_list = balance_data(data)
    else:
        balanced_data_list = [data_list[i] for i in indices]

    dataset = NumpyDataset(data_list=balanced_data_list, multi_crop_transform=multi_crop_transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(type in ["train", "ft_train"]),
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader