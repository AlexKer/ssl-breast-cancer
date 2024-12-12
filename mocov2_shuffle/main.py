import torch
import json
from dataloader import get_data_loader
from sklearn.model_selection import KFold, train_test_split
from trainer import trainer

def main(json_file):
    with open(json_file, 'r') as f:
        data_list = json.load(f)
        print("Training on IPSY1 dataset")

    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(train_val_data_indices, test_size=0.10, shuffle=True, random_state=42)
    train_loader = get_data_loader("train", train_indices, data_list = data_list, batch_size = 2)
    vis_loader = get_data_loader("test", val_indices, data_list = data_list, batch_size = 1)
    
    train = trainer(
        dataloader = train_loader,
        log_dir = "logs",
        epochs=500,
        warmup_epochs=5,
        lr=0.005,
        min_lr=1e-6,
        shuffle_prob=0.8
    )

    train.train()

if __name__ == "__main__":
    main(r"Path to json file")
    