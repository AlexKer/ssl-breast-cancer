import torch
import json
from dataloader import get_data_loader
from sklearn.model_selection import KFold, train_test_split
from trainer import trainer

def main(json_file, 
         log_dir, 
         epochs = 500, 
         warmup_epochs = 5, 
         lr = 5e-4, 
         min_lr = 1e-6, 
         shuffle_prob = 0.8, 
         dim = 128, 
         K = 68,
         resume_from = None,
         resetLR = False):

    with open(json_file, 'r') as f:
        data_list = json.load(f)
        print("Training on IPSY1 dataset")

    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(train_val_data_indices, test_size=0.10, shuffle=True, random_state=42)
    train_loader = get_data_loader("train", train_indices, data_list = data_list, batch_size = 4, drop_last=True)
    
    train = trainer(
        dataloader = train_loader,
        vis_loader = None,
        log_dir = log_dir,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        lr=lr,
        min_lr=min_lr,
        shuffle_prob=shuffle_prob,
        dim=dim,
        K=K,
        resume_from=resume_from,
        resetLR=resetLR
    )

    train.train()

if __name__ == "__main__":
    main(json_file="Path to json file", 
         log_dir="Path to log directory", # The dir to save the model checkpoints and logs
         resume_from=None, # Path to the checkpoint to resume training from
         lr=5e-4, # learning rate,
         min_lr=1e-6, # minimum learning rate
         resetLR=False) # If you want reset the learning rate with resume_from = True, set this to True and change the lr and min lrto the desired value
    