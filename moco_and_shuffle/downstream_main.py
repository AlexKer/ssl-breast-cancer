import torch
import torch.nn as nn
import json
from resnet import resnet50
from MoCov2_shuffle import MoCov2
from train_downstream import train_downstream
from dataloader import get_data_loader
from sklearn.model_selection import KFold, train_test_split
from model import Model
import torch.optim as optim

def main(json_file,
         model_path, 
         save_path,
         embedding_size=64,
         resume_from=None):
    
    torch.cuda.empty_cache()

    with open(json_file, 'r') as f:
        data_list = json.load(f)
        print("Training on IPSY1 dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_val_data_indices = list(range(len(data_list)))
    train_indices, val_indices = train_test_split(train_val_data_indices, test_size=0.10, shuffle=True, random_state=42)
    train_loader = get_data_loader("train", train_indices, data_list = data_list, batch_size = 2)

    downstream_model = resnet50(reduce=True)
    reduce_out_features = downstream_model.reduce[2].out_features
    original_reduce_q = downstream_model.reduce[:4]
    downstream_model.reduce = nn.Sequential(
        original_reduce_q,
        nn.Linear(reduce_out_features, reduce_out_features),
        nn.ReLU(),
        nn.Linear(reduce_out_features, 128)
    )

    optimizer = optim.AdamW(params=downstream_model.parameters(), lr=0.0001)
    dictionary = None

    current_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from)
        downstream_model.load_state_dict(checkpoint['model_state_dict'])
        current_epoch = checkpoint['epochos']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dictionary = checkpoint['dictionary']
    else:
        MoCov2_model = MoCov2()
        checkpoint = torch.load(model_path)
        MoCov2_model.load_state_dict(checkpoint['model_state_dict'])
        state_dict = MoCov2_model.encoder_q.state_dict()

    downstream_model.load_state_dict(state_dict, strict=False)
    downstream_model = Model(resnet=downstream_model, embedding_size=embedding_size).to(device)
    downstream_loader = get_data_loader("train", val_indices, data_list = data_list, batch_size = 3)

    best_epoch, best_accuracy, best_auc, best_f1, best_sensitivity, best_specificity = train_downstream(downstream_model, optimizer, train_loader, downstream_loader, downstream_loader, epochs=100-current_epoch, save_path=save_path, dictionary=dictionary)
    print("##################################")
    print("Minmize Loss: ", best_epoch)
    print("Best Accuracy: ", best_accuracy)
    print("Best Auc: ", best_auc)
    print("Best F1: ", best_f1)
    print("Best Sensitivity: ", best_sensitivity)
    print("Best Specificity: ", best_specificity)




