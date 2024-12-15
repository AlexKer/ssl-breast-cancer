import os
import logging
import numpy as np
import torch
import torch.distributed as dist
import json
import torch.optim as optim
import nibabel as nib
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from PIL import Image
from tqdm import tqdm
from model import Model
# from plot import plot_losses, plot_auc
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, roc_auc_score, precision_score, recall_score, f1_score
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from combined_loss import combined_loss

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train_downstream(model, optimizer, train_loader, test_loader, external_loader, epochs : int, save_path, dictionary=None):
    model.train()
    
    if not dictionary:
        dictionary['train_epoch_losses'] = []
        dictionary['test_epoch_losse'] = []
        dictionary['train_epoch_auc'] = []
        dictionary['test_epoch_auc'] = []
        dictionary['train_accury'] = []
        dictionary['test_accury'] = []
        dictionary['train_f1'] = []
        dictionary['test_f1'] = []
        dictionary['train_sensitivity'] = []
        dictionary['test_sensitivity'] = []
        dictionary['train_specificity'] = []
        dictionary['test_specificity'] = []
        dictionary['train_precision'] = []
        dictionary['test_precision'] = []

    best_epoch, best_auc_f1 = 0, 0
    best_accuracy, best_auc, best_f1, best_sensitivity, best_specificity = 0, 0, 0, 0, 0
    try:
        for epoch in range(epochs):
            model.train()
            losses = []
            total_samples = 0
            
            total_positives, true_positives = 0, 0
            total_negatives, true_negatives = 0, 0
            false_positives, false_negatives = 0, 0
            all_logits, all_labels, all_probs, all_preds = [], [], [], []
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")

            for batch_index, batch in progress_bar:
                optimizer.zero_grad()
                (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri, labels = batch

                t1_first_images = t1_first_images.cuda()
                t0_first_images = t0_first_images.cuda()
                
                labels = labels.cuda().float()
                if len(labels) == 1:
                    continue
                
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

                # check positives: gt 1, pred 1
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

            epoch_loss = np.mean(losses)
            epoch_accuracy = (true_positives + true_negatives) / total_samples

            all_logits = np.array(all_logits)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)

            # Calculate AUC, sensitivity/specificity
            epoch_auc = roc_auc_score(all_labels, all_probs)
            epoch_specificity = true_negatives / total_negatives if total_negatives != 0 else 0

            epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, zero_division=0)

            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

            print("LOSS: {:.3f}".format(epoch_loss), 
            " ACCURACY: {:.3f}".format(epoch_accuracy), 
            " AUC: {:.3f}".format(epoch_auc),
            " F1: {:.3f}".format(epoch_f1),
            " PRECISION: {:.3f}".format(epoch_precision), 
            " RECALL/SENSITIVITY: {:.3f}".format(epoch_recall), 
            " SPECIFICITY: {:.3f}".format(epoch_specificity))

            print("TOTAL SAMPLES: ", total_samples, "WITH ", total_positives, " POSITIVE SAMPLES. (", total_positives / total_samples * 100, "%)")


            test_loss, test_accuracy, test_auc, test_f1, test_precision, test_recall, test_specificity = test(model, test_loader)
            # external_loss, external_accuracy, external_auc, external_f1, external_precision, external_recall, external_specificity = test(model, external_loader)
            dictionary['train_epoch_losses'].append(epoch_loss)
            dictionary['train_epoch_auc'].append(epoch_auc)
            dictionary['train_accury'].append(epoch_accuracy)
            dictionary['train_f1'].append(epoch_f1)
            dictionary['train_sensitivity'].append(epoch_recall)
            dictionary['train_specificity'].append(epoch_specificity)
            dictionary['train_precision'].append(epoch_precision)
            
            dictionary['test_epoch_losses'].append(test_loss)
            dictionary['test_epoch_auc'].append(test_auc)
            dictionary['test_accury'].append(test_accuracy)
            dictionary['test_f1'].append(test_f1)
            dictionary['test_sensitivity'].append(test_recall)
            dictionary['test_specificity'].append(test_specificity)
            dictionary['test_precision'].append(test_precision)

            # Save checkpoint
            checkpoint_path =  f"checkpoint_epoch_{epoch}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dictionary': dictionary
            }
            save_checkpoint(checkpoint, checkpoint_path)
        total_count = total_positives + total_negatives
        print("TOTAL SAMPLES: ", total_count, "WITH ", total_positives, " POSITIVE SAMPLES. (", total_positives / total_count * 100, "%)")
        return min(dictionary['test_epoch_losses']), max(dictionary['test_accury']), max(dictionary['test_epoch_auc']), max(dictionary['test_f1']), max(dictionary['test_sensitivity']), max(dictionary['test_specificity'])
    
    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dictionary': dictionary
        }
        checkpoint_path = save_path / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
    except Exception as e:
        logging.error(f'Training failed with error: {str(e)}')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dictionary': dictionary
        }
        checkpoint_path = save_path / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        raise e

    finally:
        logging.info(f'Training finished.')
   
def test(model, test_loader):
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
                continue
            
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

            # check positives: gt 1, pred 1
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

    epoch_loss = np.mean(losses)
    epoch_accuracy = (true_positives + true_negatives) / total_samples

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Calculate AUC, sensitivity/specificity
    epoch_auc = roc_auc_score(all_labels, all_probs)
    epoch_sensitivity = true_positives / total_positives if total_positives != 0 else 0
    epoch_specificity = true_negatives / total_negatives if total_negatives != 0 else 0

    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)

    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("LOSS: {:.3f}".format(epoch_loss), 
    " ACCURACY: {:.3f}".format(epoch_accuracy), 
    " AUC: {:.3f}".format(epoch_auc),
    " F1: {:.3f}".format(epoch_f1),
    " PRECISION: {:.3f}".format(epoch_precision), 
    " RECALL/SENSITIVITY: {:.3f}".format(epoch_recall), 
    " SPECIFICITY: {:.3f}".format(epoch_specificity))

    return epoch_loss, epoch_accuracy, epoch_auc, epoch_f1, epoch_precision, epoch_recall, epoch_specificity