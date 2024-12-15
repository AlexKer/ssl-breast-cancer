import torch
import torch.nn.functional as F

def focal_loss(logits, labels, alpha, gamma):
    # BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    
    # Calculate the probabilities using sigmoid
    probs = torch.sigmoid(logits)
    # Compute the focal loss
    pt = torch.where(labels == 1, probs, 1 - probs)
    
    loss = alpha * (1 - pt).pow(gamma) * bce_loss
    
    return loss.mean()

# Final loss function combining weighted BCE and focal loss
def combined_loss(logits, labels, alpha=0.35, gamma=3.0):
    # Calculate weighted BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction = 'mean')

    # Calculate focal loss
    focal = focal_loss(logits, labels, alpha=alpha, gamma=gamma)
    
    # Combine the two losses
    # return 0.5 * bce_loss + 0.5 * focal
    return bce_loss

