import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples,
               or a list of weights for each class.
        gamma: Focusing parameter for modulating loss (gamma >= 0).
               Higher gamma reduces loss for well-classified examples.
        reduction: Specifies the reduction to apply to the output.
        
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N) where each value is 0 <= targets[i] <= C-1
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get the probability of the true class for each example
        # targets is shape (N,), we need to gather the correct class probs
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get p_t: probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss without alpha
        focal_loss = focal_term * ce_loss
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Alpha is a tensor of weights per class
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
