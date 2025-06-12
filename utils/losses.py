# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss for semantic segmentation.
    
    Reference: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    This loss is a modification of CrossEntropyLoss that down-weights the loss assigned
    to well-classified examples, forcing the model to focus on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        """
        Args:
            alpha (float): The alpha weighting factor to balance positive/negative examples.
                           Can also be a Tensor of size C (num_classes).
            gamma (float): The focusing parameter. Higher values give more focus to hard examples.
            ignore_index (int): Specifies a target value that is ignored and does not contribute
                                to the input gradient.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the loss.
        
        Args:
            inputs (torch.Tensor): The model's raw output (logits).
                                   Shape: (N, C, H, W).
            targets (torch.Tensor): The ground truth labels.
                                    Shape: (N, H, W).
        Returns:
            torch.Tensor: The calculated focal loss.
        """
        # Calculate the Cross-Entropy loss but without any reduction
        # This gives us the per-pixel negative log probability (log_pt)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get the probability of the correct class (pt)
        pt = torch.exp(-ce_loss)
        
        # Calculate the focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply the reduction method (mean, sum, etc.)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
