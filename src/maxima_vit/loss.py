# Main loss function (Huber)

import torch
import torch.nn as nn
import numpy as np

class Loss(nn.Module):
    """
    A custom loss function that calculates a weighted Huber loss for the 6D geometry.
    """
    def __init__(self, weights: list):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the final loss value.
        """
        loss = self.huber(pred, target)
        weighted_loss = loss * self.weights
        
        return weighted_loss.mean()