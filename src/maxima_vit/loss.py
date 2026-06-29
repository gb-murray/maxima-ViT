# Main loss function (Huber)

import torch
import torch.nn as nn
import numpy as np

class Loss(nn.Module):
    """
    A custom loss function that calculates a weighted, normalized Huber loss for the 6D geometry. 
    """
    def __init__(self, centers: np.ndarray, scale_factors: np.ndarray, weights: list):
        super().__init__()
        self.register_buffer('centers', torch.tensor(centers, dtype=torch.float32))
        self.register_buffer('scale_factors', torch.tensor(scale_factors, dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the final loss value.
        """
        pred_norm = (pred - self.centers) / self.scale_factors
        target_norm = (target - self.centers) / self.scale_factors

        loss = self.huber(pred_norm, target_norm)
        weighted_loss = loss * self.weights
        
        return weighted_loss.mean()