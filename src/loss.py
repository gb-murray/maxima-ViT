# Main loss function (Huber)

import torch
import torch.nn as nn

class Loss(nn.Module):
    """
    A custom loss function that calculates a weighted, normalized Huber loss for the 6D geometry. 
    """
    def __init__(self):
        super().__init__()
        # [dist, poni1, poni2, rot1, rot2, rot3]
        self.scale_factors = torch.tensor([0.30, 0.06, 0.06, 0.25, 0.25, 0.25], dtype=torch.float32)
        
        # Define weights to tune the importance of each parameter
        self.weights = torch.tensor([2.5, 10.0, 10.0, 5.0, 5.0, 1.0], dtype=torch.float32)
        
        # Initialize the base loss function
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the final loss value.
        """
        device = y_pred.device
        self.scale_factors = self.scale_factors.to(device)
        self.weights = self.weights.to(device)

        # Normalize both prediction and ground truth by the scale factors
        y_pred_norm = y_pred / self.scale_factors
        y_true_norm = y_true / self.scale_factors

        # Calculate the per-element Huber loss on the normalized values
        per_element_loss = self.huber(y_pred_norm, y_true_norm)
        
        # Apply the weights
        weighted_loss_components = per_element_loss * self.weights
        
        # Sum the weighted losses for each sample in the batch
        loss_per_sample = torch.sum(weighted_loss_components, dim=1)
        
        # Return the mean loss across the entire batch
        return torch.mean(loss_per_sample)