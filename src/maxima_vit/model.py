# Main model object, requires pre-loaded backbone and head

import torch
import torch.nn as nn

class MaxViT(nn.Module):
    def __init__(self, vit_backbone, regression_head):
        super().__init__()
        self.vit = vit_backbone
        self.head = regression_head

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        feature_vector = outputs.last_hidden_state[:, 0, :]
        geometry_params = self.head(feature_vector)
        
        return geometry_params
    
    
class MaxSWIN(nn.Module):
    def __init__(self, swin_backbone, regression_head):
        super().__init__()
        self.swin = swin_backbone
        self.head = regression_head
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, pixel_values):
        outputs = self.swin(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state               # shape: (B, N, C)
        feature_transpose = last_hidden_state.transpose(1, 2)       # shape: (B, C, N)
        feature_pool = self.pooler(feature_transpose).flatten(1)    # shape: (B, C)
        geometry_params = self.head(feature_pool)                   # shape: (B, num_outputs)

        return geometry_params