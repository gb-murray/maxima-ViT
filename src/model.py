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
    
class MaxViTMultiHead(nn.Module):
    def __init__(self, vit_backbone, hidden_dim=768):
        super().__init__()
        self.vit = vit_backbone
    
        self.head_dist = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        self.head_poni = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

        self.head_rot = nn.Sequential(
            nn.Linear(hidden_dim, 512), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

    def forward(self, pixel_values):
        
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]

        dist_pred = self.head_dist(cls_token) # Shape: (B, 1)
        poni_pred = self.head_poni(cls_token) # Shape: (B, 2)
        rot_pred = self.head_rot(cls_token)   # Shape: (B, 3)

        return torch.cat([dist_pred, poni_pred, rot_pred], dim=1)
    
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