# Main model object, requires pre-loaded backbone and head

import torch.nn as nn

class MaxViTModel(nn.Module):
    def __init__(self, vit_backbone, regression_head):
        super().__init__()
        self.vit = vit_backbone
        self.head = regression_head

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        feature_vector = outputs.last_hidden_state[:, 0, :]
        geometry_params = self.head(feature_vector)
        
        return geometry_params