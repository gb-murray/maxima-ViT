# Main model object, requires pre-loaded backbone and head

import torch.nn as nn

class XRDCalibrator(nn.Module):
    def __init__(self, vit_backbone, regression_head):
        super().__init__()
        self.vit = vit_backbone
        self.head = regression_head

    def forward(self, pixel_values):
        # Pass the image through the ViT backbone
        outputs = self.vit(pixel_values=pixel_values)
        
        # Extract the final feature vector for the [CLS] token
        # This token is designed to aggregate the global image representation
        feature_vector = outputs.last_hidden_state[:, 0, :]
        
        # Pass the features to the regression head
        geometry_params = self.head(feature_vector)
        
        return geometry_params