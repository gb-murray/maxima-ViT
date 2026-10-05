import torch
from torch import nn


class CountingBackbone(nn.Module):
    def __init__(self, image_size: int = 8, feature_dim: int = 6):
        super().__init__()
        self.calls = 0
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * image_size * image_size, feature_dim), nn.Tanh())

    def forward(self, images):
        self.calls += 1
        return self.network(images)
