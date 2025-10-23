# Helper functions

from pyFAI.calibrant import Calibrant, CALIBRANT_FACTORY
from pyFAI.detectors import Detector, detector_factory
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from .model import MaxViTModel
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torchvision import transforms
import numpy as np
from sklearn.metrics import mean_absolute_error

def get_calibrant(alias: str, wavelength: float) -> Calibrant:

    calibrant = CALIBRANT_FACTORY(alias)
    calibrant.wavelength = wavelength

    return calibrant

def get_detector(alias:str) -> Detector:

    detector = detector_factory(alias)

    return detector

def create_model(config: dict) -> nn.Module:

    """
    Instantiates a model with the specified architecture and random weights.
    """
    print(f"Creating new model architecture from config...")
    
    # Load the configuration of the pretrained model
    model_config = ViTConfig.from_pretrained(
        config['model']['backbone'],
        image_size=config['model'].get('image_size', 224)
    )
    
    # Build the model from the configuration (initializes with random weights)
    vit_backbone = ViTModel(model_config)

    # Create the regression head
    regression_head = nn.Sequential(
        nn.Linear(config['model']['vit_hidden_dim'], 512),
        nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, config['model']['num_outputs'])
    )
    return MaxViTModel(vit_backbone, regression_head)

def load_model(model_path: str, config: dict) -> nn.Module:
    """
    Loads pre-trained weights into a fresh model architecture.
    """
    print(f"Loading weights from: {model_path}")
    
    model = create_model(config)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.vit.load_state_dict(state_dict, strict=False) #type: ignore
    
    return model

def freeze_backbone(model: nn.Module):
    """
    Freezes the parameters of the ViT backbone.
    """
    print("Freezing backbone weights. Only the regression head will be trained.")
    for param in model.vit.parameters(): #type: ignore
        param.requires_grad = False

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler, writer, epoch):
    """
    Worker function to train model weights for a single epoch.
    """
    model.train()
    total_loss = 0

    num_batches = len(dataloader)

    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        is_last_batch = (i == num_batches - 1)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            predictions = model(images)
            loss = loss_fn(predictions, labels)
        
        scaler.scale(loss).backward()

        if is_last_batch:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / num_batches

def validate(model, dataloader, loss_fn, device, writer, epoch):
    """
    Returns validation loss for the current model/epoch.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    """
    Converts a 2D diffraction pattern from an array to a tensor.
    The image is padded and resampled to fit model specifications.
    """
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
    _ , h, w = image_tensor.shape
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    padding_transform = transforms.Pad(padding=(pad_w, pad_h))
    padded_tensor = padding_transform(image_tensor)

    resize_transform = transforms.Resize((image_size, image_size), antialias=True)
    final_tensor = resize_transform(padded_tensor)

    final_tensor = final_tensor.repeat(3, 1, 1)

    return final_tensor