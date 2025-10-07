# Helper functions

from pyFAI.calibrant import Calibrant, CALIBRANT_FACTORY
from pyFAI.detectors import Detector, detector_factory
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from model import MaxViTModel
from tqdm import tqdm

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
    Loads interpolated weights into a fresh model architecture.
    """
    print(f"Loading custom interpolated weights from: {model_path}")
    
    # Create the correctly-sized model architecture
    model = create_model(config)
    
    # Load the state dictionary from our interpolated file
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load these weights into the model's ViT backbone
    model.vit.load_state_dict(state_dict, strict=False) #type: ignore
    
    return model

def freeze_backbone(model: nn.Module):
    """Freezes the parameters of the ViT backbone."""
    print("Freezing backbone weights. Only the regression head will be trained.")
    for param in model.vit.parameters(): #type: ignore
        param.requires_grad = False

# Training and Validation Loops
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    # Trains a single epoch
    model.train()
    total_loss = 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    # Validates the model
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)