# Trains a ViT regression model

import os
import sys
import argparse
import yaml
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision.transforms import Resize
from transformers import ViTModel

# Project Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.model import MaxViTModel
from src.loss import HuberLoss

# Data Loading
class HDF5Dataset(Dataset):
    """PyTorch Dataset for loading data from an HDF5 file."""
    def __init__(self, hdf5_path: str, group: str, image_size: int = 224):
        self.file = h5py.File(hdf5_path, 'r')
        self.group = self.file[group]
        self.images = self.group['images'] #type: ignore
        self.labels = self.group['labels'] #type: ignore
        self.resize_transform = Resize((image_size, image_size), antialias=True)

    def __len__(self):
        return len(self.images) #type: ignore

    def __getitem__(self, idx):
        image = self.images[idx] #type: ignore
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        image_tensor = self.resize_transform(image_tensor)
        image_tensor = image_tensor.repeat(3, 1, 1)
        label_tensor = torch.from_numpy(self.labels[idx]).float() #type: ignore
        return image_tensor, label_tensor

    def close(self):
        self.file.close()

# Model Creation and Freezing
def create_model(config: dict) -> nn.Module:
    # Instantiates a new model from a pre-trained backbone
    print(f"Creating new model from backbone: {config['model']['backbone']}")
    vit_backbone = ViTModel.from_pretrained(config['model']['backbone'])
    regression_head = nn.Sequential(
        nn.Linear(config['model']['vit_hidden_dim'], 512),
        nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, config['model']['num_outputs'])
    )
    return MaxViTModel(vit_backbone, regression_head)

def load_model(model_path: str, config: dict) -> nn.Module:
    # Loads an existing model 
    print(f"Loading existing model from: {model_path}")
    model = create_model(config)
    model.load_state_dict(torch.load(model_path))
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

# Main Execution
def main(config: dict, checkpoint_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or create model
    if checkpoint_path:
        model = load_model(checkpoint_path, config).to(device)
    else:
        model = create_model(config).to(device)

    # Conditionally freeze backbone
    if config['training'].get('freeze_backbone', False):
        freeze_backbone(model)

    # Create Datasets and DataLoaders
    train_dataset = HDF5Dataset(config['data']['hdf5_path'], 'train')
    val_dataset = HDF5Dataset(config['data']['hdf5_path'], 'validation')

    num_workers = os.cpu_count() // 2 #type: ignore

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
        )

    # Instantiate loss and optimizer
    loss_fn = HuberLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Main training loop
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, config['model']['name'])
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path}")

    train_dataset.close()
    val_dataset.close()
    print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ViT model.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    parser.add_argument("--load-checkpoint", help="Path to a model checkpoint to continue training.", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config, args.load_checkpoint)