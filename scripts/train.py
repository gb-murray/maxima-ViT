# Trains a ViT regression model

import os
import sys
import argparse
import yaml
import h5py
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from torchvision.transforms import Resize
from transformers import ViTModel, ViTConfig
from sklearn.model_selection import KFold

# Project Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.model import MaxViTModel
from src.loss import Loss

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

# Main Execution
def main(config: dict, checkpoint_path: str = None): #type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # K-Fold Cross-Validation Setup
    k_folds = config['training'].get('k_folds', 5)
    
    # Load the entire training pool dataset
    full_dataset = HDF5Dataset(config['data']['hdf5_path'], 'training_pool', 
                               image_size=config['model'].get('image_size', 224))
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):  #type: ignore
        print(f"\n{'='*20} FOLD {fold + 1}/{k_folds} {'='*20}")

        # Create Data Samplers for the current fold
        train_sampler = SubsetRandomSampler(train_ids)  #type: ignore
        val_sampler = SubsetRandomSampler(val_ids)  #type: ignore

        # Create DataLoaders for the current fold
        train_loader = DataLoader(
            full_dataset, 
            batch_size=config['training']['batch_size'], 
            sampler=train_sampler
        )
        val_loader = DataLoader(
            full_dataset,
            batch_size=config['training']['batch_size'],
            sampler=val_sampler
        )
        
        # Re-initialize the model for each fold to ensure an unbiased evaluation
        print("Initializing a fresh model for this fold...")
        model = create_model(config).to(device)
        
        loss_fn = Loss() 
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate'],
            fused=True 
        )
        
        best_val_loss = float('inf')
        
        # Run the training loop for the current fold
        for epoch in range(config['training']['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = validate(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"Best validation loss for fold {fold + 1}: {best_val_loss:.6f}")
        fold_results.append(best_val_loss)
    
    # Final Cross-Validation Results
    print(f"\n{'='*20} K-FOLD CROSS-VALIDATION RESULTS {'='*20}")
    results_np = np.array(fold_results)
    mean_loss = np.mean(results_np)
    std_loss = np.std(results_np)
    
    print(f"Finished {k_folds}-fold cross-validation.")
    print(f"Average Validation Loss: {mean_loss:.6f}")
    print(f"Standard Deviation: {std_loss:.6f}")

    full_dataset.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ViT model.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    parser.add_argument("--load-checkpoint", help="Path to a model checkpoint to continue training.", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config, args.load_checkpoint)