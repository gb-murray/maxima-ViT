# Trains a ViT regression model from a given config.yaml

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.loss import Loss
from src.utils import create_model, load_model, freeze_backbone, train_one_epoch, validate
from src.data_pipeline import HDF5Dataset

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
    image_size=config['model'].get('image_size', 224)

    train_group = config['data'].get('train_group', 'train')
    val_group = config['data'].get('val_group', 'validation')
    train_dataset = HDF5Dataset(config['data']['hdf5_path'], train_group, image_size=image_size)
    val_dataset = HDF5Dataset(config['data']['hdf5_path'], val_group, image_size=image_size)

    num_workers = os.cpu_count() // 2 #type: ignore
    print(f"Using {num_workers} dataloading subprocesses.")

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
    loss_fn = Loss()
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

    print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ViT model.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    parser.add_argument("--load-checkpoint", help="Path to a model checkpoint to continue training.", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config, args.load_checkpoint)