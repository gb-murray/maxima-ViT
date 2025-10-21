# Performs k-fold cross validation 

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.amp.grad_scaler import GradScaler
from sklearn.model_selection import KFold

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.loss import Loss
from src.utils import create_model, train_one_epoch, validate
from src.data_pipeline import HDF5Dataset

def main(config: dict, checkpoint_path: str = None): #type: ignore
    """
    Runs cross-validation for k datafolds. 
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join(config['paths']['output_dir'], 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    k_folds = config['training'].get('k_folds', 5)
    
    # Load the entire training pool dataset into folds
    full_dataset = HDF5Dataset(config['data']['hdf5_path'], 'training_pool', 
                               image_size=config['model'].get('image_size', 224))
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):  #type: ignore
        print(f"\n{'='*20} FOLD {fold + 1}/{k_folds} {'='*20}")

        train_sampler = SubsetRandomSampler(train_ids)  #type: ignore
        val_sampler = SubsetRandomSampler(val_ids)  #type: ignore

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
        
        model = create_model(config).to(device)
        
        loss_fn = Loss() 
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate'],
            fused=True 
        )
        
        best_val_loss = float('inf')
        
        scaler = GradScaler(device=device.type)

        # Run the training loop for the current fold
        for epoch in range(config['training']['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, writer, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)

            val_loss = validate(model, val_loader, loss_fn, device, writer, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"Best validation loss for fold {fold + 1}: {best_val_loss:.6f}")
        fold_results.append(best_val_loss)
    
    # Final Cross-Validation Results
    print(f"\n --- K-FOLD CROSS-VALIDATION RESULTS ---")
    results_np = np.array(fold_results)
    mean_loss = np.mean(results_np)
    std_loss = np.std(results_np)
    
    print(f"Finished {k_folds}-fold cross-validation.")
    print(f"Average Validation Loss: {mean_loss:.6f}")
    print(f"Standard Deviation: {std_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ViT model.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config, args.load_checkpoint)