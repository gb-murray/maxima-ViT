# Performs k-fold cross validation 

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from sklearn.model_selection import KFold

# Project Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.loss import Loss
from src.utils import create_model, train_one_epoch, validate
from src.data_pipeline import HDF5Dataset

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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config, args.load_checkpoint)