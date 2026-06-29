# Trains a ViT regression model from a given config.yaml

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.amp.grad_scaler import GradScaler

from loss import Loss
from utils import create_model, load_model, freeze_backbone, train_one_epoch, validate
from data_pipeline import DiffractionDataset

def main(config: dict, checkpoint_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if checkpoint_path:
        model = load_model(checkpoint_path, config).to(device)
    else:
        model = create_model(config).to(device)

    log_dir = os.path.join(config['paths']['output_dir'], 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    if config['training'].get('freeze_backbone', False):
        freeze_backbone(model) #type: ignore

    image_size=config['model'].get('image_size', 1056)
    train_group = config['data'].get('train_group', 'train')
    test_group = config['data'].get('test_group', 'test')
    
    train_dataset = DiffractionDataset(
        config['data']['hdf5_path'],
        train_group,
        image_size=image_size,
    )
    test_dataset = DiffractionDataset(
        config['data']['hdf5_path'],
        test_group,
        image_size=image_size,
    )

    num_workers = os.cpu_count() // 2 #type: ignore
    print(f"Using {num_workers} subprocesses.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    centers = train_dataset.centers
    scale_factors = train_dataset.scale_factors
    weights = config['training'].get('loss_weights', [1.0, 1.0, 1.0, 1.0, 1.0, 0.1])

    loss_fn = Loss(centers=centers, scale_factors=scale_factors, weights=weights).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    scaler = GradScaler(device=device.type) 

    patience = config['training'].get('patience', float('inf'))
    num_overfit_epochs = 0

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, writer, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)

        test_loss = validate(model, test_loader, loss_fn, device, writer, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            num_overfit_epochs = 0
            model_path = os.path.join(output_dir, config['model']['name'])
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            num_overfit_epochs += 1

        if num_overfit_epochs > patience:
            print(f'Patience has been exceeded. Training has been terminated at Epoch {epoch}.')
            break

    writer.close()
    print("\nTraining complete.")

if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train a ViT model.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    parser.add_argument("--load-checkpoint", help="Path to a model checkpoint to continue training.", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config, args.load_checkpoint)