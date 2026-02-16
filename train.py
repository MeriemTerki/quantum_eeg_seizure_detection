# train.py
"""
Main training script for QCGAN
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from config import Config
from data.dataset import create_dataloaders
from models.qcgan import QCGAN
from utils.visualization import Visualizer
from utils.training_utils import EarlyStopping, AverageMeter, Timer, log_to_file
from utils.metrics import evaluate_generator


def train_epoch(qcgan, train_loader, epoch, config):
    """
    Train for one epoch
    
    Args:
        qcgan: QCGAN model
        train_loader: training dataloader
        epoch: current epoch
        config: configuration
        
    Returns:
        dict with average metrics
    """
    # Meters for tracking
    g_loss_meter = AverageMeter()
    d_loss_meter = AverageMeter()
    d_real_acc_meter = AverageMeter()
    d_fake_acc_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
    
    for batch_idx, (real_eeg, real_labels) in enumerate(pbar):
        # Train step
        metrics = qcgan.train_step(real_eeg, real_labels)
        
        # Update meters
        batch_size = real_eeg.shape[0]
        g_loss_meter.update(metrics['g_loss'], batch_size)
        d_loss_meter.update(metrics['d_loss'], batch_size)
        d_real_acc_meter.update(metrics['d_real_acc'], batch_size)
        d_fake_acc_meter.update(metrics['d_fake_acc'], batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'G_Loss': f"{g_loss_meter.avg:.4f}",
            'D_Loss': f"{d_loss_meter.avg:.4f}",
            'D_Real': f"{d_real_acc_meter.avg:.3f}",
            'D_Fake': f"{d_fake_acc_meter.avg:.3f}"
        })
    
    return {
        'g_loss': g_loss_meter.avg,
        'd_loss': d_loss_meter.avg,
        'd_real_acc': d_real_acc_meter.avg,
        'd_fake_acc': d_fake_acc_meter.avg
    }


def validate(qcgan, val_loader, config):
    """
    Validate model
    
    Args:
        qcgan: QCGAN model
        val_loader: validation dataloader
        config: configuration
        
    Returns:
        validation loss
    """
    qcgan.generator.eval()
    qcgan.discriminator.eval()
    
    g_losses = []
    d_losses = []
    
    with torch.no_grad():
        for real_eeg, real_labels in val_loader:
            batch_size = real_eeg.shape[0]
            
            # Move to device
            real_eeg = real_eeg.to(config.DEVICE)
            real_labels = real_labels.to(config.DEVICE)
            
            # Generate fake samples
            noise = torch.randn(batch_size, config.NUM_QUBITS, device=config.DEVICE)
            fake_eeg = qcgan.generator(noise, real_labels)
            
            # Discriminator outputs
            real_output = qcgan.discriminator(real_eeg, real_labels)
            fake_output = qcgan.discriminator(fake_eeg, real_labels)
            
            # Calculate losses
            real_target = torch.ones_like(real_output)
            fake_target = torch.zeros_like(fake_output)
            
            d_loss_real = qcgan.criterion(real_output, real_target)
            d_loss_fake = qcgan.criterion(fake_output, fake_target)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            g_target = torch.ones_like(fake_output)
            g_loss = qcgan.criterion(fake_output, g_target)
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
    
    qcgan.generator.train()
    qcgan.discriminator.train()
    
    return {
        'g_loss': np.mean(g_losses),
        'd_loss': np.mean(d_losses)
    }


def train(config: Config = Config):
    """
    Main training function
    
    Args:
        config: configuration object
    """
    print("\n" + "=" * 70)
    print("QCGAN Training")
    print("=" * 70)
    
    # Set seed
    config.set_seed()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Initialize model
    qcgan = QCGAN(config)
    
    # Visualizer
    viz = Visualizer(config.OUTPUT_DIR)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='min'
    )
    
    # Timer
    timer = Timer()
    timer.start()
    
    # Log file
    log_file = config.LOG_DIR / 'training.log'
    
    print(f"\nStarting training...")
    print(f"  Total epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Log file: {log_file}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_metrics = train_epoch(qcgan, train_loader, epoch, config)
        
        # Validate
        val_metrics = validate(qcgan, val_loader, config)
        
        # Update history
        qcgan.history['g_loss'].append(train_metrics['g_loss'])
        qcgan.history['d_loss'].append(train_metrics['d_loss'])
        qcgan.history['d_real_acc'].append(train_metrics['d_real_acc'])
        qcgan.history['d_fake_acc'].append(train_metrics['d_fake_acc'])
        
        # Log
        log_message = (
            f"Epoch {epoch}/{config.NUM_EPOCHS} - "
            f"Train G: {train_metrics['g_loss']:.4f}, "
            f"Train D: {train_metrics['d_loss']:.4f}, "
            f"Val G: {val_metrics['g_loss']:.4f}, "
            f"Val D: {val_metrics['d_loss']:.4f}"
        )
        print(f"\n{log_message}")
        log_to_file(log_message, log_file)
        
        # Save best model
        val_loss = val_metrics['g_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = config.MODEL_DIR / 'best_model.pth'
            qcgan.save_checkpoint(checkpoint_path, epoch, val_metrics)
        
        # Save periodic checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            checkpoint_path = config.MODEL_DIR / f'checkpoint_epoch_{epoch}.pth'
            qcgan.save_checkpoint(checkpoint_path, epoch, val_metrics)
        
        # Generate and visualize samples
        if epoch % config.VISUALIZE_INTERVAL == 0:
            print(f"\nGenerating samples for epoch {epoch}...")
            
            # Generate samples
            fake_normal = qcgan.generate(n_samples=4, condition=0)
            fake_seizure = qcgan.generate(n_samples=4, condition=1)
            
            # Get real samples for comparison
            real_eeg, real_labels = next(iter(val_loader))
            real_normal = real_eeg[real_labels == 0][:1]
            real_seizure = real_eeg[real_labels == 1][:1]
            
            # Plot comparisons
            if len(real_normal) > 0:
                viz.plot_comparison(
                    real_normal, fake_normal,
                    save_name=f'comparison_normal_epoch_{epoch}.png'
                )
            
            if len(real_seizure) > 0:
                viz.plot_comparison(
                    real_seizure, fake_seizure,
                    save_name=f'comparison_seizure_epoch_{epoch}.png'
                )
            
            # Plot PSD
            if len(real_normal) > 0:
                viz.plot_psd_comparison(
                    real_normal[0], fake_normal[0],
                    save_name=f'psd_comparison_epoch_{epoch}.png'
                )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Training complete
    elapsed = timer.stop()
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Total time: {Timer.format_time(elapsed)}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved: {config.MODEL_DIR / 'best_model.pth'}")
    
    # Plot training history
    viz.plot_training_history(qcgan.history)
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    eval_metrics = evaluate_generator(
        qcgan.generator, 
        test_loader, 
        config.DEVICE,
        n_samples=100
    )
    
    # Save final results
    results = {
        'training_time': elapsed,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch,
        'evaluation_metrics': eval_metrics,
        'history': qcgan.history
    }
    
    results_path = config.OUTPUT_DIR / 'training_results.npy'
    np.save(results_path, results)
    print(f"\nResults saved: {results_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train QCGAN for EEG generation')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=None, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=None, help='Discriminator learning rate')
    
    args = parser.parse_args()
    
    # Update config if arguments provided
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr_g:
        Config.LEARNING_RATE_G = args.lr_g
    if args.lr_d:
        Config.LEARNING_RATE_D = args.lr_d
    
    # Print configuration
    Config.print_config()
    
    # Train
    train(Config)


if __name__ == "__main__":
    main()