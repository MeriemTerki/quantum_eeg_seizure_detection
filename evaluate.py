# evaluate.py
"""
Evaluation script for trained QCGAN
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data.dataset import create_dataloaders
from models.qcgan import QCGAN
from utils.visualization import Visualizer
from utils.metrics import evaluate_generator, EEGMetrics


def load_model(checkpoint_path, config):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: path to checkpoint
        config: configuration
        
    Returns:
        qcgan: loaded model
        epoch: training epoch
    """
    print(f"\nLoading model from: {checkpoint_path}")
    
    qcgan = QCGAN(config)
    epoch, metrics = qcgan.load_checkpoint(checkpoint_path)
    
    print(f"  Loaded from epoch {epoch}")
    if metrics:
        print(f"  Validation metrics: {metrics}")
    
    return qcgan, epoch


def generate_samples_for_evaluation(qcgan, n_samples=1000, device='cpu'):
    """
    Generate balanced samples for evaluation
    
    Args:
        qcgan: trained QCGAN model
        n_samples: total number of samples (will be balanced)
        device: torch device
        
    Returns:
        generated samples and labels
    """
    print(f"\nGenerating {n_samples} samples for evaluation...")
    
    qcgan.generator.eval()
    
    n_per_class = n_samples // 2
    
    all_samples = []
    all_labels = []
    
    # Generate normal samples
    print("  Generating normal samples...")
    with torch.no_grad():
        noise = torch.randn(n_per_class, qcgan.config.NUM_QUBITS, device=device)
        labels = torch.zeros(n_per_class, dtype=torch.long, device=device)
        samples = qcgan.generator(noise, labels).cpu()
        
        all_samples.append(samples)
        all_labels.append(labels.cpu())
    
    # Generate seizure samples
    print("  Generating seizure samples...")
    with torch.no_grad():
        noise = torch.randn(n_per_class, qcgan.config.NUM_QUBITS, device=device)
        labels = torch.ones(n_per_class, dtype=torch.long, device=device)
        samples = qcgan.generator(noise, labels).cpu()
        
        all_samples.append(samples)
        all_labels.append(labels.cpu())
    
    # Concatenate
    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"  Generated {len(all_samples)} samples")
    print(f"    Normal: {torch.sum(all_labels == 0).item()}")
    print(f"    Seizure: {torch.sum(all_labels == 1).item()}")
    
    return all_samples, all_labels


def evaluate_discriminator_performance(qcgan, test_loader, device):
    """
    Evaluate discriminator's classification performance
    
    Args:
        qcgan: trained QCGAN model
        test_loader: test dataloader
        device: torch device
        
    Returns:
        metrics dict
    """
    print("\n" + "=" * 70)
    print("Evaluating Discriminator Performance")
    print("=" * 70)
    
    qcgan.discriminator.eval()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for eeg, labels in test_loader:
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            # Get discriminator output
            output = qcgan.discriminator(eeg, labels)
            
            # Get predictions
            predictions = (output > 0.5).long().squeeze()
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probs.append(output.cpu().numpy())
    
    # Concatenate
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    metrics = EEGMetrics.calculate_classification_metrics(all_labels, all_predictions)
    
    print("\nClassification Metrics:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_predictions,
        target_names=['Normal', 'Seizure']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return metrics, cm, all_probs


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Seizure'],
        yticklabels=['Normal', 'Seizure']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrix saved: {save_path}")
    plt.close()


def comprehensive_evaluation(checkpoint_path, config):
    """
    Perform comprehensive evaluation
    
    Args:
        checkpoint_path: path to model checkpoint
        config: configuration
    """
    print("\n" + "=" * 70)
    print("QCGAN Comprehensive Evaluation")
    print("=" * 70)
    
    # Load model
    qcgan, epoch = load_model(checkpoint_path, config)
    
    # Create dataloaders
    _, val_loader, test_loader = create_dataloaders(config)
    
    # Visualizer
    viz = Visualizer(config.OUTPUT_DIR / 'evaluation')
    
    # ========== 1. Generator Quality Evaluation ==========
    print("\n" + "-" * 70)
    print("1. Evaluating Generator Quality")
    print("-" * 70)
    
    gen_metrics = evaluate_generator(
        qcgan.generator,
        test_loader,
        config.DEVICE,
        n_samples=200
    )
    
    # ========== 2. Discriminator Performance ==========
    print("\n" + "-" * 70)
    print("2. Evaluating Discriminator Performance")
    print("-" * 70)
    
    disc_metrics, cm, probs = evaluate_discriminator_performance(
        qcgan, test_loader, config.DEVICE
    )
    
    # Plot confusion matrix
    cm_path = config.OUTPUT_DIR / 'evaluation' / 'confusion_matrix.png'
    plot_confusion_matrix(cm, cm_path)
    
    # ========== 3. Generate Sample Visualizations ==========
    print("\n" + "-" * 70)
    print("3. Generating Visualizations")
    print("-" * 70)
    
    # Generate samples
    fake_samples, fake_labels = generate_samples_for_evaluation(
        qcgan, n_samples=100, device=config.DEVICE
    )
    
    # Get real samples
    real_samples = []
    real_labels = []
    for eeg, labels in test_loader:
        real_samples.append(eeg)
        real_labels.append(labels)
        if len(real_samples) * eeg.shape[0] >= 100:
            break
    
    real_samples = torch.cat(real_samples, dim=0)[:100]
    real_labels = torch.cat(real_labels, dim=0)[:100]
    
    # Plot sample grid
    print("  Plotting generated samples...")
    viz.plot_sample_grid(
        fake_samples[:8], 
        fake_labels[:8],
        save_name='generated_samples_grid.png'
    )
    
    # Plot comparisons
    print("  Plotting comparisons...")
    normal_real = real_samples[real_labels == 0][:1]
    normal_fake = fake_samples[fake_labels == 0][:1]
    
    if len(normal_real) > 0 and len(normal_fake) > 0:
        viz.plot_comparison(
            normal_real, normal_fake,
            save_name='final_comparison_normal.png'
        )
        viz.plot_psd_comparison(
            normal_real[0], normal_fake[0],
            save_name='final_psd_normal.png'
        )
    
    seizure_real = real_samples[real_labels == 1][:1]
    seizure_fake = fake_samples[fake_labels == 1][:1]
    
    if len(seizure_real) > 0 and len(seizure_fake) > 0:
        viz.plot_comparison(
            seizure_real, seizure_fake,
            save_name='final_comparison_seizure.png'
        )
        viz.plot_psd_comparison(
            seizure_real[0], seizure_fake[0],
            save_name='final_psd_seizure.png'
        )
    
    # ========== 4. Save Generated Samples ==========
    print("\n" + "-" * 70)
    print("4. Saving Generated Samples")
    print("-" * 70)
    
    samples_dir = config.OUTPUT_DIR / 'generated_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Save by type
    normal_samples = fake_samples[fake_labels == 0].numpy()
    seizure_samples = fake_samples[fake_labels == 1].numpy()
    
    np.save(samples_dir / 'generated_normal.npy', normal_samples)
    np.save(samples_dir / 'generated_seizure.npy', seizure_samples)
    
    print(f"  Saved normal samples: {samples_dir / 'generated_normal.npy'}")
    print(f"  Saved seizure samples: {samples_dir / 'generated_seizure.npy'}")
    
    # ========== 5. Summary Report ==========
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    
    print("\nGenerator Quality Metrics:")
    print(f"  Wasserstein Distance: {gen_metrics['wasserstein_distance']:.4f}")
    print(f"  PSD MAE: {gen_metrics['psd_mae']:.4f}")
    
    print("\nDiscriminator Classification Metrics:")
    for key, value in disc_metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save summary
    summary = {
        'epoch': epoch,
        'generator_metrics': gen_metrics,
        'discriminator_metrics': disc_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    summary_path = config.OUTPUT_DIR / 'evaluation' / 'evaluation_summary.npy'
    np.save(summary_path, summary)
    print(f"\nEvaluation summary saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Evaluate trained QCGAN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    
    args = parser.parse_args()
    
    # Configuration
    Config.set_seed()
    Config.print_config()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python train.py")
        return
    
    # Evaluate
    comprehensive_evaluation(checkpoint_path, Config)


if __name__ == "__main__":
    main()