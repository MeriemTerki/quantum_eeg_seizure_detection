# demo.py
"""
Quick demo script to test the complete pipeline
"""

import torch
import numpy as np
from pathlib import Path

from config import Config
from data.dataset import create_dataloaders
from models.qcgan import QCGAN
from utils.visualization import Visualizer


def quick_demo():
    """
    Quick demo of the QCGAN pipeline
    """
    print("=" * 70)
    print("QCGAN-EEG Quick Demo")
    print("=" * 70)
    
    # Set seed
    Config.set_seed()
    
    # Print config
    print("\n1. Configuration:")
    print("-" * 70)
    print(f"  Device: {Config.DEVICE}")
    print(f"  Num Channels: {Config.NUM_CHANNELS}")
    print(f"  Segment Length: {Config.SEGMENT_LENGTH}")
    print(f"  Num Qubits: {Config.NUM_QUBITS}")
    print(f"  Num Patches: {Config.NUM_PATCHES}")
    
    # Check if data exists
    segments_path = Config.PROCESSED_DATA_DIR / 'segments.npy'
    labels_path = Config.PROCESSED_DATA_DIR / 'labels.npy'
    
    if not segments_path.exists() or not labels_path.exists():
        print("\n✗ Preprocessed data not found!")
        print("Please run:")
        print("  1. python download_data.py")
        print("  2. python preprocess_data.py")
        return
    
    # Load data
    print("\n2. Loading Data:")
    print("-" * 70)
    try:
        train_loader, val_loader, test_loader = create_dataloaders(Config)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Initialize model
    print("\n3. Initializing QCGAN:")
    print("-" * 70)
    try:
        qcgan = QCGAN(Config)
        print("  ✓ Generator initialized")
        print("  ✓ Discriminator initialized")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return
    
    # Test forward pass
    print("\n4. Testing Forward Pass:")
    print("-" * 70)
    
    # Get a batch
    real_eeg, real_labels = next(iter(train_loader))
    print(f"  Real EEG shape: {real_eeg.shape}")
    print(f"  Real labels: {real_labels}")
    
    # Test generator
    print("\n  Testing Generator...")
    batch_size = 4
    noise = torch.randn(batch_size, Config.NUM_QUBITS, device=Config.DEVICE)
    condition = torch.LongTensor([0, 1, 0, 1]).to(Config.DEVICE)
    
    try:
        fake_eeg = qcgan.generator(noise, condition)
        print(f"    Generated EEG shape: {fake_eeg.shape}")
        print(f"    Generated EEG range: [{fake_eeg.min():.3f}, {fake_eeg.max():.3f}]")
        print("  ✓ Generator working!")
    except Exception as e:
        print(f"  ✗ Generator error: {e}")
        return
    
    # Test discriminator
    print("\n  Testing Discriminator...")
    try:
        disc_output = qcgan.discriminator(fake_eeg, condition)
        print(f"    Discriminator output shape: {disc_output.shape}")
        print(f"    Discriminator output: {disc_output.squeeze()}")
        print("  ✓ Discriminator working!")
    except Exception as e:
        print(f"  ✗ Discriminator error: {e}")
        return
    
    # Test training step
    print("\n5. Testing Training Step:")
    print("-" * 70)
    
    try:
        metrics = qcgan.train_step(real_eeg[:4], real_labels[:4])
        print(f"  G Loss: {metrics['g_loss']:.4f}")
        print(f"  D Loss: {metrics['d_loss']:.4f}")
        print(f"  D Real Acc: {metrics['d_real_acc']:.4f}")
        print(f"  D Fake Acc: {metrics['d_fake_acc']:.4f}")
        print("  ✓ Training step working!")
    except Exception as e:
        print(f"  ✗ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test visualization
    print("\n6. Testing Visualization:")
    print("-" * 70)
    
    viz = Visualizer(Config.OUTPUT_DIR / 'demo')
    
    try:
        # Generate samples
        fake_normal = qcgan.generate(n_samples=2, condition=0)
        fake_seizure = qcgan.generate(n_samples=2, condition=1)
        
        # Plot
        viz.plot_eeg_signal(
            fake_normal[0],
            title="Demo: Generated Normal EEG",
            save_name="demo_normal.png"
        )
        
        viz.plot_eeg_signal(
            fake_seizure[0],
            title="Demo: Generated Seizure EEG",
            save_name="demo_seizure.png"
        )
        
        # Plot comparison
        real_sample = real_eeg[0:1]
        fake_sample = fake_normal[0:1]
        
        viz.plot_comparison(
            real_sample,
            fake_sample,
            save_name="demo_comparison.png"
        )
        
        print("  ✓ Visualizations saved to:", Config.OUTPUT_DIR / 'demo')
    except Exception as e:
        print(f"  ✗ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nAll components working correctly!")
    print("\nNext steps:")
    print("  1. Train the model: python train.py")
    print("  2. Evaluate results: python evaluate.py")
    print("\nFor quick training test, run:")
    print("  python train.py --epochs 5 --batch_size 8")


if __name__ == "__main__":
    quick_demo()