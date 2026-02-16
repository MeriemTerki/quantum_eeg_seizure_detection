# utils/visualization.py
"""
Visualization utilities for QCGAN
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch

from config import Config


class Visualizer:
    """Visualization tools for EEG and training"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_eeg_signal(self, eeg, title="EEG Signal", save_name=None):
        """
        Plot multi-channel EEG signal
        
        Args:
            eeg: (n_channels, n_samples) or (batch, n_channels, n_samples)
            title: plot title
            save_name: filename to save
        """
        if len(eeg.shape) == 3:
            eeg = eeg[0]  # Take first sample if batch
        
        if torch.is_tensor(eeg):
            eeg = eeg.cpu().numpy()
        
        n_channels, n_samples = eeg.shape
        time = np.arange(n_samples) / Config.SAMPLING_RATE
        
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, n_channels * 1.5))
        
        if n_channels == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(time, eeg[i], linewidth=0.5)
            ax.set_ylabel(f'Ch {i+1}')
            ax.set_xlim(0, time[-1])
            
            if i < n_channels - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (s)')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_comparison(self, real_eeg, fake_eeg, save_name=None):
        """
        Plot real vs fake EEG side by side
        
        Args:
            real_eeg: real EEG sample
            fake_eeg: generated EEG sample
            save_name: filename to save
        """
        if torch.is_tensor(real_eeg):
            real_eeg = real_eeg.cpu().numpy()
        if torch.is_tensor(fake_eeg):
            fake_eeg = fake_eeg.cpu().numpy()
        
        if len(real_eeg.shape) == 3:
            real_eeg = real_eeg[0]
            fake_eeg = fake_eeg[0]
        
        n_channels = min(5, real_eeg.shape[0])  # Show first 5 channels
        time = np.arange(real_eeg.shape[1]) / Config.SAMPLING_RATE
        
        fig, axes = plt.subplots(n_channels, 2, figsize=(14, n_channels * 1.5))
        
        for i in range(n_channels):
            # Real
            axes[i, 0].plot(time, real_eeg[i], linewidth=0.5, color='blue')
            axes[i, 0].set_ylabel(f'Ch {i+1}')
            if i == 0:
                axes[i, 0].set_title('Real EEG', fontweight='bold')
            if i < n_channels - 1:
                axes[i, 0].set_xticks([])
            else:
                axes[i, 0].set_xlabel('Time (s)')
            
            # Fake
            axes[i, 1].plot(time, fake_eeg[i], linewidth=0.5, color='red')
            if i == 0:
                axes[i, 1].set_title('Generated EEG', fontweight='bold')
            if i < n_channels - 1:
                axes[i, 1].set_xticks([])
            else:
                axes[i, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training history
        
        Args:
            history: dict with losses and accuracies
            save_name: filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['g_loss']) + 1)
        
        # Generator loss
        axes[0, 0].plot(epochs, history['g_loss'], label='Generator', color='blue')
        axes[0, 0].set_title('Generator Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Discriminator loss
        axes[0, 1].plot(epochs, history['d_loss'], label='Discriminator', color='red')
        axes[0, 1].set_title('Discriminator Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Discriminator accuracy on real
        axes[1, 0].plot(epochs, history['d_real_acc'], label='Real Accuracy', color='green')
        axes[1, 0].set_title('Discriminator Accuracy (Real)', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Discriminator accuracy on fake
        axes[1, 1].plot(epochs, history['d_fake_acc'], label='Fake Accuracy', color='orange')
        axes[1, 1].set_title('Discriminator Accuracy (Fake)', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_psd_comparison(self, real_eeg, fake_eeg, fs=256, save_name=None):
        """
        Plot power spectral density comparison
        
        Args:
            real_eeg: real EEG (n_channels, n_samples)
            fake_eeg: fake EEG (n_channels, n_samples)
            fs: sampling frequency
            save_name: filename to save
        """
        from scipy import signal
        
        if torch.is_tensor(real_eeg):
            real_eeg = real_eeg.cpu().numpy()
        if torch.is_tensor(fake_eeg):
            fake_eeg = fake_eeg.cpu().numpy()
        
        if len(real_eeg.shape) == 3:
            real_eeg = real_eeg[0]
            fake_eeg = fake_eeg[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot first 4 channels
        for i in range(min(4, real_eeg.shape[0])):
            ax = axes[i // 2, i % 2]
            
            # Calculate PSD
            freqs_real, psd_real = signal.welch(real_eeg[i], fs=fs, nperseg=256)
            freqs_fake, psd_fake = signal.welch(fake_eeg[i], fs=fs, nperseg=256)
            
            # Plot
            ax.semilogy(freqs_real, psd_real, label='Real', color='blue', alpha=0.7)
            ax.semilogy(freqs_fake, psd_fake, label='Generated', color='red', alpha=0.7)
            
            ax.set_title(f'Channel {i+1}', fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')
            ax.legend()
            ax.grid(True)
            ax.set_xlim([0, 50])  # Focus on 0-50 Hz
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_sample_grid(self, samples, labels, n_samples=8, save_name=None):
        """
        Plot grid of generated samples
        
        Args:
            samples: (n_samples, n_channels, segment_length)
            labels: (n_samples,)
            n_samples: number of samples to plot
            save_name: filename to save
        """
        if torch.is_tensor(samples):
            samples = samples.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        n_samples = min(n_samples, len(samples))
        n_cols = 2
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        time = np.arange(samples.shape[2]) / Config.SAMPLING_RATE
        
        for i in range(n_samples):
            ax = axes[i]
            
            # Plot first channel only for grid
            ax.plot(time, samples[i, 0], linewidth=0.5)
            
            label_name = "Seizure" if labels[i] == 1 else "Normal"
            ax.set_title(f'Sample {i+1} ({label_name})', fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization...")
    
    viz = Visualizer()
    
    # Create dummy data
    dummy_eeg = np.random.randn(19, 2560)
    
    # Test plotting
    viz.plot_eeg_signal(dummy_eeg, title="Test EEG", save_name="test_eeg.png")
    
    print("✓ Visualization working!")