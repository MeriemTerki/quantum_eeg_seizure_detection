# models/qcgan.py
"""
Complete QCGAN model combining generator and discriminator
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config
from models.quantum_generator import QuantumGenerator
from models.quantum_discriminator import QuantumDiscriminator


class QCGAN:
    """
    Quantum Conditional GAN for EEG generation
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.device = config.DEVICE
        
        # Initialize models
        print("Initializing QCGAN...")
        print(f"  Device: {self.device}")
        
        self.generator = QuantumGenerator(config).to(self.device)
        self.discriminator = QuantumDiscriminator(config).to(self.device)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(config.BETA1, config.BETA2)
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(config.BETA1, config.BETA2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
    
    def train_discriminator(self, real_eeg, real_labels, fake_eeg, fake_labels):
        """
        Train discriminator for one step
        
        Args:
            real_eeg: real EEG samples
            real_labels: labels for real samples
            fake_eeg: generated EEG samples
            fake_labels: labels for fake samples
            
        Returns:
            loss, real_accuracy, fake_accuracy
        """
        self.discriminator.zero_grad()
        
        batch_size = real_eeg.shape[0]
        
        # Labels with smoothing
        real_target = torch.ones(batch_size, 1, device=self.device) * (1 - self.config.LABEL_SMOOTHING)
        fake_target = torch.zeros(batch_size, 1, device=self.device)
        
        # Train on real
        real_output = self.discriminator(real_eeg, real_labels)
        loss_real = self.criterion(real_output, real_target)
        
        # Train on fake
        fake_output = self.discriminator(fake_eeg.detach(), fake_labels)
        loss_fake = self.criterion(fake_output, fake_target)
        
        # Combined loss
        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # Calculate accuracies
        real_acc = ((real_output > 0.5).float().mean()).item()
        fake_acc = ((fake_output < 0.5).float().mean()).item()
        
        return d_loss.item(), real_acc, fake_acc
    
    def train_generator(self, fake_eeg, fake_labels):
        """
        Train generator for one step
        
        Args:
            fake_eeg: generated EEG samples
            fake_labels: labels for fake samples
            
        Returns:
            loss
        """
        self.generator.zero_grad()
        
        batch_size = fake_eeg.shape[0]
        
        # Generator wants discriminator to think fakes are real
        real_target = torch.ones(batch_size, 1, device=self.device)
        
        # Get discriminator's opinion
        fake_output = self.discriminator(fake_eeg, fake_labels)
        
        # Calculate loss
        g_loss = self.criterion(fake_output, real_target)
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def train_step(self, real_eeg, real_labels):
        """
        Complete training step
        
        Args:
            real_eeg: (batch_size, n_channels, segment_length)
            real_labels: (batch_size,) - seizure type
            
        Returns:
            dict with losses and accuracies
        """
        batch_size = real_eeg.shape[0]
        
        # Move to device
        real_eeg = real_eeg.to(self.device)
        real_labels = real_labels.to(self.device)
        
        # ========== Train Discriminator ==========
        d_losses = []
        d_real_accs = []
        d_fake_accs = []
        
        for _ in range(self.config.DISCRIMINATOR_STEPS):
            # Generate fake samples
            noise = torch.randn(batch_size, self.config.NUM_QUBITS, device=self.device)
            fake_labels = real_labels  # Use same labels as real
            
            fake_eeg = self.generator(noise, fake_labels)
            
            # Train discriminator
            d_loss, d_real_acc, d_fake_acc = self.train_discriminator(
                real_eeg, real_labels, fake_eeg, fake_labels
            )
            
            d_losses.append(d_loss)
            d_real_accs.append(d_real_acc)
            d_fake_accs.append(d_fake_acc)
        
        # ========== Train Generator ==========
        noise = torch.randn(batch_size, self.config.NUM_QUBITS, device=self.device)
        fake_labels = real_labels
        
        fake_eeg = self.generator(noise, fake_labels)
        g_loss = self.train_generator(fake_eeg, fake_labels)
        
        # Return metrics
        return {
            'g_loss': g_loss,
            'd_loss': np.mean(d_losses),
            'd_real_acc': np.mean(d_real_accs),
            'd_fake_acc': np.mean(d_fake_accs)
        }
    
    def generate(self, n_samples, condition):
        """
        Generate EEG samples
        
        Args:
            n_samples: number of samples to generate
            condition: seizure type (0 or 1)
            
        Returns:
            generated EEG samples
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.config.NUM_QUBITS, device=self.device)
            labels = torch.LongTensor([condition] * n_samples).to(self.device)
            
            fake_eeg = self.generator(noise, labels)
        
        self.generator.train()
        
        return fake_eeg.cpu()
    
    def save_checkpoint(self, path, epoch, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.history = checkpoint['history']
        
        print(f"  Checkpoint loaded: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        
        return checkpoint['epoch'], checkpoint.get('metrics', None)


if __name__ == "__main__":
    # Test QCGAN
    print("Testing QCGAN")
    print("=" * 50)
    
    import numpy as np
    Config.set_seed()
    
    qcgan = QCGAN()
    
    # Test training step
    batch_size = 4
    real_eeg = torch.randn(batch_size, Config.NUM_CHANNELS, Config.SEGMENT_LENGTH)
    real_labels = torch.LongTensor([0, 1, 0, 1])
    
    print("\nTesting training step...")
    metrics = qcgan.train_step(real_eeg, real_labels)
    
    print(f"  G Loss: {metrics['g_loss']:.4f}")
    print(f"  D Loss: {metrics['d_loss']:.4f}")
    print(f"  D Real Acc: {metrics['d_real_acc']:.4f}")
    print(f"  D Fake Acc: {metrics['d_fake_acc']:.4f}")
    
    # Test generation
    print("\nTesting generation...")
    generated = qcgan.generate(n_samples=2, condition=1)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n✓ QCGAN working!")