# utils/training_utils.py
"""
Training utilities
"""

import torch
import numpy as np
from pathlib import Path
import time


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        """
        Args:
            patience: how many epochs to wait after last improvement
            min_delta: minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if training should stop
        
        Args:
            score: current validation metric
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def get_elapsed(self):
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed
    
    @staticmethod
    def format_time(seconds):
        """Format seconds to readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def save_generated_samples(generator, save_dir, n_samples=100, device='cpu'):
    """
    Generate and save samples
    
    Args:
        generator: trained generator
        save_dir: directory to save samples
        n_samples: number of samples to generate
        device: torch device
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    generator.eval()
    
    # Generate samples for each condition
    for condition in [0, 1]:
        condition_name = "normal" if condition == 0 else "seizure"
        
        with torch.no_grad():
            noise = torch.randn(n_samples, generator.config.NUM_QUBITS, device=device)
            labels = torch.LongTensor([condition] * n_samples).to(device)
            
            samples = generator(noise, labels).cpu().numpy()
        
        # Save
        save_path = save_dir / f'generated_{condition_name}.npy'
        np.save(save_path, samples)
        print(f"  Saved {n_samples} {condition_name} samples: {save_path}")
    
    generator.train()


def log_to_file(message, log_file):
    """Append message to log file"""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(message + '\n')


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    
    for i, loss in enumerate([1.0, 0.9, 0.85, 0.84, 0.84, 0.83, 0.83]):
        should_stop = early_stop(loss)
        print(f"Epoch {i+1}: Loss={loss:.2f}, Stop={should_stop}")
        if should_stop:
            break
    
    # Test timer
    timer = Timer()
    timer.start()
    time.sleep(1)
    elapsed = timer.stop()
    print(f"\nElapsed time: {Timer.format_time(elapsed)}")
    
    print("\n✓ Training utilities working!")