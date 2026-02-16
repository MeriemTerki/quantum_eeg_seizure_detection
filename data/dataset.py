# data/dataset.py
"""
PyTorch dataset and dataloader utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

from config import Config


class EEGSeizureDataset(Dataset):
    """PyTorch dataset for EEG seizure data"""
    
    def __init__(self, segments: np.ndarray, labels: np.ndarray, 
                 transform=None):
        """
        Args:
            segments: (n_samples, n_channels, segment_length)
            labels: (n_samples,) - 0 for seizure, -1 for normal
            transform: optional transform
        """
        self.segments = segments
        self.labels = labels
        self.transform = transform
        
        # Convert labels: -1 (normal) -> 0, 0 (seizure) -> 1
        self.binary_labels = (labels == 0).astype(np.int64)
        
        # Print dataset info
        n_seizure = np.sum(self.binary_labels == 1)
        n_normal = np.sum(self.binary_labels == 0)
        
        print(f"  Dataset size: {len(self.segments)}")
        print(f"  Seizure: {n_seizure} ({n_seizure/len(self.segments)*100:.1f}%)")
        print(f"  Normal: {n_normal} ({n_normal/len(self.segments)*100:.1f}%)")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]  # (n_channels, segment_length)
        label = self.binary_labels[idx]
        
        # Convert to tensors
        segment = torch.FloatTensor(segment)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            segment = self.transform(segment)
        
        return segment, label
    
    def get_sample_weights(self):
        """Get weights for balanced sampling"""
        # Count classes
        unique, counts = np.unique(self.binary_labels, return_counts=True)
        class_weights = 1.0 / counts
        
        # Assign weight to each sample
        sample_weights = np.array([class_weights[label] for label in self.binary_labels])
        
        return sample_weights


def create_dataloaders(config: Config = Config):
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "=" * 70)
    print("Creating Dataloaders")
    print("=" * 70)
    
    # Load preprocessed data
    segments_path = config.PROCESSED_DATA_DIR / 'segments.npy'
    labels_path = config.PROCESSED_DATA_DIR / 'labels.npy'
    
    if not segments_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found! Please run: python preprocess_data.py"
        )
    
    print(f"\nLoading data from {config.PROCESSED_DATA_DIR}")
    segments = np.load(segments_path)
    labels = np.load(labels_path)
    
    print(f"  Loaded {len(segments)} segments")
    print(f"  Shape: {segments.shape}")
    
    # Convert labels for stratification (-1, 0) -> (0, 1)
    binary_labels = (labels == 0).astype(int)
    
    # Split data: train+val vs test
    print(f"\nSplitting data:")
    print(f"  Train: {config.TRAIN_RATIO*100:.0f}%")
    print(f"  Val: {config.VAL_RATIO*100:.0f}%")
    print(f"  Test: {config.TEST_RATIO*100:.0f}%")
    
    train_val_segments, test_segments, train_val_labels, test_labels = train_test_split(
        segments, labels,
        test_size=config.TEST_RATIO,
        stratify=binary_labels,
        random_state=config.RANDOM_SEED
    )
    
    # Split train into train and val
    binary_train_val = (train_val_labels == 0).astype(int)
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    
    train_segments, val_segments, train_labels, val_labels = train_test_split(
        train_val_segments, train_val_labels,
        test_size=val_ratio_adjusted,
        stratify=binary_train_val,
        random_state=config.RANDOM_SEED
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_segments)}")
    print(f"  Val: {len(val_segments)}")
    print(f"  Test: {len(test_segments)}")
    
    # Create datasets
    print(f"\nCreating datasets:")
    print("Train dataset:")
    train_dataset = EEGSeizureDataset(train_segments, train_labels)
    
    print("Validation dataset:")
    val_dataset = EEGSeizureDataset(val_segments, val_labels)
    
    print("Test dataset:")
    test_dataset = EEGSeizureDataset(test_segments, test_labels)
    
    # Create weighted sampler for balanced training
    if config.USE_WEIGHTED_SAMPLING:
        print("\nUsing weighted sampling for balanced training")
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("=" * 70)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataloader creation
    Config.set_seed()
    
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Test batch
    print("\nTesting batch loading:")
    for segments, labels in train_loader:
        print(f"  Batch segments shape: {segments.shape}")
        print(f"  Batch labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break