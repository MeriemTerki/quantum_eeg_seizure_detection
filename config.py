# config.py
import os
from pathlib import Path
import torch

class Config:
    """Configuration for QCGAN-EEG model"""
    
    # ===================== PATHS =====================
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Create directories
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ===================== DATASET =====================
    USE_CHB_MIT = True  # Using CHB-MIT dataset
    DATASET_NAME = "CHB-MIT" if USE_CHB_MIT else "TUSZ"
    
    # ===================== EEG PARAMETERS =====================
    # CHB-MIT channels (will select first 19)
    NUM_CHANNELS = 19
    SAMPLING_RATE = 256  # Hz (resampled)
    SEGMENT_DURATION = 10  # seconds
    SEGMENT_LENGTH = SAMPLING_RATE * SEGMENT_DURATION  # 2560 samples
    
    # Filtering
    BANDPASS_LOW = 0.5  # Hz
    BANDPASS_HIGH = 75  # Hz
    NOTCH_FREQ = 60  # Hz (US data is 60Hz, EU is 50Hz)
    
    # ===================== SEIZURE TYPES =====================
    if USE_CHB_MIT:
        # Simplified for CHB-MIT
        SEIZURE_TYPES = {
            'SEIZURE': 0,  # Generic seizure
            'NORMAL': -1,  # Non-seizure
        }
        NUM_SEIZURE_TYPES = 1
        SELECTED_SEIZURE_TYPES = ['SEIZURE']
        NUM_SELECTED_TYPES = 1
    else:
        # Full TUSZ classification
        SEIZURE_TYPES = {
            'FNSZ': 0,  # Focal Non-Specific Seizure
            'GNSZ': 1,  # Generalized Non-Specific Seizure
            'CPSZ': 2,  # Complex Partial Seizure
            'TCSZ': 3,  # Tonic-Clonic Seizure
        }
        NUM_SEIZURE_TYPES = 4
        SELECTED_SEIZURE_TYPES = ['FNSZ', 'GNSZ', 'CPSZ', 'TCSZ']
        NUM_SELECTED_TYPES = 4
    
    # ===================== QUANTUM PARAMETERS =====================
    # Quantum Circuit
    NUM_QUBITS = 8  # Per sub-generator
    NUM_AUXILIARY_QUBITS = 1
    NUM_LAYERS = 2  # Entanglement layers
    
    # Patch Method - CRITICAL FOR QUANTUM EFFICIENCY
    NUM_PATCHES = 4  # Divide signal into patches
    PATCH_SIZE = SEGMENT_LENGTH // NUM_PATCHES  # 640 samples per patch
    
    # Quantum encoding size (each patch is further compressed)
    QUANTUM_PATCH_SIZE = 2 ** (NUM_QUBITS - NUM_AUXILIARY_QUBITS)  # 128
    
    # Conditional encoding (for seizure types)
    NUM_CONDITION_QUBITS = 2  # Can encode 4 conditions
    
    # Quantum device
    QUANTUM_DEVICE = "default.qubit"  # PennyLane simulator
    # QUANTUM_DEVICE = "lightning.qubit"  # Faster alternative
    
    # ===================== MODEL ARCHITECTURE =====================
    # Generator
    LATENT_DIM = 100  # Noise dimension
    
    # Discriminator
    DISC_HIDDEN_DIM = 256
    
    # ===================== TRAINING PARAMETERS =====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE_G = 0.001  # Generator
    LEARNING_RATE_D = 0.0001  # Discriminator
    BETA1 = 0.5  # Adam beta1
    BETA2 = 0.999  # Adam beta2
    
    # Adversarial training
    DISCRIMINATOR_STEPS = 5  # Train D for K steps per G step
    LABEL_SMOOTHING = 0.1  # Smooth labels to stabilize training
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Data balancing
    USE_WEIGHTED_SAMPLING = True
    OVERSAMPLE_MINORITY = True
    
    # ===================== LOGGING =====================
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save model every N epochs
    VISUALIZE_INTERVAL = 20  # Generate samples every N epochs
    
    USE_WANDB = False  # Set True for Weights & Biases
    WANDB_PROJECT = "qcgan-eeg-seizure"
    WANDB_ENTITY = None  # Your wandb username
    
    # ===================== EVALUATION =====================
    EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
    SAVE_GENERATED_SAMPLES = True
    NUM_SAMPLES_TO_GENERATE = 100
    
    # ===================== DEVICE =====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===================== RANDOM SEED =====================
    RANDOM_SEED = 42
    
    @classmethod
    def set_seed(cls):
        """Set random seed for reproducibility"""
        import random
        import numpy as np
        import torch
        
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 70)
        print(f"{'QCGAN-EEG Configuration':^70}")
        print("=" * 70)
        print(f"\n{'DATASET':^70}")
        print("-" * 70)
        print(f"  Dataset: {cls.DATASET_NAME}")
        print(f"  Raw Data: {cls.RAW_DATA_DIR}")
        print(f"  Processed Data: {cls.PROCESSED_DATA_DIR}")
        
        print(f"\n{'EEG PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Channels: {cls.NUM_CHANNELS}")
        print(f"  Sampling Rate: {cls.SAMPLING_RATE} Hz")
        print(f"  Segment: {cls.SEGMENT_DURATION}s ({cls.SEGMENT_LENGTH} samples)")
        print(f"  Bandpass: {cls.BANDPASS_LOW}-{cls.BANDPASS_HIGH} Hz")
        
        print(f"\n{'SEIZURE TYPES':^70}")
        print("-" * 70)
        print(f"  Number of types: {cls.NUM_SELECTED_TYPES}")
        print(f"  Types: {', '.join(cls.SELECTED_SEIZURE_TYPES)}")
        
        print(f"\n{'QUANTUM PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Qubits per sub-generator: {cls.NUM_QUBITS}")
        print(f"  Auxiliary qubits: {cls.NUM_AUXILIARY_QUBITS}")
        print(f"  Entanglement layers: {cls.NUM_LAYERS}")
        print(f"  Number of patches: {cls.NUM_PATCHES}")
        print(f"  Patch size: {cls.PATCH_SIZE} samples")
        print(f"  Quantum patch size: {cls.QUANTUM_PATCH_SIZE} samples")
        print(f"  Quantum device: {cls.QUANTUM_DEVICE}")
        
        print(f"\n{'TRAINING PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning rates: G={cls.LEARNING_RATE_G}, D={cls.LEARNING_RATE_D}")
        print(f"  Discriminator steps: {cls.DISCRIMINATOR_STEPS}")
        print(f"  Device: {cls.DEVICE}")
        
        print("=" * 70)


if __name__ == "__main__":
    Config.set_seed()
    Config.print_config()