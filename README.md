# Quantum Conditional GAN for EEG Seizure Detection

A quantum-classical hybrid deep learning system for generating synthetic EEG seizure data using Quantum Conditional Generative Adversarial Networks (QCGAN).

## Overview

This project implements a novel approach to address the scarcity and class imbalance of epileptic seizure EEG data by using quantum computing techniques. The system uses:

- **Quantum Generator**: Patch-based quantum circuit to generate synthetic EEG signals
- **Quantum Discriminator**: Quantum-enhanced classifier to distinguish real from fake signals
- **Conditional Generation**: Control seizure type generation (normal vs seizure)

## Features

- ✅ Quantum circuit implementation using PennyLane
- ✅ Patch method for efficient quantum resource usage
- ✅ Support for CHB-MIT and TUSZ datasets
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools for EEG signals
- ✅ Easy-to-use training and evaluation scripts

## Project Structure
```
quantum_eeg_seizure_detection/
├── config.py                 # Configuration settings
├── download_data.py          # Data download script
├── preprocess_data.py        # Preprocessing pipeline
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── demo.py                   # Quick demo
├── data/                     # Data processing modules
│   ├── preprocessing.py
│   └── dataset.py
├── models/                   # Model implementations
│   ├── quantum_generator.py
│   ├── quantum_discriminator.py
│   └── qcgan.py
├── utils/                    # Utility functions
│   ├── quantum_utils.py
│   ├── metrics.py
│   ├── visualization.py
│   └── training_utils.py
└── outputs/                  # Generated outputs
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/quantum_eeg_seizure_detection.git
cd quantum_eeg_seizure_detection
```

### 2. Create virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "import pennylane as qml; import torch; import mne; print('✓ All packages installed successfully!')"
```

## Quick Start

### Step 1: Download Data
```bash
python download_data.py
```

Choose number of patients to download (3-5 recommended for testing).

### Step 2: Preprocess Data
```bash
python preprocess_data.py
```

This will:
- Load EDF files
- Apply filters (bandpass, notch)
- Resample to 256 Hz
- Normalize signals
- Segment into 10-second windows
- Save processed data

### Step 3: Run Demo
```bash
python demo.py
```

This tests all components without full training.

### Step 4: Train Model
```bash
# Quick test (5 epochs)
python train.py --epochs 5 --batch_size 8

# Full training
python train.py --epochs 100 --batch_size 32
```

Training outputs:
- Model checkpoints in `models/checkpoints/`
- Training logs in `logs/`
- Visualizations in `outputs/`

### Step 5: Evaluate Model
```bash
python evaluate.py --checkpoint models/checkpoints/best_model.pth
```

## Usage

### Training with Custom Parameters
```bash
python train.py \
    --epochs 100 \
    --batch_size 32 \
    --lr_g 0.001 \
    --lr_d 0.0001
```

### Generate Samples
```python
from models.qcgan import QCGAN
from config import Config
import torch

# Load trained model
qcgan = QCGAN(Config)
qcgan.load_checkpoint('models/checkpoints/best_model.pth')

# Generate samples
normal_samples = qcgan.generate(n_samples=10, condition=0)
seizure_samples = qcgan.generate(n_samples=10, condition=1)
```

### Evaluate Generated Samples
```python
from utils.metrics import EEGMetrics
import numpy as np

# Load real and fake samples
real_eeg = np.load('data/processed/segments.npy')
fake_eeg = np.load('outputs/generated_samples/generated_seizure.npy')

# Calculate Wasserstein distance
wd = EEGMetrics.wasserstein_distance_metric(real_eeg[:100], fake_eeg[:100])
print(f"Wasserstein Distance: {wd:.4f}")

# Compare PSD
psd_mae = EEGMetrics.compare_psd(real_eeg[0], fake_eeg[0])
print(f"PSD MAE: {psd_mae:.4f}")
```

## Configuration

Edit `config.py` to customize:
```python
# Quantum parameters
NUM_QUBITS = 8
NUM_PATCHES = 4
NUM_LAYERS = 2

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE_G = 0.001
LEARNING_RATE_D = 0.0001

# EEG parameters
SAMPLING_RATE = 256
SEGMENT_DURATION = 10
NUM_CHANNELS = 19
```

## Model Architecture

### Quantum Generator
- **Input**: Random noise (latent space) + condition (seizure type)
- **Quantum Circuit**: Patch-based parameterized quantum circuits
- **Output**: Synthetic EEG signal (19 channels × 2560 samples)

### Quantum Discriminator
- **Input**: EEG signal + condition
- **Classical Pre-processing**: Dimensionality reduction
- **Quantum Circuit**: Conditional quantum classifier
- **Output**: Probability (real vs fake)

## Datasets

### CHB-MIT (Default)
- **Source**: PhysioNet
- **Patients**: 24 pediatric patients
- **Sampling Rate**: 256 Hz
- **Channels**: 23 EEG channels
- **Download**: Automatic via `download_data.py`

### TUSZ (Optional)
- **Source**: Temple University
- **Patients**: 6,000+ patients
- **Size**: ~50GB
- **Access**: Requires registration at isip.piconepress.com

## Troubleshooting

### Issue: CUDA out of memory
```bash
# Reduce batch size
python train.py --batch_size 8
```

### Issue: Quantum circuit too slow
```python
# In config.py, change:
QUANTUM_DEVICE = "lightning.qubit"  # Faster simulator
NUM_PATCHES = 8  # More patches = less qubits per patch
```

### Issue: Data preprocessing fails
```bash
# Install missing dependency
pip install mne pyedflib

# Check MNE installation
python -c "import mne; print(mne.__version__)"
```

## Results

Expected metrics after training:
- **Wasserstein Distance**: < 0.5 (lower is better)
- **PSD MAE**: < 0.1 (lower is better)
- **Discriminator Accuracy**: 70-85%
- **Generator Loss**: Converges to ~0.6-0.8

## Citation

If you use this code in your research, please cite:
```bibtex
@software{qcgan_eeg_2024,
  title={Quantum Conditional GAN for EEG Seizure Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/quantum_eeg_seizure_detection}
}
```

## References

1. **Qu et al. (2023)**: Quantum conditional generative adversarial network based on patch method for abnormal electrocardiogram generation
2. **CHB-MIT Database**: https://physionet.org/content/chbmit/1.0.0/
3. **PennyLane**: https://pennylane.ai/

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues:
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourusername/quantum_eeg_seizure_detection/issues

## Acknowledgments

- Temple University for TUSZ dataset
- PhysioNet for CHB-MIT dataset
- PennyLane team for quantum ML framework