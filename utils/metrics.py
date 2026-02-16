"""
Evaluation metrics for QCGAN EEG generation.
"""

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EEGMetrics:
    """Collection of EEG-related evaluation metrics."""

    @staticmethod
    def _to_numpy(x):
        """Convert torch tensor / list-like input to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def wasserstein_distance_metric(real_eeg, fake_eeg):
        """
        Compute average 1D Wasserstein distance between real and fake EEG samples.

        Args:
            real_eeg: array-like, shape (n_samples, ...)
            fake_eeg: array-like, shape (n_samples, ...)

        Returns:
            float: mean Wasserstein distance across paired samples
        """
        real = EEGMetrics._to_numpy(real_eeg)
        fake = EEGMetrics._to_numpy(fake_eeg)

        n = min(len(real), len(fake))
        if n == 0:
            return float("nan")

        distances = []
        for i in range(n):
            real_flat = np.ravel(real[i])
            fake_flat = np.ravel(fake[i])
            distances.append(wasserstein_distance(real_flat, fake_flat))

        return float(np.mean(distances))

    @staticmethod
    def compare_psd(real_sample, fake_sample, fs=256):
        """
        Compare PSD of one real and one fake sample using MAE.

        Args:
            real_sample: array-like, shape (channels, time) or (time,)
            fake_sample: array-like, shape (channels, time) or (time,)
            fs: sampling rate in Hz

        Returns:
            float: mean absolute error between normalized PSDs
        """
        real = EEGMetrics._to_numpy(real_sample)
        fake = EEGMetrics._to_numpy(fake_sample)

        if real.ndim == 1:
            real = real[None, :]
        if fake.ndim == 1:
            fake = fake[None, :]

        n_channels = min(real.shape[0], fake.shape[0])
        if n_channels == 0:
            return float("nan")

        psd_errors = []
        for ch in range(n_channels):
            _, psd_real = welch(real[ch], fs=fs, nperseg=min(256, real.shape[-1]))
            _, psd_fake = welch(fake[ch], fs=fs, nperseg=min(256, fake.shape[-1]))

            # Normalize for robust shape comparison
            psd_real = psd_real / (np.sum(psd_real) + 1e-12)
            psd_fake = psd_fake / (np.sum(psd_fake) + 1e-12)

            psd_errors.append(np.mean(np.abs(psd_real - psd_fake)))

        return float(np.mean(psd_errors))

    @staticmethod
    def calculate_classification_metrics(y_true, y_pred):
        """
        Calculate standard binary classification metrics.

        Args:
            y_true: true labels
            y_pred: predicted labels

        Returns:
            dict with accuracy, precision, recall, f1_score
        """
        y_true = EEGMetrics._to_numpy(y_true).astype(int)
        y_pred = EEGMetrics._to_numpy(y_pred).astype(int)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }


def evaluate_generator(generator, test_loader, device, n_samples=100):
    """
    Evaluate generator quality against real test samples.

    Args:
        generator: trained generator model
        test_loader: dataloader yielding (eeg, labels)
        device: torch device
        n_samples: maximum number of real/fake samples to compare

    Returns:
        dict with generator quality metrics
    """
    was_training = generator.training
    generator.eval()

    real_batches = []
    label_batches = []
    collected = 0

    with torch.no_grad():
        for real_eeg, labels in test_loader:
            real_batches.append(real_eeg)
            label_batches.append(labels)
            collected += real_eeg.shape[0]
            if collected >= n_samples:
                break

    if not real_batches:
        if was_training:
            generator.train()
        return {
            "wasserstein_distance": float("nan"),
            "psd_mae": float("nan"),
        }

    real_eeg = torch.cat(real_batches, dim=0)[:n_samples].to(device)
    labels = torch.cat(label_batches, dim=0)[:n_samples].to(device)

    noise_dim = getattr(getattr(generator, "config", None), "NUM_QUBITS", real_eeg.shape[-1])
    noise = torch.randn(real_eeg.shape[0], noise_dim, device=device)

    with torch.no_grad():
        fake_eeg = generator(noise, labels)

    wd = EEGMetrics.wasserstein_distance_metric(real_eeg, fake_eeg)

    n = min(real_eeg.shape[0], fake_eeg.shape[0])
    psd_scores = [
        EEGMetrics.compare_psd(real_eeg[i], fake_eeg[i], fs=256)
        for i in range(n)
    ]
    psd_mae = float(np.mean(psd_scores)) if psd_scores else float("nan")

    if was_training:
        generator.train()

    return {
        "wasserstein_distance": wd,
        "psd_mae": psd_mae,
    }
