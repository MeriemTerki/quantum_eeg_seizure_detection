# data/preprocessing.py
"""
Preprocessing module for CHB-MIT EEG data
"""

import numpy as np
import mne
from scipy import signal
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config


class EEGPreprocessor:
    """Preprocess CHB-MIT EEG data"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
    def load_edf_file(self, edf_path: Path):
        """
        Load EDF file using MNE
        
        Returns:
            raw: MNE Raw object
        """
        try:
            raw = mne.io.read_raw_edf(
                edf_path, 
                preload=True, 
                verbose=False
            )
            return raw
        except Exception as e:
            print(f"Error loading {edf_path}: {e}")
            return None
    
    def select_channels(self, raw):
        """
        Select EEG channels (exclude non-EEG channels)
        
        Args:
            raw: MNE Raw object
            
        Returns:
            raw: MNE Raw object with selected channels
        """
        # Get all channel names
        all_channels = raw.ch_names
        
        # Exclude non-EEG channels
        exclude = ['T9-P7', 'P7-O1', 'VNS', 'T9', 'T10']
        eeg_channels = [ch for ch in all_channels 
                       if not any(ex in ch for ex in exclude)]
        
        # Select first N channels
        selected_channels = eeg_channels[:self.config.NUM_CHANNELS]
        
        # Pick channels
        raw.pick_channels(selected_channels, ordered=True)
        
        return raw
    
    def apply_filters(self, raw):
        """
        Apply bandpass and notch filters
        
        Args:
            raw: MNE Raw object
            
        Returns:
            raw: filtered MNE Raw object
        """
        # Apply bandpass filter
        raw.filter(
            self.config.BANDPASS_LOW,
            self.config.BANDPASS_HIGH,
            fir_design='firwin',
            verbose=False
        )
        
        # Apply notch filter
        raw.notch_filter(
            self.config.NOTCH_FREQ,
            fir_design='firwin',
            verbose=False
        )
        
        return raw
    
    def resample_data(self, raw):
        """
        Resample to target sampling rate
        
        Args:
            raw: MNE Raw object
            
        Returns:
            raw: resampled MNE Raw object
        """
        current_sfreq = raw.info['sfreq']
        
        if current_sfreq != self.config.SAMPLING_RATE:
            raw.resample(self.config.SAMPLING_RATE, verbose=False)
        
        return raw
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize to zero mean, unit variance per channel
        
        Args:
            data: (n_channels, n_samples)
            
        Returns:
            normalized data
        """
        normalized = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            mean = np.mean(data[i])
            std = np.std(data[i])
            
            if std > 0:
                normalized[i] = (data[i] - mean) / std
            else:
                normalized[i] = data[i] - mean
        
        return normalized
    
    def parse_summary(self, summary_path: Path) -> Dict:
        """
        Parse CHB-MIT summary file
        
        Returns:
            dict: {filename: [(start, end), ...]}
        """
        seizure_info = {}
        
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        num_seizures = 0
        seizure_times = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('File Name:'):
                # Save previous file info
                if current_file:
                    seizure_info[current_file] = seizure_times
                
                # Start new file
                current_file = line.split(':')[1].strip()
                seizure_times = []
                
            elif line.startswith('Number of Seizures in File:'):
                num_seizures = int(line.split(':')[1].strip())
                
            elif line.startswith('Seizure Start Time:'):
                start_time = int(line.split(':')[1].strip().split()[0])
                
            elif line.startswith('Seizure End Time:'):
                end_time = int(line.split(':')[1].strip().split()[0])
                seizure_times.append((start_time, end_time))
        
        # Save last file
        if current_file:
            seizure_info[current_file] = seizure_times
        
        return seizure_info
    
    def segment_data(self, data: np.ndarray, 
                    sample_rate: int,
                    seizure_times: List[Tuple] = None) -> Tuple:
        """
        Segment data into fixed windows
        
        Args:
            data: (n_channels, n_samples)
            sample_rate: sampling rate
            seizure_times: list of (start, end) tuples in seconds
            
        Returns:
            segments: (n_segments, n_channels, segment_length)
            labels: (n_segments,)
        """
        segment_samples = self.config.SEGMENT_LENGTH
        n_samples = data.shape[1]
        n_segments = n_samples // segment_samples
        
        segments = []
        labels = []
        
        for i in range(n_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            
            segment = data[:, start_idx:end_idx]
            
            # Determine label (seizure or not)
            start_time = start_idx / sample_rate
            end_time = end_idx / sample_rate
            
            is_seizure = False
            
            if seizure_times:
                for sz_start, sz_end in seizure_times:
                    # Calculate overlap
                    overlap_start = max(start_time, sz_start)
                    overlap_end = min(end_time, sz_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    # If >50% overlap, label as seizure
                    if overlap / (end_time - start_time) >= 0.5:
                        is_seizure = True
                        break
            
            # Label: 0 for seizure, -1 for non-seizure
            label = 0 if is_seizure else -1
            
            segments.append(segment)
            labels.append(label)
        
        return np.array(segments), np.array(labels)
    
    def preprocess_file(self, edf_path: Path, 
                       seizure_times: List[Tuple] = None) -> Dict:
        """
        Complete preprocessing pipeline for one file
        
        Returns:
            dict with segments and labels
        """
        # Load
        raw = self.load_edf_file(edf_path)
        if raw is None:
            return None
        
        # Select channels
        raw = self.select_channels(raw)
        
        # Filter
        raw = self.apply_filters(raw)
        
        # Resample
        raw = self.resample_data(raw)
        
        # Get data array
        data = raw.get_data()  # (n_channels, n_samples)
        
        # Normalize
        data = self.normalize_data(data)
        
        # Segment
        segments, labels = self.segment_data(
            data, 
            self.config.SAMPLING_RATE, 
            seizure_times
        )
        
        return {
            'segments': segments,
            'labels': labels,
            'file_path': str(edf_path)
        }
    
    def preprocess_patient(self, patient_dir: Path) -> Tuple:
        """
        Preprocess all files for one patient
        
        Returns:
            segments, labels
        """
        # Parse summary
        summary_file = list(patient_dir.glob("*-summary.txt"))[0]
        seizure_info = self.parse_summary(summary_file)
        
        all_segments = []
        all_labels = []
        
        # Process each EDF file
        edf_files = sorted(patient_dir.glob("*.edf"))
        
        for edf_file in tqdm(edf_files, desc=f"  {patient_dir.name}", leave=False):
            # Get seizure times for this file
            seizure_times = seizure_info.get(edf_file.name, [])
            
            # Preprocess
            result = self.preprocess_file(edf_file, seizure_times)
            
            if result is not None:
                all_segments.append(result['segments'])
                all_labels.append(result['labels'])
        
        # Concatenate
        if all_segments:
            all_segments = np.concatenate(all_segments, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            return all_segments, all_labels
        else:
            return None, None


def preprocess_dataset():
    """
    Preprocess entire CHB-MIT dataset
    """
    print("=" * 70)
    print("Preprocessing CHB-MIT Dataset")
    print("=" * 70)
    
    preprocessor = EEGPreprocessor()
    
    # Find patient directories
    patient_dirs = sorted([
        d for d in Config.RAW_DATA_DIR.iterdir() 
        if d.is_dir() and d.name.startswith('chb')
    ])
    
    if not patient_dirs:
        print("\n✗ No patient data found!")
        print("Please run: python download_data.py")
        return False
    
    print(f"\nFound {len(patient_dirs)} patients")
    print(f"Output directory: {Config.PROCESSED_DATA_DIR}")
    
    all_segments = []
    all_labels = []
    patient_info = []
    
    # Process each patient
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        try:
            segments, labels = preprocessor.preprocess_patient(patient_dir)
            
            if segments is not None:
                all_segments.append(segments)
                all_labels.append(labels)
                
                # Store info
                patient_info.append({
                    'patient_id': patient_dir.name,
                    'n_segments': len(segments),
                    'n_seizure': np.sum(labels == 0),
                    'n_normal': np.sum(labels == -1)
                })
                
        except Exception as e:
            print(f"\n  Error processing {patient_dir.name}: {e}")
            continue
    
    # Concatenate all
    if not all_segments:
        print("\n✗ No data processed successfully!")
        return False
    
    all_segments = np.concatenate(all_segments, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Preprocessing Complete")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Segment shape: {all_segments[0].shape}")
    print(f"  Seizure segments: {np.sum(all_labels == 0)} ({np.sum(all_labels == 0)/len(all_labels)*100:.1f}%)")
    print(f"  Normal segments: {np.sum(all_labels == -1)} ({np.sum(all_labels == -1)/len(all_labels)*100:.1f}%)")
    
    print(f"\nPer-patient breakdown:")
    df = pd.DataFrame(patient_info)
    print(df.to_string(index=False))
    
    # Save processed data
    output_dir = Config.PROCESSED_DATA_DIR
    
    print(f"\nSaving to {output_dir}...")
    np.save(output_dir / 'segments.npy', all_segments)
    np.save(output_dir / 'labels.npy', all_labels)
    df.to_csv(output_dir / 'patient_info.csv', index=False)
    
    print("\n✓ Preprocessing complete!")
    print("\nNext steps:")
    print("  1. Run: python train.py")
    print("  2. Or explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    
    return True


if __name__ == "__main__":
    Config.print_config()
    preprocess_dataset()