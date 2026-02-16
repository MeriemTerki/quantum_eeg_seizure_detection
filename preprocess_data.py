from pathlib import Path
from config import Config
from data.preprocessing import EEGPreprocessor
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_chb_mit_dataset():
    """Preprocess entire CHB-MIT dataset"""
    
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
    print("  1. Run: python demo.py")
    print("  2. Or train: python train.py --epochs 5 --batch_size 8")
    
    return True


if __name__ == "__main__":
    Config.set_seed()
    Config.print_config()
    preprocess_chb_mit_dataset()