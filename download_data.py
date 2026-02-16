# download_data.py
"""
Download CHB-MIT EEG seizure dataset
"""

import os
import urllib.request
from pathlib import Path
import requests
from tqdm import tqdm
import time

from config import Config


class DataDownloader:
    """Download CHB-MIT dataset from PhysioNet"""
    
    def __init__(self):
        self.raw_dir = Config.RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://physionet.org/files/chbmit/1.0.0/"
        
    def download_file(self, url, destination):
        """Download single file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            
            return True
            
        except Exception as e:
            print(f"  Error downloading: {e}")
            return False
    
    def download_patient(self, patient_id, num_files=5):
        """
        Download data for one patient
        
        Args:
            patient_id: patient ID (e.g., 'chb01')
            num_files: number of EDF files to download per patient
        """
        patient_dir = self.raw_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        print(f"\nDownloading {patient_id}...")
        
        # Download summary file first
        summary_file = f"{patient_id}-summary.txt"
        summary_url = f"{self.base_url}{patient_id}/{summary_file}"
        summary_path = patient_dir / summary_file
        
        if not summary_path.exists():
            print(f"  Downloading summary...")
            success = self.download_file(summary_url, summary_path)
            if not success:
                print(f"  Failed to download summary, skipping patient")
                return False
        else:
            print(f"  Summary already exists")
        
        # Download EDF files
        files_downloaded = 0
        for i in range(1, 50):  # Try up to 50 files
            if files_downloaded >= num_files:
                break
                
            file_num = f"{i:02d}"
            edf_file = f"{patient_id}_{file_num}.edf"
            edf_url = f"{self.base_url}{patient_id}/{edf_file}"
            edf_path = patient_dir / edf_file
            
            if edf_path.exists():
                print(f"  {edf_file} exists, skipping")
                files_downloaded += 1
                continue
            
            print(f"  Downloading {edf_file}...")
            success = self.download_file(edf_url, edf_path)
            
            if success:
                files_downloaded += 1
            else:
                # File might not exist, try next number
                if edf_path.exists():
                    edf_path.unlink()  # Remove partial download
                continue
            
            time.sleep(0.5)  # Be nice to server
        
        print(f"  ✓ Downloaded {files_downloaded} files for {patient_id}")
        return files_downloaded > 0
    
    def download_dataset(self, num_patients=5, files_per_patient=5):
        """
        Download CHB-MIT dataset
        
        Args:
            num_patients: number of patients to download
            files_per_patient: number of EDF files per patient
        """
        print("=" * 70)
        print("Downloading CHB-MIT Scalp EEG Database")
        print("=" * 70)
        print(f"\nDataset info:")
        print(f"  Source: PhysioNet")
        print(f"  URL: {self.base_url}")
        print(f"  Destination: {self.raw_dir}")
        print(f"  Patients to download: {num_patients}")
        print(f"  Files per patient: {files_per_patient}")
        print(f"\nThis may take 10-30 minutes depending on connection...")
        
        # Download patients
        patients_downloaded = 0
        for i in range(1, num_patients + 1):
            patient_id = f"chb{i:02d}"
            
            success = self.download_patient(patient_id, files_per_patient)
            
            if success:
                patients_downloaded += 1
        
        print("\n" + "=" * 70)
        print(f"Download complete!")
        print(f"  Patients downloaded: {patients_downloaded}/{num_patients}")
        print(f"  Location: {self.raw_dir}")
        print("=" * 70)
        
        print("\nNext steps:")
        print("  1. Run: python preprocess_data.py")
        print("  2. Then: python train.py")
        
        return patients_downloaded > 0


def main():
    """Main download function"""
    downloader = DataDownloader()
    
    print("\nHow many patients do you want to download?")
    print("  - Quick test: 3 patients (~500MB, 5-10 min)")
    print("  - Small dataset: 5 patients (~1GB, 10-20 min)")
    print("  - Medium dataset: 10 patients (~2GB, 20-40 min)")
    print("  - Full dataset: 24 patients (~5GB, 1-2 hours)")
    
    while True:
        try:
            choice = input("\nEnter number of patients (3-24) [default: 5]: ").strip()
            if choice == "":
                num_patients = 5
            else:
                num_patients = int(choice)
            
            if 1 <= num_patients <= 24:
                break
            else:
                print("Please enter a number between 1 and 24")
        except ValueError:
            print("Please enter a valid number")
    
    files_per_patient = 5  # Fixed for now
    
    print(f"\nStarting download of {num_patients} patients...")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm == 'y':
        success = downloader.download_dataset(num_patients, files_per_patient)
        
        if success:
            print("\n✓ Download successful!")
        else:
            print("\n✗ Download failed. Please check your internet connection.")
    else:
        print("Download cancelled")


if __name__ == "__main__":
    main()