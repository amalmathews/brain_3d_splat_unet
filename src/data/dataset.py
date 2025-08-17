"""
Simple BraTS dataset loader
"""

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path

class BraTSDataset(Dataset):
    def __init__(self, data_dir, max_cases=None):
        self.data_dir = Path(data_dir)
        
        # Find all patient directories
        self.case_dirs = sorted(list(self.data_dir.glob("BraTS2021_*")))
        
        # Limit cases for testing
        if max_cases:
            self.case_dirs = self.case_dirs[:max_cases]
        
        print(f"Found {len(self.case_dirs)} cases")
    
    def __len__(self):
        return len(self.case_dirs)
    
    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_name = case_dir.name
        
        # Load all 4 modalities
        t1 = self._load_nifti(case_dir / f"{case_name}_t1.nii.gz")
        t1ce = self._load_nifti(case_dir / f"{case_name}_t1ce.nii.gz") 
        t2 = self._load_nifti(case_dir / f"{case_name}_t2.nii.gz")
        flair = self._load_nifti(case_dir / f"{case_name}_flair.nii.gz")
        
        # Load segmentation
        seg = self._load_nifti(case_dir / f"{case_name}_seg.nii.gz")
        
        # Stack modalities: (4, H, W, D)
        image = np.stack([t1, t1ce, t2, flair], axis=0)
        
        return {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(seg),
            'case_id': case_name
        }
    
    def _load_nifti(self, path):
        """Load and normalize NIfTI file"""
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)
        
        # Simple normalization
        if data.max() > 0:
            data = data / data.max()
        
        return data

# Test the dataset
if __name__ == "__main__":
    dataset = BraTSDataset(
        "BraTS2021/BraTS2021_Training_Data",
        max_cases=5  # Test with 5 cases
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Case ID: {sample['case_id']}")
    print(f"Unique mask values: {torch.unique(sample['mask'])}")