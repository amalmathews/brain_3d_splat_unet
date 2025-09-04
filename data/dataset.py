# src/data/dataset.py
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path
import os
import random

class BraTSDataset(Dataset):
    def __init__(self, data_dir, max_cases=None, patch_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.cases = []
        
        for root, dirs, files in os.walk(data_dir):
            for d in dirs:
                full_path = Path(root) / d
                if full_path.is_dir():
                    self.cases.append(full_path)
        
        if max_cases:
            self.cases = self.cases[:max_cases]
        
        print(f"Found {len(self.cases)} cases")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        case = case_dir.name

        vols = [
            self._load_nifti(case_dir / f"{case}_t1.nii.gz"),
            self._load_nifti(case_dir / f"{case}_t1ce.nii.gz"),
            self._load_nifti(case_dir / f"{case}_t2.nii.gz"),
            self._load_nifti(case_dir / f"{case}_flair.nii.gz"),
        ]
        seg = self._load_nifti(case_dir / f"{case}_seg.nii.gz", is_label=True)

        image = np.stack(vols, axis=0).astype(np.float32)  # (4, D, H, W)
        
        # Random crop
        image, seg = self._random_crop(image, seg, self.patch_size)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(seg).long(),
            "case_id": case
        }
    
    def _random_crop(self, image, seg, crop_size):
        d, h, w = image.shape[1:]
        cd, ch, cw = crop_size
        
        if d <= cd or h <= ch or w <= cw:
            # Center crop if too small
            start_d = max((d - cd) // 2, 0)
            start_h = max((h - ch) // 2, 0)
            start_w = max((w - cw) // 2, 0)
        else:
            start_d = random.randint(0, d - cd)
            start_h = random.randint(0, h - ch)
            start_w = random.randint(0, w - cw)
        
        end_d = min(start_d + cd, d)
        end_h = min(start_h + ch, h)
        end_w = min(start_w + cw, w)
        
        image_crop = image[:, start_d:end_d, start_h:end_h, start_w:end_w]
        seg_crop = seg[start_d:end_d, start_h:end_h, start_w:end_w]
        
        return image_crop, seg_crop
    
    def _load_nifti(self, path, is_label=False):
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32)  # (H, W, D)
        data = np.moveaxis(data, -1, 0)                   # (D, H, W)
        
        if is_label:
            lab = data.astype(np.int64)
            lab[lab == 4] = 3  # Map label 4 to 3
            return lab
        
        # Normalize image
        nz = data[data > 0]
        if nz.size:
            p1, p99 = np.percentile(nz, (1, 99))
            data = np.clip(data, p1, p99)
            mu, sigma = nz.mean(), nz.std() + 1e-6
            data = (data - mu) / sigma
        return data