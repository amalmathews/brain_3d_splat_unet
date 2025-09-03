# Brain Tumor Segmentation with Uncertainty Quantification

3D brain tumor segmentation using MONAI UNet with Monte Carlo Dropout for epistemic uncertainty estimation, featuring comprehensive 3D visualization and clinical volume analysis.

## Features

- **MONAI UNet Backbone**: State-of-the-art 3D segmentation architecture
- **Monte Carlo Dropout**: Epistemic uncertainty quantification during inference
- **3D Visualization**: Interactive point clouds and mesh generation
- **Clinical Analysis**: Automated volume measurements and clinical metrics
- **Uncertainty Mapping**: Spatial uncertainty visualization for clinical decision support

## Results

- **Validation Dice Score**: 0.53 ± 0.05 on BraTS2021
- **Uncertainty Calibration**: Reliable epistemic uncertainty estimates
- **3D Reconstruction**: Real-time point cloud and mesh generation
- **Clinical Metrics**: Automated tumor volume and ratio calculations

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup
```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
monai>=1.3.0
nibabel>=5.0.0
scikit-image>=0.19.0
open3d>=0.17.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
pandas>=1.5.0
scipy>=1.9.0
```

## Quick Start

### 1. Download Pre-trained Model
```bash
# Download trained weights (replace with actual link)
wget https://github.com/yourusername/brain-tumor-uncertainty/releases/download/v1.0/best_model.pth
mkdir -p checkpoints
mv best_model.pth checkpoints/
```

### 2. Prepare Data
Organize BraTS data in the following structure:
```
data/
└── BraTS2021/
    └── BraTS2021_Training_Data/
        └── BraTS2021_00000/
            ├── BraTS2021_00000_t1.nii.gz
            ├── BraTS2021_00000_t1ce.nii.gz
            ├── BraTS2021_00000_t2.nii.gz
            ├── BraTS2021_00000_flair.nii.gz
            └── BraTS2021_00000_seg.nii.gz
```

### 3. Run 3D Visualization Demo
```bash
python src/visualization/seg_to_pointcloud.py
```

This generates:
- Interactive 3D visualization window
- Point cloud file (`outputs/3d_demo_pointcloud.ply`)
- 3D meshes (`outputs/3d_demo_mesh_label_*.ply`)
- Volume analysis report (`outputs/3d_demo_volume_analysis.json`)

## Usage

### Training
```bash
python src/scripts/train_uncertainty.py
```

Configuration options:
- `BATCH_SIZE`: Training batch size (default: 8)
- `EPOCHS`: Number of training epochs (default: 50)
- `PATCH_SIZE`: Input patch dimensions (default: 96×96×96)
- `MC_SAMPLES`: Monte Carlo dropout samples (default: 10)

### Inference with Uncertainty
```python
from src.visualization.seg_to_pointcloud import SegmentationTo3D

# Initialize model
converter = SegmentationTo3D(
    model_path="checkpoints/best_model.pth",
    device='cuda'
)

# Run inference with uncertainty
results = converter.visualize_3d_results(
    case_path="path/to/case/directory",
    save_path="outputs/results"
)

pcd, meshes, uncertainties, volume_analysis = results
```

### Volume Analysis
The system provides comprehensive clinical metrics:

```python
# Clinical volume measurements
volume_analysis = {
    'regions': {
        1: {'name': 'Necrosis', 'volume_cm3': 2.34, 'percentage': 23.4},
        2: {'name': 'Edema', 'volume_cm3': 15.67, 'percentage': 67.2},
        3: {'name': 'Enhancing', 'volume_cm3': 3.21, 'percentage': 9.4}
    },
    'clinical_metrics': {
        'total_tumor_volume_cm3': 21.22,
        'necrosis_ratio': 0.42,
        'edema_to_tumor_ratio': 2.84
    }
}
```

## Project Structure

```
brain-tumor-uncertainty/
├── src/
│   ├── models/
│   │   └── nnunet_uncertainty.py      # MONAI UNet with MC Dropout
│   ├── data/
│   │   └── dataset.py                 # BraTS dataset loader
│   ├── scripts/
│   │   └── train_uncertainty.py       # Training pipeline
│   ├── evaluation/
│   │   ├── uncertainty_metrics.py     # Evaluation framework
│   │   ├── baseline_methods.py        # Comparison methods
│   │   └── cross_validation.py        # CV framework
│   └── visualization/
│       └── seg_to_pointcloud.py       # 3D visualization pipeline
├── notebooks/
│   └── demo.ipynb                     # Interactive demo
├── outputs/                           # Generated results
├── checkpoints/                       # Model weights
└── docs/                             # Documentation
```

## Methodology

### Model Architecture
- **Backbone**: MONAI UNet with residual blocks
- **Input**: 4-channel brain MRI (T1, T1ce, T2, FLAIR)
- **Output**: 4-class segmentation (background, necrosis, edema, enhancing)
- **Uncertainty**: Monte Carlo Dropout with 10 forward passes

### Uncertainty Quantification
- **Epistemic Uncertainty**: Model uncertainty via MC Dropout
- **Spatial Mapping**: Voxel-wise uncertainty estimation
- **Clinical Integration**: Uncertainty-guided confidence intervals

### 3D Visualization Pipeline
1. **Sliding Window Inference**: Process full brain volumes
2. **Point Cloud Generation**: Convert segmentation to 3D points
3. **Mesh Reconstruction**: Marching cubes algorithm for smooth surfaces
4. **Interactive Visualization**: Open3D-based 3D viewer

## Evaluation

Cross-validation results on BraTS2021:

| Metric | Value | 95% CI |
|--------|-------|--------|
| Dice Score | 0.530 | [0.485, 0.575] |
| Hausdorff Distance | 12.3mm | [10.1, 14.5] |
| Volume Correlation | 0.89 | [0.84, 0.93] |
| Calibration ECE | 0.045 | [0.032, 0.058] |

## Clinical Applications

- **Treatment Planning**: Volume-based therapy decisions
- **Surgical Guidance**: Uncertainty maps for risk assessment
- **Follow-up Monitoring**: Quantitative progression tracking
- **Radiologist Support**: Confidence-aware predictions

## Limitations

- Trained on single dataset (BraTS2021)
- Requires CUDA-capable hardware for real-time inference
- Uncertainty calibration may vary across institutions
- Limited to pre-operative MRI sequences

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request
