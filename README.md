# 🧠 Brain 3D Splat UNet

**Uncertainty-Aware 3D Brain Tumor Reconstruction using Hybrid UNet + Gaussian Splatting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Project Overview

This project combines state-of-the-art deep learning techniques for brain tumor analysis:
- **3D UNet segmentation** for multi-modal MRI processing
- **Uncertainty quantification** for clinical safety and reliability  
- **Gaussian Splatting** for enhanced 3D surface reconstruction
- **Interactive visualization** for medical professionals


## ✅ Current Status

- 🎯 **3D UNet architecture** working on GPU (5.6M parameters)
- 📊 **BraTS 2021 dataset** integrated (1251 brain tumor cases)
- 🚀 **Training pipeline** functional (GPU enabled, loss decreasing)
- 📋 **Data exploration** complete with visualization tools
- 🔧 **Ready for** uncertainty quantification and Gaussian Splatting integration

## 🏗️ Architecture Pipeline

```
Input (4 MRI Modalities) → 3D UNet → Segmentation + Uncertainty → Point Clouds → Gaussian Splatting → Interactive 3D Visualization
```

### Key Innovation
**Hybrid approach**: Traditional medical segmentation + cutting-edge 3D reconstruction for unprecedented tumor visualization with confidence estimates.

## 🔧 Quick Start

### Prerequisites
```bash
pip install torch torchvision nibabel matplotlib plotly jupyter
```

### Basic Usage
```bash
# Clone repository
git clone https://github.com/amalmathews/brain_3d_splat_unet.git
cd brain_3d_splat_unet

# Test the 3D UNet model
python src/models/unet.py

# Test training pipeline
python scripts/train_simple.py

# Explore data
jupyter lab notebooks/01_explore_brats_data.ipynb
```

## 📊 Dataset: BraTS 2021

- **Size**: 1251 brain tumor cases
- **Modalities**: T1, T1ce, T2, FLAIR MRI sequences
- **Input Shape**: (240, 240, 155) per modality
- **Labels**: 4 classes
  - 0: Background (normal brain)
  - 1: Necrotic core (dead tumor tissue)
  - 2: Peritumoral edema (swelling)
  - 4: Enhancing tumor (active tumor)

### Data Organization
```
data/raw/BraTS2021/BraTS2021_Training_Data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
└── ...
```

## 🎯 Technical Highlights

### 3D UNet Architecture
- **Multi-modal input**: 4 MRI sequences processed simultaneously
- **Skip connections**: Preserve fine tumor boundary details
- **GPU optimized**: Efficient 3D convolutions for volumetric data
- **Medical imaging focused**: Designed specifically for brain tumor segmentation

### Planned Innovations
1. **Monte Carlo Dropout**: Native uncertainty estimation
2. **Point Cloud Extraction**: Surface-aware feature extraction
3. **Gaussian Splatting**: Novel 3D reconstruction technique
4. **Clinical Interface**: Interactive uncertainty visualization

## 📁 Project Structure

```
brain_3d_splat_unet/
├── src/
│   ├── data/
│   │   ├── dataset.py           # BraTS data loader
│   │   └── visualize_data.py    # Data exploration tools
│   ├── models/
│   │   └── unet.py             # 3D UNet architecture
│   ├── training/               # Training utilities
│   ├── inference/              # Inference and visualization
│   └── utils/                  # Helper functions
├── scripts/
│   └── train_simple.py         # Training pipeline
├── notebooks/
│   └── 01_explore_brats_data.ipynb
├── checkpoints/                # Model checkpoints
├── config/                     # Configuration files
└── PROJECT_LOG.md             # Detailed progress tracking
```

## 🔬 Research Focus

### Why This Combination?
- **Medical AI Compliance**: Uncertainty quantification for clinical safety
- **Novel Integration**: First combination of UNet + Gaussian Splatting for medical imaging
- **European Standards**: Designed for strict regulatory environments
- **Portfolio Value**: Demonstrates cutting-edge ML + practical medical applications

### Target Outcomes
- **Segmentation Accuracy**: Competitive with nnUNet (80-90% Dice score)
- **3D Reconstruction**: Superior to traditional surface rendering
- **Clinical Usability**: Uncertainty-aware predictions for medical professionals
- **Research Impact**: Novel architecture suitable for publication


## 📈 Development Roadmap

- [x] **Phase 1**: Data exploration and baseline UNet ✅
- [ ] **Phase 2**: Uncertainty quantification integration
- [ ] **Phase 3**: Gaussian Splatting implementation  
- [ ] **Phase 4**: End-to-end pipeline optimization
- [ ] **Phase 5**: Interactive web interface
- [ ] **Phase 6**: Performance benchmarking vs. nnUNet


## 📄 License

MIT License - see LICENSE file for details.

---

**🎯 Goal**: Create a novel, uncertainty-aware brain tumor analysis pipeline that combines the reliability of medical imaging standards with cutting-edge 3D reconstruction techniques.