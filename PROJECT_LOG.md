# Brain 3D Splat UNet - Project Log

## Project Overview
**Uncertainty-Aware 3D Brain Tumor Reconstruction using Hybrid UNet + Gaussian Splatting**

Target: European research labs and medical AI companies (Switzerland/Netherlands)

## Current Status ✅
- ✅ BraTS 2021 dataset downloaded and organized (1251 cases)
- ✅ Data exploration and visualization completed
- ✅ 3D UNet architecture designed and tested
- ✅ Training pipeline working (GPU enabled, loss decreasing 1.14→1.11)
- ✅ GitHub repository set up with proper .gitignore

## Key Technical Decisions

### Dataset: BraTS 2021
- **Size**: 1251 training cases
- **Input**: 4 MRI modalities (T1, T1ce, T2, FLAIR)
- **Shape**: (240, 240, 155) per modality → resized to (64, 64, 64) for training
- **Labels**: 4 classes (0=background, 1=necrosis, 2=edema, 4=enhancing tumor)

### Architecture: 3D UNet
- **Parameters**: 5,649,204
- **Input**: (batch, 4, 64, 64, 64) - 4 MRI modalities
- **Output**: (batch, 4, 64, 64, 64) - 4 class predictions
- **Working**: Loss decreasing, GPU enabled, checkpoints saving

## Next Steps
1. **Uncertainty Quantification**: Add Monte Carlo Dropout to UNet
2. **Point Cloud Generation**: Extract surface points from segmentation
3. **Gaussian Splatting**: 3D surface reconstruction and refinement
4. **Performance**: Compare with nnUNet baseline
5. **Web Interface**: Interactive visualization
