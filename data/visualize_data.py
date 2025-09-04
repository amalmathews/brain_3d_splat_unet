import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

def load_case(case_dir):
    case_name = case_dir.name
    
    # Load all 4 modalities
    t1 = nib.load(case_dir / f"{case_name}_t1.nii.gz").get_fdata()
    t1ce = nib.load(case_dir / f"{case_name}_t1ce.nii.gz").get_fdata()
    t2 = nib.load(case_dir / f"{case_name}_t2.nii.gz").get_fdata()
    flair = nib.load(case_dir / f"{case_name}_flair.nii.gz").get_fdata()
    seg = nib.load(case_dir / f"{case_name}_seg.nii.gz").get_fdata()
    
    return {
        'T1': t1,
        'T1ce': t1ce, 
        'T2': t2,
        'FLAIR': flair,
        'Segmentation': seg
    }, case_name

def visualize_case(data, case_name, slice_idx=None):
    
    # If no slice specified, use middle slice
    if slice_idx is None:
        slice_idx = data['T1'].shape[2] // 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{case_name} - Slice {slice_idx}', fontsize=16)
    
    # Plot each modality
    modalities = ['T1', 'T1ce', 'T2', 'FLAIR', 'Segmentation']
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    
    for i, (mod, pos) in enumerate(zip(modalities, positions)):
        ax = axes[pos[0], pos[1]]
        
        if mod == 'Segmentation':
            # Special colormap for segmentation
            im = ax.imshow(data[mod][:, :, slice_idx], cmap='tab10', vmin=0, vmax=4)
            ax.set_title(f'{mod}\n0=Background, 1=Necrosis, 2=Edema, 4=Enhancing')
        else:
            # Regular grayscale for MRI
            im = ax.imshow(data[mod][:, :, slice_idx], cmap='gray')
            ax.set_title(f'{mod}')
        
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    # Add some stats
    stats_text = f"""
    Data Shape: {data['T1'].shape}
    T1 range: [{data['T1'].min():.1f}, {data['T1'].max():.1f}]
    T1ce range: [{data['T1ce'].min():.1f}, {data['T1ce'].max():.1f}]
    T2 range: [{data['T2'].min():.1f}, {data['T2'].max():.1f}]
    FLAIR range: [{data['FLAIR'].min():.1f}, {data['FLAIR'].max():.1f}]
    
    Segmentation labels: {np.unique(data['Segmentation'])}
    """
    
    fig.text(0.7, 0.3, stats_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def compare_tumor_slices(data, case_name):
    """Find and show slices with most tumor"""
    seg = data['Segmentation']
    
    # Find slices with tumor (non-zero values)
    tumor_per_slice = []
    for i in range(seg.shape[2]):
        tumor_pixels = np.sum(seg[:, :, i] > 0)
        tumor_per_slice.append(tumor_pixels)
    
    # Get top 3 slices with most tumor
    top_slices = np.argsort(tumor_per_slice)[-3:]
    
    print(f"Slices with most tumor: {top_slices}")
    print(f"Tumor pixels per slice: {[tumor_per_slice[i] for i in top_slices]}")
    
    # Visualize these slices
    for slice_idx in top_slices:
        print(f"\n--- Slice {slice_idx} ---")
        visualize_case(data, case_name, slice_idx)

def main():
    """Main visualization function"""
    # Path to your data
    data_path = Path("BraTS2021/BraTS2021_Training_Data")
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("Make sure you're running from src/data/ directory")
        return
    
    # Get all cases
    case_dirs = sorted(list(data_path.glob("BraTS2021_*")))
    print(f"Found {len(case_dirs)} cases")
    
    # Load first case
    print("Loading first case...")
    data, case_name = load_case(case_dirs[0])
    
    print(f"Case: {case_name}")
    print(f"Data shape: {data['T1'].shape}")
    print(f"Segmentation labels: {np.unique(data['Segmentation'])}")
    
    # Visualize middle slice
    print("\n1. Visualizing middle slice...")
    visualize_case(data, case_name)
    
    # Find and visualize tumor slices
    print("\n2. Finding slices with most tumor...")
    compare_tumor_slices(data, case_name)
    
    # Try another case
    print(f"\n3. Loading case with tumor (trying case 10)...")
    if len(case_dirs) > 10:
        data2, case_name2 = load_case(case_dirs[10])
        visualize_case(data2, case_name2)
        compare_tumor_slices(data2, case_name2)

if __name__ == "__main__":
    main()