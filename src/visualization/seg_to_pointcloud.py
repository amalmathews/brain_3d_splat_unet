# src/visualization/seg_to_pointcloud.py
import torch
import numpy as np
import nibabel as nib
from skimage import measure
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

class SegmentationTo3D:
    def __init__(self, model_path, device='cuda'):
        """Initialize with trained uncertainty model"""
        self.device = torch.device(device)
        self.model = None
        self.load_model(model_path)
        
        # Color mapping for different tumor regions
        self.colors = {
            0: [0.8, 0.8, 0.8],    # Background - light gray
            1: [1.0, 0.0, 0.0],    # Necrosis - red  
            2: [0.0, 1.0, 0.0],    # Edema - green
            3: [0.0, 0.0, 1.0],    # Enhancing - blue
        }
    
    def load_model(self, model_path):
        """Load trained uncertainty model"""
        from src.models.nnunet_uncertainty import nnUNetUncertainty
        
        self.model = nnUNetUncertainty(
            input_channels=4,
            num_classes=4,
            dropout_p=0.3
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def predict_with_uncertainty(self, image_path, mc_samples=20, patch_size=(96, 96, 96)):
        """Make prediction with uncertainty on a single case using sliding window"""
        # Load BraTS case
        case_dir = Path(image_path)
        case_name = case_dir.name
        
        # Load all modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        volumes = []
        
        for mod in modalities:
            file_path = case_dir / f"{case_name}_{mod}.nii.gz"
            if file_path.exists():
                img = nib.load(file_path)
                data = np.asarray(img.dataobj, dtype=np.float32)
                data = np.moveaxis(data, -1, 0)  # (D, H, W)
                
                # Normalize
                nz = data[data > 0]
                if nz.size:
                    p1, p99 = np.percentile(nz, (1, 99))
                    data = np.clip(data, p1, p99)
                    mu, sigma = nz.mean(), nz.std() + 1e-6
                    data = (data - mu) / sigma
                volumes.append(data)
            else:
                print(f"Warning: {file_path} not found")
                return None, None, None
        
        # Stack and prepare for model
        image_array = np.stack(volumes, axis=0).astype(np.float32)  # (4, D, H, W)
        
        # Use sliding window prediction for full volume
        pred_labels, epistemic_uncertainty = self._sliding_window_prediction(
            image_array, patch_size, mc_samples
        )
        
        return pred_labels, epistemic_uncertainty, image_array
    
    def _sliding_window_prediction(self, image_array, patch_size=(96, 96, 96), mc_samples=20):
        """Sliding window prediction for full volume"""
        c, d, h, w = image_array.shape
        pd, ph, pw = patch_size
        
        # Output arrays
        pred_probs = np.zeros((4, d, h, w), dtype=np.float32)  # 4 classes
        uncertainty_map = np.zeros((d, h, w), dtype=np.float32)
        count_map = np.zeros((d, h, w), dtype=np.int32)
        
        # Calculate step size (with overlap)
        step_d, step_h, step_w = pd // 2, ph // 2, pw // 2
        
        print(f"Processing volume {image_array.shape} with patches {patch_size}")
        
        # Sliding window
        for z in range(0, d - pd + 1, step_d):
            for y in range(0, h - ph + 1, step_h):
                for x in range(0, w - pw + 1, step_w):
                    # Extract patch
                    patch = image_array[:, z:z+pd, y:y+ph, x:x+pw]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)
                    
                    # Get prediction with uncertainty
                    with torch.no_grad():
                        uncertainty_results = self.model.monte_carlo_forward(
                            patch_tensor, n_samples=mc_samples
                        )
                    
                    # Extract results
                    patch_pred = uncertainty_results['predictions'].cpu().numpy()[0]  # (4, pd, ph, pw)
                    patch_uncertainty = uncertainty_results['epistemic_uncertainty'].cpu().numpy()[0, 0]  # (pd, ph, pw)
                    
                    # Accumulate results
                    pred_probs[:, z:z+pd, y:y+ph, x:x+pw] += patch_pred
                    uncertainty_map[z:z+pd, y:y+ph, x:x+pw] += patch_uncertainty
                    count_map[z:z+pd, y:y+ph, x:x+pw] += 1
        
        # Average overlapping predictions
        for i in range(4):
            pred_probs[i] = np.divide(pred_probs[i], count_map, 
                                    out=np.zeros_like(pred_probs[i]), 
                                    where=count_map!=0)
        
        uncertainty_map = np.divide(uncertainty_map, count_map,
                                  out=np.zeros_like(uncertainty_map),
                                  where=count_map!=0)
        
        # Convert to label map
        pred_labels = np.argmax(pred_probs, axis=0)
        
        print(f"Prediction completed. Unique labels: {np.unique(pred_labels)}")
        
        return pred_labels, uncertainty_map
    
    def segmentation_to_pointcloud(self, segmentation, uncertainty=None, spacing=(1.0, 1.0, 1.0)):
        """Convert segmentation to colored point cloud"""
        points = []
        colors = []
        uncertainties = []
        
        # Get coordinates of all non-background voxels
        coords = np.where(segmentation > 0)
        
        if len(coords[0]) == 0:
            print("No foreground voxels found")
            return None
        
        # Convert voxel coordinates to world coordinates
        for i in range(len(coords[0])):
            z, y, x = coords[0][i], coords[1][i], coords[2][i]
            
            # World coordinates with spacing
            world_coord = [x * spacing[0], y * spacing[1], z * spacing[2]]
            points.append(world_coord)
            
            # Color based on segmentation label
            label = segmentation[z, y, x]
            color = self.colors.get(label, [0.5, 0.5, 0.5])
            colors.append(color)
            
            # Add uncertainty if available
            if uncertainty is not None:
                uncertainties.append(uncertainty[z, y, x])
            else:
                uncertainties.append(0.0)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd, np.array(uncertainties)
    
    def create_mesh_from_segmentation(self, segmentation, label=1, smooth=True):
        """Create 3D mesh from segmentation using marching cubes"""
        # Extract specific label
        binary_mask = (segmentation == label).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            return None
        
        # Marching cubes to get mesh
        try:
            verts, faces, normals, values = measure.marching_cubes(
                binary_mask, 
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
            
            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            
            # Color the mesh
            color = self.colors.get(label, [0.5, 0.5, 0.5])
            mesh.paint_uniform_color(color)
            
            if smooth:
                mesh = mesh.filter_smooth_simple(number_of_iterations=5)
                mesh.compute_vertex_normals()
            
            return mesh
        except Exception as e:
            print(f"Error creating mesh for label {label}: {e}")
            return None
    
    def visualize_3d_results(self, case_path, save_path=None):
        """Complete pipeline: predict -> point cloud -> 3D visualization"""
        print(f"Processing case: {case_path}")
        
        # Make prediction
        pred_labels, uncertainty, image_array = self.predict_with_uncertainty(case_path)
        
        if pred_labels is None:
            return
        
        print(f"Prediction shape: {pred_labels.shape}")
        print(f"Unique labels: {np.unique(pred_labels)}")
        
        # Create point cloud
        pcd, uncertainties = self.segmentation_to_pointcloud(pred_labels, uncertainty)
        
        if pcd is None:
            return
        
        print(f"Point cloud created with {len(pcd.points)} points")
        
        # Create meshes for each tumor region
        meshes = []
        for label in [1, 2, 3]:  # Skip background
            if label in np.unique(pred_labels):
                mesh = self.create_mesh_from_segmentation(pred_labels, label)
                if mesh is not None:
                    meshes.append(mesh)
                    print(f"Created mesh for label {label}")
        
        # Visualize
        vis_objects = [pcd] + meshes
        
        if save_path:
            # Save point cloud
            o3d.io.write_point_cloud(f"{save_path}_pointcloud.ply", pcd)
            
            # Save meshes
            for i, mesh in enumerate(meshes):
                o3d.io.write_triangle_mesh(f"{save_path}_mesh_label_{i+1}.ply", mesh)
            
            print(f"Saved results to {save_path}")
        
        # Interactive visualization
        o3d.visualization.draw_geometries(
            vis_objects,
            window_name="Brain Tumor 3D Visualization",
            width=1200,
            height=800
        )
        
        return pcd, meshes, uncertainties

# Demo script
def main():
    # Initialize converter with your trained model
    converter = SegmentationTo3D(
        model_path="checkpoints/best_model.pth",
        device='cuda'
    )
    
    # Path to a BraTS case directory
    case_path = "src/data/BraTS2021/BraTS2021_Training_Data/BraTS2021_00000"
    
    # Create 3D visualization
    results = converter.visualize_3d_results(
        case_path=case_path,
        save_path="outputs/3d_demo"
    )
    
    if results:
        pcd, meshes, uncertainties = results
        
        # Additional analysis
        print(f"Uncertainty statistics:")
        print(f"  Mean: {uncertainties.mean():.4f}")
        print(f"  Std: {uncertainties.std():.4f}")
        print(f"  Max: {uncertainties.max():.4f}")

if __name__ == "__main__":
    main()