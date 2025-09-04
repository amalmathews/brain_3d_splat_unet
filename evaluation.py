# evaluation/simple_eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
from pathlib import Path
import json
import sys
import os

# Add project root to path

class SimpleEvaluator:
    def __init__(self, save_dir="outputs/evaluation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model(self, model, val_loader, device):
        """Essential uncertainty evaluation"""
        model.eval()
        
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        dice_scores = []
        
        print(f"Evaluating {len(val_loader)} cases...")
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 15:  # Limit for analysis
                break
                
            x = batch['image'].to(device)
            y = batch['mask'].to(device)
            if len(y.shape) > 4:
                y = y.squeeze(1)
            
            with torch.no_grad():
                # Get MC predictions  
                uncertainty_results = model.monte_carlo_forward(x, n_samples=10)
                
                # Debug: print what keys are available
                print(f"Available keys: {uncertainty_results.keys()}")
                
                # Handle different return formats
                if 'predictions' in uncertainty_results:
                    predictions = uncertainty_results['predictions']
                elif 'prob' in uncertainty_results:
                    predictions = uncertainty_results['prob']
                else:
                    # Fallback: direct model prediction
                    predictions = torch.softmax(model(x), dim=1)
                
                # Get uncertainty - handle different key names
                if 'epistemic' in uncertainty_results and 'aleatoric' in uncertainty_results:
                    epistemic = uncertainty_results['epistemic']
                    aleatoric = uncertainty_results['aleatoric']
                elif 'entropy' in uncertainty_results:
                    epistemic = uncertainty_results['entropy']
                    aleatoric = torch.zeros_like(epistemic)
                elif 'mi' in uncertainty_results:
                    epistemic = uncertainty_results['mi']
                    aleatoric = uncertainty_results.get('entropy', torch.zeros_like(epistemic))
                else:
                    # Fallback: calculate simple variance as uncertainty
                    mc_predictions = []
                    for _ in range(5):
                        mc_pred = torch.softmax(model(x), dim=1)
                        mc_predictions.append(mc_pred)
                    mc_stack = torch.stack(mc_predictions)
                    epistemic = torch.var(mc_stack, dim=0).mean(dim=1, keepdim=True)
                    aleatoric = torch.zeros_like(epistemic)
                
                # Calculate Dice
                pred_labels = torch.argmax(predictions, dim=1)
                dice = self._calculate_dice(pred_labels, y)
                dice_scores.append(dice.item())
                
                print(f"Case {batch_idx+1}: Dice = {dice.item():.3f}")
                
                # Store for analysis
                all_predictions.append(predictions.cpu())
                all_uncertainties.append((epistemic + aleatoric).cpu())
                all_targets.append(y.cpu())
        
        if len(all_predictions) == 0:
            print("ERROR: No data processed!")
            return None
            
        # Combine all data
        predictions = torch.cat(all_predictions, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        results = {
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'calibration': self._simple_calibration(predictions, targets),
            'uncertainty_stats': self._uncertainty_stats(uncertainties)
        }
        
        # Create plots
        self._create_plots(predictions, uncertainties, targets, dice_scores)
        
        # Save results
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _calculate_dice(self, pred, target):
        """Calculate Dice coefficient"""
        # Ensure predictions and targets are in the right format
        if len(pred.shape) > len(target.shape):
            pred = pred.squeeze()
        if len(target.shape) > len(pred.shape):
            target = target.squeeze()
            
        # Convert to binary masks for each class
        dice_scores = []
        
        for class_id in range(1, 4):  # Classes 1, 2, 3 (skip background)
            pred_class = (pred == class_id).float()
            target_class = (target == class_id).float()
            
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class)
            
            if union > 0:
                dice = (2.0 * intersection) / (union + 1e-8)
            else:
                dice = torch.tensor(1.0)  # Perfect score if no ground truth or prediction
                
            dice_scores.append(dice.item())
        
        # Return mean Dice across classes
        return torch.tensor(np.mean(dice_scores))
    
    def _simple_calibration(self, predictions, targets):
        """Simple calibration analysis"""
        probs = torch.softmax(predictions, dim=1)
        max_probs = torch.max(probs, dim=1)[0].flatten().numpy()
        
        pred_labels = torch.argmax(predictions, dim=1).flatten().numpy()
        true_labels = targets.flatten().numpy()
        correct = (pred_labels == true_labels).astype(float)
        
        # Simple ECE calculation with 10 bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            
            if np.sum(in_bin) > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * np.sum(in_bin)
        
        ece = ece / len(max_probs)
        
        # Brier score
        brier = brier_score_loss(correct, max_probs)
        
        return {
            'ece': float(ece),
            'brier_score': float(brier)
        }
    
    def _uncertainty_stats(self, uncertainties):
        """Basic uncertainty statistics"""
        uncertainties = uncertainties.flatten().numpy()
        
        return {
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'median': float(np.median(uncertainties)),
            'max': float(np.max(uncertainties))
        }
    
    def _create_plots(self, predictions, uncertainties, targets, dice_scores):
        """Create essential plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Dice scores
        ax1.hist(dice_scores, bins=10, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(dice_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(dice_scores):.3f}')
        ax1.set_xlabel('Dice Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Segmentation Performance')
        ax1.legend()
        
        # 2. Uncertainty distribution
        uncertainty_vals = uncertainties.flatten().numpy()
        ax2.hist(uncertainty_vals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Uncertainty Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Uncertainty Distribution')
        ax2.set_yscale('log')
        
        # 3. Confidence vs Dice correlation
        probs = torch.softmax(predictions, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        # Sample a subset for visualization
        sample_size = min(1000, len(dice_scores))
        sample_idx = np.random.choice(len(dice_scores), sample_size, replace=False)
        
        ax3.scatter([max_probs[i].mean().item() for i in sample_idx], 
                   [dice_scores[i] for i in sample_idx], alpha=0.6)
        ax3.set_xlabel('Mean Confidence')
        ax3.set_ylabel('Dice Score')
        ax3.set_title('Confidence vs Performance')
        
        # 4. Prediction accuracy distribution
        pred_labels = torch.argmax(predictions, dim=1)
        accuracies = []
        for i in range(len(pred_labels)):
            acc = (pred_labels[i] == targets[i]).float().mean().item()
            accuracies.append(acc)
            
        ax4.hist(accuracies, bins=15, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Pixel Accuracy')
        ax4.set_ylabel('Frequency') 
        ax4.set_title('Pixel-wise Accuracy Distribution')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to: {self.save_dir / 'evaluation_plots.png'}")

def main():
    # Import your modules
    from models.nnunet_uncertainty import nnUNetUncertainty
    from data.dataset import BraTSDataset
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model - try different checkpoint names
    checkpoint_files = [
        "checkpoints/best_model.pth",
        "checkpoints/final_model.pth",
        "checkpoints/monai_uncertainty_epoch_50.pth"
    ]
    
    checkpoint_path = None
    for cp in checkpoint_files:
        if os.path.exists(cp):
            checkpoint_path = cp
            break
    
    if not checkpoint_path:
        print("Available checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                print(f"  - {f}")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    model = nnUNetUncertainty(input_channels=4, num_classes=4, dropout_p=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Find data directory
    data_paths = [
        "data/BraTS2021/BraTS2021_Training_Data",
        "../data/BraTS2021/BraTS2021_Training_Data",
        "src/data/BraTS2021/BraTS2021_Training_Data"
    ]
    
    data_dir = None
    for dp in data_paths:
        if os.path.exists(dp):
            data_dir = dp
            break
    
    if not data_dir:
        print("Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  - {item}")
        return
    
    print(f"Loading dataset from: {data_dir}")
    
    # Create dataset
    dataset = BraTSDataset(data_dir=data_dir, max_cases=None, patch_size=(96, 96, 96))
    
    if len(dataset.cases) == 0:
        print(f"No cases found in {data_dir}")
        return
    
    # Use subset for evaluation
    num_eval = min(20, len(dataset.cases))
    dataset.cases = dataset.cases[:num_eval]
    print(f"Using {len(dataset.cases)} cases for evaluation")
    
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Run evaluation
    evaluator = SimpleEvaluator()
    results = evaluator.evaluate_model(model, val_loader, device)
    
    if results:
        # Print results
        print("\nEVALUATION RESULTS:")
        print("=" * 40)
        print(f"Mean Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
        print(f"Expected Calibration Error: {results['calibration']['ece']:.4f}")
        print(f"Brier Score: {results['calibration']['brier_score']:.4f}")
        print(f"Mean Uncertainty: {results['uncertainty_stats']['mean']:.4f}")
        print(f"Max Uncertainty: {results['uncertainty_stats']['max']:.4f}")

if __name__ == "__main__":
    main()