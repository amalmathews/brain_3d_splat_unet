# src/evaluation/essential_evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from pathlib import Path
import json

class UncertaintyEvaluator:
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
            if batch_idx >= 20:  # Limit for essential analysis
                break
                
            x = batch['image'].to(device)
            y = batch['mask'].to(device).squeeze(1)
            
            with torch.no_grad():
                # Get MC predictions
                uncertainty_results = model.monte_carlo_forward(x, n_samples=10)
                
                predictions = uncertainty_results['predictions']  # (B, C, D, H, W)
                epistemic = uncertainty_results['epistemic']      # (B, 1, D, H, W)
                aleatoric = uncertainty_results['aleatoric']      # (B, 1, D, H, W)
                
                # Calculate Dice
                pred_labels = torch.argmax(predictions, dim=1)
                dice = self._calculate_dice(pred_labels, y)
                dice_scores.append(dice.item())
                
                # Store for analysis
                all_predictions.append(predictions.cpu())
                all_uncertainties.append((epistemic + aleatoric).cpu())
                all_targets.append(y.cpu())
        
        # Combine all data
        predictions = torch.cat(all_predictions, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Run three essential evaluations
        results = {
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'calibration': self._calibration_analysis(predictions, targets),
            'error_detection': self._error_detection_analysis(predictions, uncertainties, targets),
            'uncertainty_stats': self._uncertainty_statistics(uncertainties)
        }
        
        # Create essential plots
        self._create_plots(predictions, uncertainties, targets, dice_scores)
        
        # Save results
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _calculate_dice(self, pred, target):
        """Calculate Dice coefficient"""
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection) / (union + 1e-8)
        return dice
    
    def _calibration_analysis(self, predictions, targets):
        """Model calibration analysis"""
        # Convert to probabilities
        probs = torch.softmax(predictions, dim=1)
        max_probs = torch.max(probs, dim=1)[0].flatten().numpy()
        
        # Get predictions and correctness
        pred_labels = torch.argmax(predictions, dim=1).flatten().numpy()
        true_labels = targets.flatten().numpy()
        correct = (pred_labels == true_labels).astype(float)
        
        # Calculate calibration metrics
        bin_boundaries = np.linspace(0, 1, 11)  # 10 bins
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = 0
                
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        
        # Expected Calibration Error
        ece = np.mean([abs(acc - conf) for acc, conf in zip(accuracies, confidences)])
        
        # Brier Score
        brier = brier_score_loss(correct, max_probs)
        
        return {
            'ece': float(ece),
            'brier_score': float(brier),
            'bin_accuracies': accuracies,
            'bin_confidences': confidences
        }
    
    def _error_detection_analysis(self, predictions, uncertainties, targets):
        """How well uncertainty detects errors"""
        pred_labels = torch.argmax(predictions, dim=1).flatten().numpy()
        true_labels = targets.flatten().numpy()
        errors = (pred_labels != true_labels).astype(float)
        uncertainty_vals = uncertainties.flatten().numpy()
        
        # AUC for uncertainty as error detector
        try:
            auc = roc_auc_score(errors, uncertainty_vals)
        except:
            auc = 0.5
            
        # Correlation
        correlation = np.corrcoef(uncertainty_vals, errors)[0, 1]
        
        return {
            'auc_score': float(auc),
            'correlation': float(correlation)
        }
    
    def _uncertainty_statistics(self, uncertainties):
        """Basic uncertainty statistics"""
        uncertainties = uncertainties.flatten().numpy()
        
        return {
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'median': float(np.median(uncertainties)),
            'percentile_95': float(np.percentile(uncertainties, 95))
        }
    
    def _create_plots(self, predictions, uncertainties, targets, dice_scores):
        """Create three essential plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Dice score distribution
        ax1.hist(dice_scores, bins=15, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(dice_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(dice_scores):.3f}')
        ax1.set_xlabel('Dice Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Segmentation Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Calibration plot
        calib_results = self._calibration_analysis(predictions, targets)
        ax2.bar(calib_results['bin_confidences'], calib_results['bin_accuracies'], 
               width=0.08, alpha=0.7, edgecolor='black')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Calibration (ECE: {calib_results["ece"]:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Uncertainty distribution
        uncertainty_vals = uncertainties.flatten().numpy()
        ax3.hist(uncertainty_vals, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Uncertainty Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Uncertainty Distribution')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error detection
        pred_labels = torch.argmax(predictions, dim=1).flatten().numpy()
        true_labels = targets.flatten().numpy()
        errors = (pred_labels != true_labels).astype(float)
        
        # Sample for visualization
        sample_idx = np.random.choice(len(uncertainty_vals), 5000, replace=False)
        ax4.scatter(uncertainty_vals[sample_idx], errors[sample_idx], alpha=0.3, s=1)
        ax4.set_xlabel('Uncertainty')
        ax4.set_ylabel('Error (1=Wrong, 0=Correct)')
        ax4.set_title('Uncertainty vs Errors')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_evaluation(checkpoint_path, data_dir, save_dir="outputs/evaluation"):
    """Run essential evaluation"""
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, project_root)
    
    from src.models.nnunet_uncertainty import nnUNetUncertainty
    from src.data.dataset import BraTSDataset
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = nnUNetUncertainty(input_channels=4, num_classes=4, dropout_p=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset (use different cases than training)
    dataset = BraTSDataset(data_dir=data_dir, max_cases=30, patch_size=(96, 96, 96))
    dataset.cases = dataset.cases[800:830]  # Use evaluation cases
    
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Run evaluation
    evaluator = UncertaintyEvaluator(save_dir)
    results = evaluator.evaluate_model(model, val_loader, device)
    
    # Print results
    print("\nESSENTIAL EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Mean Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"Expected Calibration Error: {results['calibration']['ece']:.4f}")
    print(f"Brier Score: {results['calibration']['brier_score']:.4f}")
    print(f"Uncertainty-Error AUC: {results['error_detection']['auc_score']:.4f}")
    print(f"Mean Uncertainty: {results['uncertainty_stats']['mean']:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_evaluation(
        checkpoint_path="checkpoints/best_model.pth",
        data_dir="src/data/BraTS2021/BraTS2021_Training_Data"
    )