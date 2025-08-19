"""
Training script for uncertainty-aware UNet on BraTS data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.uncertainty_unet import UncertaintyUNet3D
from data.dataset import BraTSDataset

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    intersection = (pred * target_one_hot).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target_one_hot.sum() + smooth)
    return dice

def uncertainty_loss(model, images, targets, n_samples=5):
    """Calculate uncertainty-aware loss"""
    # Standard prediction
    model.eval()
    pred = model(images)
    
    # Cross entropy loss
    ce_loss = nn.CrossEntropyLoss()(pred, targets)
    
    # Monte Carlo predictions for uncertainty
    model.train()
    mc_predictions = []
    for _ in range(n_samples):
        mc_pred = model(images)
        mc_predictions.append(mc_pred)
    
    mc_predictions = torch.stack(mc_predictions)
    
    # Calculate variance (uncertainty)
    mc_mean = mc_predictions.mean(dim=0)
    mc_var = mc_predictions.var(dim=0)
    
    # Uncertainty regularization (encourage meaningful uncertainty)
    pred_errors = torch.abs(torch.argmax(pred, dim=1) - targets).float()
    uncertainty_reg = torch.mean(torch.abs(mc_var.mean(dim=1) - pred_errors))
    
    total_loss = ce_loss + 0.1 * uncertainty_reg
    
    return total_loss, ce_loss, uncertainty_reg, mc_mean, mc_var.mean(dim=1)

def train_uncertainty_unet():
    """Main training function"""
    print("🧠 Training Uncertainty-Aware UNet on BraTS Data")
    print("=" * 50)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset setup
    print("📊 Loading BraTS dataset...")
    train_dataset = BraTSDataset(
        "data/BraTS2021/BraTS2021_Training_Data",
        max_cases=10  # Start with 10 cases for quick training
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Model setup
    print("🏗️ Creating uncertainty UNet...")
    model = UncertaintyUNet3D(
        in_channels=4, 
        n_classes=4, 
        base_channels=16,  # Small for faster training
        dropout_rate=0.3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    print("\n🚀 Starting training...")
    
    num_epochs = 5  # Quick training for testing
    train_losses = []
    dice_scores = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_dice = 0
        num_batches = 0
        
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Resize for faster training
            images = torch.nn.functional.interpolate(
                images, size=(64, 64, 64), mode='trilinear', align_corners=False
            )
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=(64, 64, 64), mode='nearest'
            ).squeeze(1).long()
            
            print(f"  Batch {batch_idx+1}: Processing {batch['case_id'][0]}")
            
            # Forward pass with uncertainty
            optimizer.zero_grad()
            
            total_loss, ce_loss, unc_loss, mc_mean, uncertainty = uncertainty_loss(
                model, images, masks, n_samples=3
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Calculate Dice score
            dice = dice_coefficient(mc_mean, masks)
            
            # Log progress
            epoch_loss += total_loss.item()
            epoch_dice += dice.item()
            num_batches += 1
            
            print(f"    Total Loss: {total_loss.item():.4f}, CE: {ce_loss.item():.4f}, "
                  f"Unc: {unc_loss.item():.4f}, Dice: {dice.item():.4f}")
            print(f"    Uncertainty stats - Mean: {uncertainty.mean():.4f}, "
                  f"Max: {uncertainty.max():.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        
        train_losses.append(avg_loss)
        dice_scores.append(avg_dice)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Dice: {avg_dice:.4f}")
        print("-" * 40)
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'dice_scores': dice_scores,
        'config': {
            'in_channels': 4,
            'n_classes': 4,
            'base_channels': 16,
            'dropout_rate': 0.3
        }
    }, "checkpoints/uncertainty_unet.pth")
    
    print("✅ Model saved to checkpoints/uncertainty_unet.pth")
    
    # Plot training curves
    print("📈 Creating training plots...")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(dice_scores, 'g-', label='Dice Score')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("📊 Training curves saved as training_curves.png")
    
    # Test uncertainty inference
    print("\n🎲 Testing uncertainty inference...")
    test_uncertainty_inference(model, train_loader, device)
    
    print("\n🎉 Training completed successfully!")
    
    return model, train_losses, dice_scores

def test_uncertainty_inference(model, dataloader, device):
    """Test uncertainty inference on one sample"""
    
    # Get one batch
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        case_id = batch['case_id'][0]
        
        # Resize
        images = torch.nn.functional.interpolate(
            images, size=(64, 64, 64), mode='trilinear', align_corners=False
        )
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1).float(), size=(64, 64, 64), mode='nearest'
        ).squeeze(1).long()
        
        print(f"🔍 Testing uncertainty inference on {case_id}")
        
        # Monte Carlo inference
        model.train()  # Enable dropout
        n_samples = 10
        predictions = []
        
        with torch.no_grad():
            for i in range(n_samples):
                pred = model(images)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)  # Average across classes
        
        print(f"  Mean prediction shape: {mean_pred.shape}")
        print(f"  Uncertainty shape: {uncertainty.shape}")
        print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
        print(f"  Mean uncertainty: {uncertainty.mean():.4f}")
        
        # Calculate prediction accuracy
        pred_classes = torch.argmax(mean_pred, dim=1)
        accuracy = (pred_classes == masks).float().mean()
        print(f"  Prediction accuracy: {accuracy:.4f}")
        
        break  # Just test one sample
    
    print("✅ Uncertainty inference test completed!")

if __name__ == "__main__":
    model, losses, dice_scores = train_uncertainty_unet()