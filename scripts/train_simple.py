import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.unet import UNet3D
from data.dataset import BraTSDataset

def dice_loss(pred, target):
    """Simple Dice loss"""
    smooth = 1e-6
    pred = torch.softmax(pred, dim=1)
    
    # Convert target to one-hot
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    intersection = (pred * target_one_hot).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target_one_hot.sum() + smooth)
    
    return 1 - dice

def test_training():
    """Test training on a few samples"""
    print("🧠 Testing Brain Tumor UNet Training")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = BraTSDataset(
        "data/BraTS2021/BraTS2021_Training_Data",
        max_cases=3  # Just 3 cases for quick test
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = UNet3D(in_channels=4, n_classes=4, base_channels=16)  # Smaller for testing
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    
    for epoch in range(2):  # Just 2 epochs
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)  # (1, 4, 240, 240, 155)
            masks = batch['mask'].to(device)   # (1, 240, 240, 155)
            
            # Resize for faster testing
            images = torch.nn.functional.interpolate(images, size=(64, 64, 64), mode='trilinear')
            masks = torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=(64, 64, 64), mode='nearest').squeeze(1).long()
            
            print(f"Batch {batch_idx}: Image {images.shape}, Mask {masks.shape}")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)  # (1, 4, 64, 64, 64)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            print(f"  Loss: {loss.item():.4f}")
            
            # Quick validation
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                unique_pred = torch.unique(pred)
                unique_true = torch.unique(masks)
                
                print(f"  Predicted classes: {unique_pred.cpu().numpy()}")
                print(f"  True classes: {unique_true.cpu().numpy()}")
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss/len(dataloader):.4f}")
        print("-" * 30)
    
    print("✅ Training test completed successfully!")
    
    # Save test checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "checkpoints/test_checkpoint.pth")
    
    print("💾 Test checkpoint saved to checkpoints/test_checkpoint.pth")

if __name__ == "__main__":
    test_training()