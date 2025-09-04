import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, RandCropByPosNegLabeld, ToTensord
from tqdm import tqdm
import numpy as np
import os
import sys
import gc
import time
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.uncertainty_unet import UncertaintyUNet3D
from data.dataset import BraTSDataset

# ---------- Config ----------
NUM_CLASSES = 4
BATCH_SIZE = 1
EPOCHS = 100
MC_SAMPLES = 5
LEARNING_RATE = 1e-4
PATCH_SIZE = (96, 96, 96)
GRADIENT_ACCUMULATION_STEPS = 2
USE_MIXED_PRECISION = True

# Validation frequency
VAL_EVERY = 5
SAVE_EVERY = 10
LOG_EVERY = 10

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Clear cache
torch.cuda.empty_cache()
gc.collect()

print(f"Using device: {device}")

# ---------- Model ----------
model = UncertaintyUNet3D(
    in_channels=4, 
    n_classes=NUM_CLASSES, 
    base_channels=16,
    dropout_rate=0.3,
    use_checkpointing=True
).to(device)

model_size = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model parameters: {model_size:.2f}M")

# ---------- Loss and Metrics ----------
loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    reduction="mean"
)

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean_batch",
    get_not_nans=False
)

# ---------- Optimizer and Scheduler ----------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Mixed precision
scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None

# ---------- Data ----------
print("Loading data...")

# Define transforms
train_transforms = Compose([
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,  # (96, 96, 96)
        pos=1,  # 1 positive sample (with tumor)
        neg=1,  # 1 negative sample (without tumor)
        num_samples=2,  # Total 2 patches per volume
        image_key="image",
        image_threshold=0,
        allow_smaller=True
    ),
    ToTensord(keys=["image", "label"])
])

val_transforms = Compose([
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=1,
        neg=0,  # Only positive samples for validation
        num_samples=1,
        image_key="image", 
        image_threshold=0,
        allow_smaller=True
    ),
    ToTensord(keys=["image", "label"])
])

# Create datasets with transforms
train_dataset = BraTSDataset(
    data_dir="src/data/BraTS2021/BraTS2021_Training_Data",
    transforms=train_transforms
)

val_dataset = BraTSDataset(
    data_dir="src/data/BraTS2021/BraTS2021_Training_Data", 
    transforms=val_transforms
)

# Split dataset
total_cases = len(train_dataset.cases)
train_cases = int(0.8 * total_cases)

train_dataset.cases = train_dataset.cases[:train_cases]
val_dataset.cases = val_dataset.cases[train_cases:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print("Note: Each training volume generates 2 patches (1 pos + 1 neg)")

# ---------- Training tracking ----------
train_metrics = defaultdict(list)
val_metrics = defaultdict(list)
best_dice = 0.0

# Create output directory
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ---------- Training Loop ----------
print("\nStarting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Training Phase
    model.train()
    train_loss = 0
    train_dice_scores = []
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(train_pbar):
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask'].to(device, non_blocking=True)
        
        if len(y.shape) == 4:
            y = y.unsqueeze(1)
        
        # Forward pass
        if USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = loss_fn(outputs, y) / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(x)
            loss = loss_fn(outputs, y) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Calculate training dice score
        with torch.no_grad():
            pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            dice_metric(pred_labels, y)
            batch_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            train_dice_scores.append(batch_dice)
        
        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        # Update progress bar
        if batch_idx % LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({
                'Loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
                'Dice': f'{batch_dice:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Memory cleanup
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    # Calculate epoch metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = np.mean(train_dice_scores)
    
    train_metrics['loss'].append(avg_train_loss)
    train_metrics['dice'].append(avg_train_dice)
    
    # Update learning rate
    scheduler.step()
    
    # Validation Phase
    if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
        model.eval()
        val_loss = 0
        val_dice_scores = []
        val_uncertainties = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                x = batch['image'].to(device, non_blocking=True)
                y = batch['mask'].to(device, non_blocking=True)
                
                if len(y.shape) == 4:
                    y = y.unsqueeze(1)
                
                # Standard prediction for loss
                if USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(x)
                        loss = loss_fn(outputs, y)
                else:
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                
                val_loss += loss.item()
                
                # Dice score
                pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                dice_metric(pred_labels, y)
                batch_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                val_dice_scores.append(batch_dice)
                
                # Uncertainty estimation
                model.apply(lambda m: m.train() if isinstance(m, (nn.Dropout, nn.Dropout3d)) else None)
                uncertainty_preds = []
                for _ in range(3):
                    mc_out = model(x)
                    uncertainty_preds.append(torch.softmax(mc_out, dim=1))
                
                if uncertainty_preds:
                    uncertainty_preds = torch.stack(uncertainty_preds)
                    pred_var = uncertainty_preds.var(dim=0).mean().item()
                    val_uncertainties.append(pred_var)
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{batch_dice:.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = np.mean(val_dice_scores)
        avg_val_uncertainty = np.mean(val_uncertainties) if val_uncertainties else 0
        
        val_metrics['loss'].append(avg_val_loss)
        val_metrics['dice'].append(avg_val_dice)
        val_metrics['uncertainty'].append(avg_val_uncertainty)
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, 'checkpoints/best_model.pth')
    
    # Epoch summary
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
    if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, Uncertainty: {avg_val_uncertainty:.4f}")
        print(f"  Best Dice: {best_dice:.4f}")
    print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Save checkpoint
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
    
    # Plot metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(train_metrics['loss'], label='Train Loss', color='blue')
        if val_metrics['loss']:
            val_epochs = list(range(0, epoch+1, VAL_EVERY))
            if 0 not in val_epochs:
                val_epochs = [0] + val_epochs
            plt.plot(val_epochs[:len(val_metrics['loss'])], val_metrics['loss'], 
                    label='Val Loss', color='red', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress - Loss')
        plt.legend()
        plt.grid(True)
        
        # Dice plot
        plt.subplot(1, 3, 2)
        plt.plot(train_metrics['dice'], label='Train Dice', color='blue')
        if val_metrics['dice']:
            val_epochs = list(range(0, epoch+1, VAL_EVERY))
            if 0 not in val_epochs:
                val_epochs = [0] + val_epochs
            plt.plot(val_epochs[:len(val_metrics['dice'])], val_metrics['dice'], 
                    label='Val Dice', color='red', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Training Progress - Dice Score')
        plt.legend()
        plt.grid(True)
        
        # Uncertainty plot
        plt.subplot(1, 3, 3)
        if val_metrics['uncertainty']:
            val_epochs = list(range(0, epoch+1, VAL_EVERY))
            if 0 not in val_epochs:
                val_epochs = [0] + val_epochs
            plt.plot(val_epochs[:len(val_metrics['uncertainty'])], val_metrics['uncertainty'], 
                    label='Validation Uncertainty', color='green', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'outputs/training_progress_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
        plt.close()

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time/3600:.2f} hours")
print(f"Best validation Dice score: {best_dice:.4f}")

# Final save
torch.save({
    'epoch': EPOCHS-1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_dice': best_dice,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'total_time': total_time
}, 'checkpoints/final_model.pth')

print("Training completed!")