import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import numpy as np
import os
import sys
import gc
import time
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.nnunet_uncertainty import nnUNetUncertainty
from src.data.dataset import BraTSDataset

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint and resume training"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        

        train_metrics = checkpoint.get('train_metrics', defaultdict(list))
        val_metrics = checkpoint.get('val_metrics', defaultdict(list))
        best_dice = checkpoint.get('best_dice', 0.0)
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
        return start_epoch, train_metrics, val_metrics, best_dice
    else:
        print("No checkpoint found, starting from scratch")
        return 0, defaultdict(list), defaultdict(list), 0.0

# ---------- Config ----------
NUM_CLASSES = 4
BATCH_SIZE = 8
EPOCHS = 50
MC_SAMPLES = 10
LEARNING_RATE = 1e-4
PATCH_SIZE = (96, 96, 96)
GRADIENT_ACCUMULATION_STEPS = 2
USE_MIXED_PRECISION = True

# Checkpoint configuration
RESUME_FROM_CHECKPOINT = None  
# RESUME_FROM_CHECKPOINT = "checkpoints/epoch_10.pth" 

# Validation frequency
VAL_EVERY = 5
SAVE_EVERY = 1
LOG_EVERY = 5

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Clear cache
torch.cuda.empty_cache()
gc.collect()

print(f"Using device: {device}")

# ---------- Model (MONAI UNet) ----------
model = nnUNetUncertainty(
    input_channels=4,
    num_classes=NUM_CLASSES,
    dropout_p=0.3
).to(device)

model_size = sum(p.numel() for p in model.parameters()) / 1e6
print(f"MONAI UNet model parameters: {model_size:.2f}M")

# ---------- Loss and Metrics ----------
loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=True,
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

train_dataset = BraTSDataset(
    data_dir="src/data/BraTS2021/BraTS2021_Training_Data",
    patch_size=PATCH_SIZE
)

val_dataset = BraTSDataset(
    data_dir="src/data/BraTS2021/BraTS2021_Training_Data",
    patch_size=PATCH_SIZE
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

# Create output directory
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ---------- Load checkpoint or start fresh ----------
start_epoch, train_metrics, val_metrics, best_dice = load_checkpoint(
    model, optimizer, scheduler, RESUME_FROM_CHECKPOINT
)

print("Folders created successfully")

# ---------- Training Loop ----------
print(f"\nStarting MONAI UNet uncertainty training from epoch {start_epoch+1}...")
start_time = time.time()

for epoch in range(start_epoch, EPOCHS):
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
    
    # Save checkpoint based on SAVE_EVERY hyperparameter
    if (epoch + 1) % SAVE_EVERY == 0:
        try:
            checkpoint_path = f'checkpoints/epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'train_dice': avg_train_dice,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_dice': best_dice
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    # Validation Phase
    avg_val_loss = 0
    avg_val_dice = 0
    avg_epistemic = 0
    avg_aleatoric = 0
    
    if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
        print("Running validation...")
        model.eval()
        val_loss = 0
        val_dice_scores = []
        val_uncertainties = {'epistemic': [], 'aleatoric': [], 'predictive': []}
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
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
                    
                    # Uncertainty estimation (only on a few batches to save time)
                    if batch_idx < 10:
                        uncertainty_results = model.monte_carlo_forward(x, n_samples=5)
                        
                        val_uncertainties['epistemic'].append(
                            uncertainty_results['epistemic_uncertainty'].mean().item()
                        )
                        val_uncertainties['aleatoric'].append(
                            uncertainty_results['aleatoric_uncertainty'].mean().item()
                        )
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}', 
                        'Dice': f'{batch_dice:.4f}'
                    })
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = np.mean(val_dice_scores)
            avg_epistemic = np.mean(val_uncertainties['epistemic']) if val_uncertainties['epistemic'] else 0
            avg_aleatoric = np.mean(val_uncertainties['aleatoric']) if val_uncertainties['aleatoric'] else 0
            
            val_metrics['loss'].append(avg_val_loss)
            val_metrics['dice'].append(avg_val_dice)
            val_metrics['epistemic_uncertainty'].append(avg_epistemic)
            val_metrics['aleatoric_uncertainty'].append(avg_aleatoric)
            
            # Save best model
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                try:
                    best_model_path = 'checkpoints/best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_dice': best_dice,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    }, best_model_path)
                    print(f"New best model saved: {best_model_path} (Dice: {best_dice:.4f})")
                except Exception as e:
                    print(f"Error saving best model: {e}")
                    
        except Exception as e:
            print(f"Error during validation: {e}")
    
    # Epoch summary
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
    if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
        print(f"  Uncertainties - Epistemic: {avg_epistemic:.4f}, Aleatoric: {avg_aleatoric:.4f}")
        print(f"  Best Dice: {best_dice:.4f}")
    print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Create simple plot every 5 epochs
    if (epoch + 1) % 5 == 0:
        try:
            plt.figure(figsize=(10, 4))
            
            # Loss plot
            plt.subplot(1, 2, 1)
            epochs_list = list(range(1, len(train_metrics['loss']) + 1))
            plt.plot(epochs_list, train_metrics['loss'], label='Train Loss', color='blue')
            if val_metrics['loss']:
                val_epochs_list = []
                for i in range(len(val_metrics['loss'])):
                    val_epochs_list.append((i * VAL_EVERY) + 1)
                plt.plot(val_epochs_list, val_metrics['loss'], label='Val Loss', color='red', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()
            plt.grid(True)
            
            # Dice plot
            plt.subplot(1, 2, 2)
            plt.plot(epochs_list, train_metrics['dice'], label='Train Dice', color='blue')
            if val_metrics['dice']:
                plt.plot(val_epochs_list, val_metrics['dice'], label='Val Dice', color='red', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.title('Dice Score')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = f'outputs/progress_epoch_{epoch+1}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved: {plot_path}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")

# Final save
try:
    final_path = 'checkpoints/final_model.pth'
    torch.save({
        'epoch': EPOCHS-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_dice': best_dice,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'total_time': time.time() - start_time
    }, final_path)
    print(f"Final model saved: {final_path}")
except Exception as e:
    print(f"Error saving final model: {e}")

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time/3600:.2f} hours")
print(f"Best validation Dice score: {best_dice:.4f}")
print("MONAI UNet uncertainty training completed!")