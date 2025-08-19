"""
UNet with Monte Carlo Dropout for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyConvBlock3D(nn.Module):
    """3D Convolution block with dropout for uncertainty"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout2 = nn.Dropout3d(dropout_rate)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        return x

class UncertaintyUNet3D(nn.Module):
    """3D UNet with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, in_channels=4, n_classes=4, base_channels=32, dropout_rate=0.3):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        # Encoder with dropout
        self.enc1 = UncertaintyConvBlock3D(in_channels, base_channels, dropout_rate)
        self.enc2 = UncertaintyConvBlock3D(base_channels, base_channels * 2, dropout_rate)
        self.enc3 = UncertaintyConvBlock3D(base_channels * 2, base_channels * 4, dropout_rate)
        self.enc4 = UncertaintyConvBlock3D(base_channels * 4, base_channels * 8, dropout_rate)
        
        # Bottleneck with dropout
        self.bottleneck = UncertaintyConvBlock3D(base_channels * 8, base_channels * 16, dropout_rate)
        
        # Decoder with dropout
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 
                                         kernel_size=2, stride=2)
        self.dec4 = UncertaintyConvBlock3D(base_channels * 16, base_channels * 8, dropout_rate)
        
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 
                                         kernel_size=2, stride=2)
        self.dec3 = UncertaintyConvBlock3D(base_channels * 8, base_channels * 4, dropout_rate)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 
                                         kernel_size=2, stride=2)
        self.dec2 = UncertaintyConvBlock3D(base_channels * 4, base_channels * 2, dropout_rate)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 
                                         kernel_size=2, stride=2)
        self.dec1 = UncertaintyConvBlock3D(base_channels * 2, base_channels, dropout_rate)
        
        # Final classification layer
        self.final_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """Standard forward pass"""
        # Encoder path
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x
    
    def monte_carlo_forward(self, x, n_samples=20):
        """
        Monte Carlo sampling for uncertainty estimation
        """
        # Enable dropout during inference
        self.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        # Stack predictions: (n_samples, batch, classes, H, W, D)
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate statistics
        mean_prediction = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        # Calculate predictive entropy (uncertainty)
        probs = F.softmax(predictions, dim=2)
        mean_probs = probs.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1, keepdim=True)
        
        return {
            'prediction': mean_prediction,
            'uncertainty': entropy,
            'variance': variance.mean(dim=1, keepdim=True),
            'samples': predictions
        }

# Test the uncertainty model
if __name__ == "__main__":
    print("🎯 Testing Uncertainty UNet")
    print("=" * 30)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = UncertaintyUNet3D(in_channels=4, n_classes=4, base_channels=16, dropout_rate=0.3)
    model = model.to(device)
    
    # Test input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 4, 32, 32, 32).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test standard forward
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Standard output shape: {output.shape}")
    
    # Test Monte Carlo forward
    print("\n🎲 Testing Monte Carlo sampling...")
    uncertainty_output = model.monte_carlo_forward(input_tensor, n_samples=5)
    
    print(f"Prediction shape: {uncertainty_output['prediction'].shape}")
    print(f"Uncertainty shape: {uncertainty_output['uncertainty'].shape}")
    print(f"Samples shape: {uncertainty_output['samples'].shape}")
    
    # Uncertainty statistics
    uncertainty = uncertainty_output['uncertainty']
    print(f"\nUncertainty stats:")
    print(f"  Mean: {uncertainty.mean():.4f}")
    print(f"  Min: {uncertainty.min():.4f}")
    print(f"  Max: {uncertainty.max():.4f}")
    
    print("\n✅ Uncertainty UNet test completed!")