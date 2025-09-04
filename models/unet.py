"""
Simple 3D UNet for brain tumor segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    """3D Convolution block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    """Simple 3D UNet for brain tumor segmentation"""
    
    def __init__(self, in_channels=4, n_classes=4, base_channels=32):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 16)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 
                                         kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 
                                         kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 
                                         kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 
                                         kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)
        
        # Final classification layer
        self.final_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.enc1(x)  # Skip connection 1
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)  # Skip connection 2
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)  # Skip connection 3
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)  # Skip connection 4
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Concatenate skip connection
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

# Test the model
if __name__ == "__main__":
    # Create model
    model = UNet3D(in_channels=4, n_classes=4, base_channels=32)
    
    # Test with dummy data
    batch_size = 1
    input_tensor = torch.randn(batch_size, 4, 64, 64, 64)  # Small size for testing
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device available: {device}")
    
    if torch.cuda.is_available():
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"GPU test successful! Output shape: {output.shape}")
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")