# models/uncertainty_unet.py
"""
Memory-Efficient 3D UNet with Monte Carlo Dropout for Epistemic Uncertainty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

Norm3d = nn.InstanceNorm3d

class UncertaintyConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = Norm3d(out_channels)
        self.drop1 = nn.Dropout3d(dropout_rate)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = Norm3d(out_channels)
        self.drop2 = nn.Dropout3d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.drop1(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.drop2(x)
        return x

class UncertaintyUNet3D(nn.Module):
    def __init__(self, in_channels=4, n_classes=4, base_channels=16, dropout_rate=0.3, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # Encoder (smaller base channels to reduce memory)
        self.enc1 = UncertaintyConvBlock3D(in_channels, base_channels, dropout_rate)
        self.enc2 = UncertaintyConvBlock3D(base_channels, base_channels*2, dropout_rate)
        self.enc3 = UncertaintyConvBlock3D(base_channels*2, base_channels*4, dropout_rate)
        self.enc4 = UncertaintyConvBlock3D(base_channels*4, base_channels*8, dropout_rate)
        
        # Bottleneck
        self.bottleneck = UncertaintyConvBlock3D(base_channels*8, base_channels*16, dropout_rate)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(base_channels*16, base_channels*8, 2, 2, bias=False)
        self.dec4 = UncertaintyConvBlock3D(base_channels*16, base_channels*8, dropout_rate)
        
        self.upconv3 = nn.ConvTranspose3d(base_channels*8, base_channels*4, 2, 2, bias=False)
        self.dec3 = UncertaintyConvBlock3D(base_channels*8, base_channels*4, dropout_rate)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, 2, bias=False)
        self.dec2 = UncertaintyConvBlock3D(base_channels*4, base_channels*2, dropout_rate)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, 2, bias=False)
        self.dec1 = UncertaintyConvBlock3D(base_channels*2, base_channels, dropout_rate)
        
        self.final_conv = nn.Conv3d(base_channels, n_classes, 1, bias=True)
        self.pool = nn.MaxPool3d(2, 2)

    def _encoder_block(self, x, encoder, pool=True):
        """Wrapper for gradient checkpointing"""
        if self.use_checkpointing and self.training:
            x = checkpoint(encoder, x, use_reentrant=False)
        else:
            x = encoder(x)
        
        if pool:
            return x, self.pool(x)
        return x

    def _decoder_block(self, x, skip, upconv, decoder):
        """Wrapper for gradient checkpointing"""
        x = upconv(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        
        if self.use_checkpointing and self.training:
            x = checkpoint(decoder, x, use_reentrant=False)
        else:
            x = decoder(x)
        return x

    def forward(self, x):
        # Encoder path with skip connections
        e1, x = self._encoder_block(x, self.enc1)
        e2, x = self._encoder_block(x, self.enc2)
        e3, x = self._encoder_block(x, self.enc3)
        e4, x = self._encoder_block(x, self.enc4)
        
        # Bottleneck
        x = self._encoder_block(x, self.bottleneck, pool=False)

        # Decoder path
        x = self._decoder_block(x, e4, self.upconv4, self.dec4)
        x = self._decoder_block(x, e3, self.upconv3, self.dec3)
        x = self._decoder_block(x, e2, self.upconv2, self.dec2)
        x = self._decoder_block(x, e1, self.upconv1, self.dec1)

        return self.final_conv(x)

    @torch.no_grad()
    def monte_carlo_forward(self, x, n_samples=10):
        """Memory-efficient Monte Carlo forward pass"""
        self.eval()
        self.apply(self._enable_dropout_only)
        
        # Store predictions on CPU to save GPU memory
        cpu_preds = []
        
        for i in range(n_samples):
            with torch.cuda.amp.autocast():
                logits = self.forward(x)
                prob = F.softmax(logits, dim=1)
            
            cpu_preds.append(prob.cpu())
            
            # Clear GPU memory periodically
            if i % 3 == 0:
                torch.cuda.empty_cache()
        
        # Move back to GPU for final computation
        preds = torch.stack([p.to(x.device) for p in cpu_preds], dim=0)
        
        p_mean = preds.mean(0)
        
        # Compute uncertainties
        entropy = -(p_mean * p_mean.clamp_min(1e-8).log()).sum(1, keepdim=True)
        exp_entropy = -(preds * preds.clamp_min(1e-8).log()).sum(2).mean(0, keepdim=True)
        mutual_info = entropy - exp_entropy
        
        # Clean up
        del preds, cpu_preds
        torch.cuda.empty_cache()
        
        return {
            "prob": p_mean,
            "entropy": entropy,
            "mutual_info": mutual_info
        }

    def _enable_dropout_only(self, m):
        """Enable only dropout layers during MC inference"""
        if isinstance(m, (nn.Dropout, nn.Dropout3d)):
            m.train()

    def get_model_size(self):
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

# Test memory usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with different configurations
    configs = [
        {"base_channels": 8, "name": "Tiny"},
        {"base_channels": 16, "name": "Small"},
        {"base_channels": 32, "name": "Medium"},
    ]
    
    for config in configs:
        model = UncertaintyUNet3D(
            in_channels=4,
            n_classes=4,
            base_channels=config["base_channels"],
            use_checkpointing=True
        ).to(device)
        
        size_mb = model.get_model_size()
        print(f"{config['name']} model: {size_mb:.2f} MB")
        
        # Test forward pass
        try:
            x = torch.randn(1, 4, 64, 64, 64).to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
            print(f"  Forward pass successful: {out.shape}")
            
            # Test MC inference
            mc_out = model.monte_carlo_forward(x, n_samples=5)
            print(f"  MC inference successful: {mc_out['prob'].shape}")
            
        except RuntimeError as e:
            print(f"  Failed: {e}")
        
        del model
        torch.cuda.empty_cache()
        print()