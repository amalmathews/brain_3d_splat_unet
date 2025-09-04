# src/models/nnunet_uncertainty.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet

class nnUNetUncertainty(nn.Module):
    """
    MONAI UNet with Monte Carlo Dropout for uncertainty quantification
    """
    def __init__(self, 
                 input_channels=4, 
                 num_classes=4,
                 dropout_p=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        
        # MONAI UNet backbone (proven architecture)
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=num_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=dropout_p
        )
    
    def forward(self, x):
        """Forward pass through UNet backbone"""
        return self.backbone(x)
    
    def _set_dropout_mode(self):
        """Force dropout layers to stay active during inference"""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()
    
    def monte_carlo_forward(self, x, n_samples=20):
        """Monte Carlo forward pass for uncertainty estimation"""
        # Set model to eval but keep dropout active
        self.eval()
        self._set_dropout_mode()
        
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, classes, ...)
        
        # Calculate uncertainty metrics
        mean_pred = predictions.mean(dim=0)
        predictive_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1, keepdim=True)
        
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = -torch.mean(
            torch.sum(predictions * torch.log(predictions + 1e-8), dim=2), 
            dim=0, keepdim=True
        )
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        
        return {
            'predictions': mean_pred,
            'predictive_entropy': predictive_entropy,
            'aleatoric_uncertainty': expected_entropy,
            'epistemic_uncertainty': mutual_info,
            'samples': predictions
        }