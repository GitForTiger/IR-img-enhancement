import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

# NEURAL NETWORK ARCHITECTURE

class UNetDown(nn.Module):
    """Downsampling block for the U-Net architecture."""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """Upsampling block utilizing Bilinear interpolation to prevent grid artifacts."""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """The core Pix2Pix Generator model translating 1-channel IR to 3-channel RGB."""
    def __init__(self, in_channels=1, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Decoder with skip connections
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


# OOP PRODUCTION WRAPPER

class SatelliteColorizer:
    """
    High-level interface for deploying the GeneratorUNet in production.
    Abstracts away device mapping, weight loading, and tensor manipulation.
    """
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initializing SatelliteColorizer on: {self.device}")
        
        # Initialize architecture and load pre-trained weights
        self.model = GeneratorUNet(in_channels=1, out_channels=3)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            logger.info("Successfully loaded production weights.")
        except Exception as e:
            logger.error(f"Failed to load weights from {model_path}. Error: {e}")
            raise
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Executes a forward pass on the input IR tensor.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed float32 tensor of shape (1, 1, H, W) 
                                         scaled between [-1, 1].
        Returns:
            np.ndarray: An RGB image array of shape (H, W, 3) scaled between [0, 1] 
                        ready for visualization.
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            
            # Squeeze out batch dimensions and move back to CPU
            output_array = output.squeeze().cpu().numpy()
            
            # Denormalize from [-1, 1] to [0, 1] for visual rendering
            output_array = (output_array + 1) / 2.0
            
            # Rearrange dimensions from (Channels, Height, Width) to (Height, Width, Channels)
            output_rgb = np.clip(output_array.transpose(1, 2, 0), 0, 1)
            
            return output_rgb