import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """A convolutional block with BatchNorm, ReLU, and SEBlock."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        x = self.block(x)
        x = self.se_block(x)
        return x

class UNet(nn.Module):
    """U-Net model with SE blocks for image dehazing."""
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for feature in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(feature*2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid() # Use Sigmoid to output values in [0, 1]

    def forward(self, x):
        skip_connections = []
        # Downsample
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsample
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.decoder)):
            x = self.upsamples[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape[2:] != skip_connection.shape[2:]:
                # Use F.interpolate for more flexible resizing
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](concat_skip)
            
        return self.activation(self.final_conv(x))