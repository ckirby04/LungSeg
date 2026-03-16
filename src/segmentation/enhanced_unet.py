"""
Enhanced 3D U-Net with Deep Supervision and Improved Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.15):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = self.relu(out)
        return out


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.15):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.15, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate3D(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.conv = ResidualConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DeepSupervisedUNet3D(nn.Module):
    """
    3D U-Net with Deep Supervision for small lesion detection

    Args:
        in_channels: Number of input channels (default: 1 for single-channel CT)
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, depth=4,
                 dropout_p=0.15, deep_supervision=True):
        super().__init__()
        self.depth = depth
        self.deep_supervision = deep_supervision
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.inc = ResidualConvBlock(in_channels, channels[0], dropout_p)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p) for i in range(depth)
        ])
        self.bottleneck = ResidualConvBlock(channels[depth], channels[depth], dropout_p)
        self.up_blocks = nn.ModuleList([
            UpBlock(channels[i+1], channels[i], dropout_p, use_attention=True)
            for i in range(depth-1, -1, -1)
        ])
        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(channels[depth-1-i], out_channels, kernel_size=1) for i in range(depth)
            ])

    def forward(self, x, return_ds=None):
        if return_ds is None:
            return_ds = self.deep_supervision
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        ds_outputs = []
        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i+2)]
            x = up(x, skip)
            if self.deep_supervision and return_ds and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))
        main_output = self.outc(x)
        if self.deep_supervision and return_ds:
            return [main_output] + ds_outputs
        return main_output


class HybridUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, depth=4, dropout_p=0.15):
        super().__init__()
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.inc = ResidualConvBlock(in_channels, channels[0], dropout_p)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p) for i in range(depth)
        ])
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(channels[depth], channels[depth], dropout_p),
            SqueezeExcitation3D(channels[depth])
        )
        self.up_blocks = nn.ModuleList([
            UpBlock(channels[i+1], channels[i], dropout_p, use_attention=True)
            for i in range(depth-1, -1, -1)
        ])
        self.outc = nn.Sequential(
            nn.Conv3d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[0], out_channels, kernel_size=1)
        )
        self.ds_heads = nn.ModuleList([
            nn.Conv3d(channels[depth-1-i], out_channels, kernel_size=1) for i in range(depth)
        ])

    def forward(self, x, return_ds=True):
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        ds_outputs = []
        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i+2)]
            x = up(x, skip)
            if return_ds and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))
        main_output = self.outc(x)
        if return_ds and self.training:
            return [main_output] + ds_outputs
        return main_output


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing DeepSupervisedUNet3D...")
    model = DeepSupervisedUNet3D(in_channels=1, base_channels=16, depth=4, deep_supervision=True).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    with torch.no_grad():
        outputs = model(x, return_ds=True)
        print(f"Outputs: {len(outputs)}, main: {outputs[0].shape}")
    print("Models tested successfully!")
