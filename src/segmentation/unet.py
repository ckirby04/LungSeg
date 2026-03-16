"""
Lightweight 3D U-Net for 3D medical image segmentation
Optimized for consumer GPUs with limited VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

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


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


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


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate3D(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        skip_att = self.attention(g=x, x=skip)
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv(x)
        return x


class LightweightUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for medical image segmentation

    Args:
        in_channels: Number of input channels (default: 1 for single-channel CT)
        out_channels: Number of output classes (default: 1 for binary segmentation)
        base_channels: Base number of feature channels (default: 16)
        depth: Number of downsampling levels (default: 3)
        dropout_p: Dropout probability (default: 0.1)
        use_attention: Use attention gates in decoder (default: False)
        use_residual: Use residual connections in conv blocks (default: False)
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, depth=3,
                 dropout_p=0.1, use_attention=False, use_residual=False, deep_supervision=False):
        super().__init__()
        self.depth = depth
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.deep_supervision = deep_supervision
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        ConvBlockType = ResidualConvBlock if use_residual else ConvBlock
        self.inc = ConvBlockType(in_channels, channels[0], dropout_p)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p) for i in range(depth)
        ])
        UpBlockType = AttentionUpBlock if use_attention else UpBlock
        self.up_blocks = nn.ModuleList([
            UpBlockType(channels[i+1], channels[i], dropout_p) for i in range(depth-1, -1, -1)
        ])
        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(channels[depth - 1 - i], out_channels, kernel_size=1) for i in range(depth)
            ])

    def forward(self, x):
        x1 = self.inc(x)
        skip_connections = [x1]
        x = x1
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        skip_connections = skip_connections[:-1][::-1]
        aux_outputs = []
        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[i])
            if self.deep_supervision:
                aux_outputs.append(self.ds_heads[i](x))
        x = self.outc(x)
        if self.deep_supervision:
            return x, aux_outputs
        return x


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return (1 - tversky_index) ** self.gamma


class EnhancedCombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.4, focal_tversky_weight=0.4, bce_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal_tversky = self.focal_tversky_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.focal_tversky_weight * focal_tversky + self.bce_weight * bce


class BoundaryLoss(nn.Module):
    def forward(self, pred, dist_map):
        pred_prob = torch.sigmoid(pred)
        return (pred_prob * dist_map).mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing LightweightUNet3D...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightUNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3).to(device)
    print(f"Total parameters: {count_parameters(model):,}")
    x = torch.randn(2, 1, 96, 96, 96).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
