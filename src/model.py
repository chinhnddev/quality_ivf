"""
MINIMAL CHANGE: Add auxiliary head to your 83% baseline
Keep EVERYTHING else identical
[FIXED VERSION]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
# ATTENTION BLOCKS (UNCHANGED)
# ═══════════════════════════════════════════════════════════════

class SimAM(nn.Module):
    """Simple Attention Module"""
    def __init__(self, channels: int):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mu).pow(2)
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + 1e-6)) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels: int, k_size: int | None = None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if k_size is None:
            k_size = self._get_adaptive_kernel_size(channels)

        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _get_adaptive_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 else t + 1
        return max(3, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS (UNCHANGED)
# ═══════════════════════════════════════════════════════════════

class MultiScaleBlock(nn.Module):
    """Multi-scale convolution block (dilation version)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = out_channels // 3
        c3 = out_channels - 2 * c

        def conv3x3_bn_relu(in_c: int, out_c: int, dilation: int):
            return nn.Sequential(
                nn.Conv2d(
                    in_c, out_c,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.branch1 = conv3x3_bn_relu(in_channels, c, dilation=1)
        self.branch2 = conv3x3_bn_relu(in_channels, c, dilation=2)
        self.branch3 = conv3x3_bn_relu(in_channels, c3, dilation=3)

        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.simam = SimAM(out_channels)

        self.use_res = in_channels == out_channels
        if not self.use_res:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        x = self.proj(x)
        x = self.simam(x)
        if self.use_res:
            return x + identity
        return x + self.skip(identity)


class DWConvBlock(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        self.dw = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.simam = SimAM(out_channels)
        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.simam(x)
        if self.use_res:
            x = x + identity
        return x


class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


# ═══════════════════════════════════════════════════════════════
# MODEL: BASELINE + AUXILIARY HEAD ONLY (FIXED!)
# ═══════════════════════════════════════════════════════════════

class IVF_EffiMorphPP(nn.Module):
    """
    BASELINE + AUXILIARY HEAD
    
    CHANGES FROM 83% BASELINE:
      ✓ Added auxiliary classifier at stage3
      ✓ Fixed fusion interpolation
      ✓ NOTHING ELSE changed!
    """
    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.3,
        width_mult: float = 1.0,
        base_channels: int = 32,
        divisor: int = 8,
        task: str = "exp",
        use_coral: bool = False,
        use_aux: bool = True,
    ):
        super().__init__()
        self.task = task
        self.use_coral = use_coral
        self.use_aux = use_aux

        def make_divisible(v: float, d: int = 8) -> int:
            return int((v + d / 2) // d * d)

        base = make_divisible(base_channels * width_mult, divisor)
        c1 = make_divisible(2 * base, divisor)
        c2 = make_divisible(4 * base, divisor)
        c3 = make_divisible(8 * base, divisor)
        c4 = make_divisible(16 * base, divisor)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stage1 = MultiScaleBlock(base, c1)
        self.stage1_down = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage2 = DWConvBlock(c1, c2, stride=2, dilation=1)
        self.stage3 = DWConvBlock(c2, c3, stride=1, dilation=2)
        self.stage4 = DWConvBlock(c3, c4, stride=1, dilation=4)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(c4, k_size=None)

        # Main Head
        hidden = max(128, c4 // 2)
        self.gap = GeM(p=3.0, eps=1e-6)
        self.dropout = nn.Dropout(dropout_p)

        output_dim = num_classes - 1 if use_coral else num_classes
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden, output_dim),
        )

        # Auxiliary Head
        if use_aux:
            self.aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout_p * 0.3),
                nn.Linear(c3, output_dim),
            )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Forward pass with optional auxiliary output
        
        Args:
            x: Input [B, 3, H, W]
            return_aux: If True and training, return (main_out, aux_out)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage1_down(x)

        s2 = self.stage2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # ════════════════════════════════════════════════════════
        # Auxiliary output from stage3 (before fusion)
        # ════════════════════════════════════════════════════════
        if return_aux and self.use_aux and self.training:
            aux_out = self.aux_head(s3)

        # ════════════════════════════════════════════════════════
        # FIXED: Interpolate before fusion
        # ════════════════════════════════════════════════════════
        target_size = s4.shape[2:]  # Get s4 spatial size [14, 14]
        
        # Upsample s2 and s3 to match s4
        s2_up = F.interpolate(s2, size=target_size, mode='bilinear', align_corners=False)
        s3_up = F.interpolate(s3, size=target_size, mode='bilinear', align_corners=False)
        
        # Concat (now all same size)
        fused = torch.cat([s2_up, s3_up, s4], dim=1)
        fused = self.fusion(fused)
        fused = self.eca(fused)

        # Main output
        x = self.gap(fused).flatten(1)
        x = self.dropout(x)
        main_out = self.head(x)

        # Return
        if return_aux and self.use_aux and self.training:
            return main_out, aux_out
        return main_out