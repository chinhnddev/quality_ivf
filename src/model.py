import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# DropPath (Stochastic Depth)
# -------------------------
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    Paper: "Deep Networks with Stochastic Depth" https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        output = x.div(keep_prob) * random_tensor
        return output


# -------------------------
# Attention blocks
# -------------------------
class SimAM(nn.Module):
    """
    Simple Attention Module
    Paper: https://arxiv.org/abs/2106.03105
    """
    def __init__(self, channels: int):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mu).pow(2)
        # epsilon 1e-6 để tránh numerical instability
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + 1e-6)) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
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
        """
        Adaptive kernel size theo ECA paper
        k = |log2(C) / γ + b / γ|_odd
        """
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 else t + 1  # đảm bảo lẻ
        return max(3, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)   # B,1,C
        y = self.conv(y).view(b, c, 1, 1)   # B,C,1,1
        return x * self.sigmoid(y)


# -------------------------
# Multi-scale block
# -------------------------
class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block với 3 branches (3x3, 5x5, 7x7)
    + DropPath regularization
    """
    def __init__(self, in_channels: int, out_channels: int, drop_path: float = 0.0):
        super().__init__()
        c = out_channels // 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c, 5, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - 2 * c, 7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels - 2 * c),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.simam = SimAM(out_channels)

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_res = in_channels == out_channels
        if not self.use_res:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.use_res else self.skip(x)

        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        out = self.proj(out)
        out = self.simam(out)

        # Apply DropPath on residual branch
        out = identity + self.drop_path(out)
        return out


# -------------------------
# Depthwise block with residual
# -------------------------
class DWConvBlock(nn.Module):
    """
    Depthwise Separable Convolution với residual connection
    + DropPath regularization
    Hỗ trợ stride=1/2, dilation>=1
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dilation: int = 1, drop_path: float = 0.0):
        super().__init__()

        self.dw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
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

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.dw(x)
        out = self.pw(out)
        out = self.simam(out)

        if self.use_res:
            # Apply DropPath on residual branch
            out = identity + self.drop_path(out)
        else:
            out = out

        return out


# -------------------------
# GeM Pooling
# -------------------------
class GeM(nn.Module):
    """Generalized Mean Pooling - learnable pooling exponent"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


# -------------------------
# IVF_EffiMorphPP
# -------------------------
class IVF_EffiMorphPP(nn.Module):
    """
    IVF_EffiMorphPP with DropPath (Stochastic Depth)
    - Dynamic interpolation size (no hardcode 7x7)
    - Adaptive ECA kernel size
    - GeM Pooling
    - CORAL optional (EXP -> K-1 logits)
    - DropPath regularization với linear decay
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
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.use_coral = use_coral

        def make_divisible(v: float, d: int = 8) -> int:
            return int((v + d / 2) // d * d)

        base = make_divisible(base_channels * width_mult, divisor)
        c1 = make_divisible(2 * base, divisor)
        c2 = make_divisible(4 * base, divisor)
        c3 = make_divisible(8 * base, divisor)
        c4 = make_divisible(16 * base, divisor)

        # Stochastic Depth: Linear decay rule
        # Tăng dần từ 0 (stage1) đến drop_path_rate (stage4)
        num_blocks = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        # Stem: /2
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Stage 1 (no downsample)
        self.stage1 = MultiScaleBlock(base, c1, drop_path=dpr[0])

        # /2
        self.stage1_down = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Stage 2: /2
        self.stage2 = DWConvBlock(c1, c2, stride=2, dilation=1, drop_path=dpr[1])

        # Stage 3: /2
        self.stage3 = DWConvBlock(c2, c3, stride=2, dilation=2, drop_path=dpr[2])

        # Stage 4: /2
        self.stage4 = DWConvBlock(c3, c4, stride=2, dilation=1, drop_path=dpr[3])

        # Fusion: concat(s2,s3,s4)->c4
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(c4, k_size=None)

        hidden = max(128, c4 // 2)
        self.gap = GeM(p=3.0, eps=1e-6)
        self.dropout = nn.Dropout(dropout_p)

        output_dim = num_classes - 1 if (task == "exp" and use_coral) else num_classes
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage1_down(x)

        s2 = self.stage2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        target_size = s4.shape[2:]
        s2_up = F.interpolate(s2, size=target_size, mode="bilinear", align_corners=False)
        s3_up = F.interpolate(s3, size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([s2_up, s3_up, s4], dim=1)
        fused = self.fusion(fused)
        fused = self.eca(fused)

        x = self.gap(fused).flatten(1)
        x = self.dropout(x)
        return self.head(x)