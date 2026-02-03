import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# Attention blocks (UPDATED to CBAM for morphology focus)
# -------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_attn = self.channel_gate(x)
        x = x * ch_attn
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        sp_attn = self.spatial_gate(torch.cat([mean, std], dim=1))
        return x * sp_attn

class ECA(nn.Module):
    # Giữ nguyên như cũ
    def __init__(self, channels: int, k_size: int | None = None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if k_size is None:
            k_size = self._get_adaptive_kernel_size(channels)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
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

# -------------------------
# Multi-scale block (updated with CBAM)
# -------------------------
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = out_channels // 3
        c3 = out_channels - 2 * c

        def conv3x3_bn_relu(in_c: int, out_c: int, dilation: int):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
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

        self.cbam = CBAM(out_channels)  # Thay SimAM bằng CBAM cho spatial focus

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
        x = self.cbam(x)
        if self.use_res:
            return x + identity
        return x + self.skip(identity)

# -------------------------
# Depthwise block (updated with CBAM)
# -------------------------
class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(out_channels)  # Thay SimAM
        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.cbam(x)
        if self.use_res:
            x = x + identity
        return x

# -------------------------
# Morphology Branch (NEW for v2, inspired by dual-branch papers)
# -------------------------
class MorphologyBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Simple edge/morphology extractor (Sobel-like conv for cavity boundaries)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1, bias=False),  # Detect edges
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=2, dilation=2, bias=False),  # Expand for morphology
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels // 2)

    def forward(self, x):
        x = self.edge_conv(x)
        return self.cbam(x)

# -------------------------
# ASPP Light (NEW for multi-scale RF in fusion)
# -------------------------
class ASPPLight(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // 3, 1, bias=False),  # 1x1
            nn.Conv2d(in_channels, out_channels // 3, 3, padding=3, dilation=3, bias=False),  # Dilation 3
            nn.Conv2d(in_channels, out_channels // 3, 3, padding=6, dilation=6, bias=False),  # Dilation 6
        ])
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(self, x):
        feats = [F.relu(branch(x)) for branch in self.branches]
        return self.proj(torch.cat(feats, dim=1))

# -------------------------
# GeM giữ nguyên
# -------------------------

class IVF_EffiMorphPP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.4,  # Tăng nhẹ chống overfit
        width_mult: float = 1.1,  # Giữ nhẹ để thử
        base_channels: int = 32,
        divisor: int = 8,
        task: str = "exp",
        use_coral: bool = True,  # Bật ordinal mặc định
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

        # Stem chung cho cả dual-branch
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Branch 1: Spatial (gốc, multi-scale)
        self.spatial_branch = nn.Sequential(
            MultiScaleBlock(base, c1),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            DWConvBlock(c1, c2, stride=2, dilation=1),
            DWConvBlock(c2, c3, stride=1, dilation=2),
            DWConvBlock(c3, c4, stride=1, dilation=4),
        )

        # Branch 2: Morphology (new, focus edge/cavity)
        self.morph_branch = MorphologyBranch(base, c4)

        # Fusion dual-branch + ASPP light
        self.fusion = nn.Sequential(
            nn.Conv2d(c4 + c4, c4, 1, bias=False),  # Concat 2 branches
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPPLight(c4, c4)  # Multi-scale RF
        self.eca = ECA(c4)

        hidden = max(256, c4 // 2)  # Tăng hidden
        self.gap = GeM(p=4.0)  # Tăng p cho salient
        self.dropout = nn.Dropout(dropout_p)

        output_dim = num_classes - 1 if use_coral else num_classes
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Branch 1: Spatial
        spatial = self.spatial_branch(x)

        # Branch 2: Morphology
        morph = self.morph_branch(x)

        # Fusion
        fused = torch.cat([spatial, morph], dim=1)
        fused = self.fusion(fused)
        fused = self.aspp(fused)
        fused = self.eca(fused)

        x = self.gap(fused).flatten(1)
        x = self.dropout(x)
        return self.head(x)