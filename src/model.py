import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Attention blocks
# -------------------------
class SimAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mu).pow(2)
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + 1e-8)) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


# -------------------------
# Multi-scale block
# -------------------------
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

        self.use_res = in_channels == out_channels
        if not self.use_res:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        identity = x

        x = torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x)], dim=1
        )
        x = self.proj(x)
        x = self.simam(x)

        if self.use_res:
            x = x + identity
        else:
            x = x + self.skip(identity)
        return x


# -------------------------
# Depthwise block with residual
# -------------------------
class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        self.dw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
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

        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x

        x = self.dw(x)
        x = self.pw(x)
        x = self.simam(x)

        if self.use_res:
            x = x + identity
        return x


# -------------------------
# IVF_EffiMorphPP
# -------------------------
class IVF_EffiMorphPP(nn.Module):
    """
    Stable IVF_EffiMorphPP
    - BN + Residual
    - Concat fusion
    - LayerNorm head (batch-size safe)
    - Supports CORAL ordinal regression for EXP task
    """

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.3,
        width_mult: float = 1.0,
        base_channels: int = 32,
        divisor: int = 8,
        eca_k: int = 5,
        task: str = "exp",
        use_coral: bool = False,
    ):
        super().__init__()

        def make_divisible(v, d=8):
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

        self.stage2 = DWConvBlock(c1, c2, stride=2)
        self.stage3 = DWConvBlock(c2, c3, stride=2, dilation=1)
        self.stage4 = DWConvBlock(c3, c4, stride=2)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(c4, eca_k)

        # Head
        hidden = max(128, c4 // 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)

        # For CORAL ordinal regression on EXP task, output K-1 logits instead of K
        # CORAL uses K-1 thresholds for K classes (0 to K-1)
        output_dim = num_classes - 1 if (task == "exp" and use_coral) else num_classes
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage1_down(x)

        s2 = self.stage2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        s2_up = F.interpolate(s2, size=(7, 7), mode="bilinear", align_corners=False)
        s3_up = F.interpolate(s3, size=(7, 7), mode="bilinear", align_corners=False)

        fused = torch.cat([s2_up, s3_up, s4], dim=1)
        fused = self.fusion(fused)
        fused = self.eca(fused)

        x = self.gap(fused).flatten(1)
        x = self.dropout(x)
        x = self.head(x)
        return x


# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    model = IVF_EffiMorphPP(num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)
    print("Params:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M")
