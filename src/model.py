import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimAM(nn.Module):
    def __init__(self, channels):
        super(SimAM, self).__init__()
        self.channels = channels
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 1e-8)) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
    def __init__(self, channels, k_size=5):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        c = out_channels // 3
        self.branch1 = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(in_channels, c, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(in_channels, out_channels - 2*c, kernel_size=7, padding=3)
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.simam = SimAM(out_channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.proj(x)
        x = self.simam(x)
        return x


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(DWConvBlock, self).__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.simam = SimAM(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.simam(x)
        return x


class IVF_EffiMorphPP(nn.Module):
    """
    Scalable IVF_EffiMorphPP.

    Scaling rule:
      base = make_divisible(base_channels * width_mult)
      stage1_out = 2*base
      stage2_out = 4*base
      stage3_out = 8*base
      stage4_out = 16*base

    Default (base_channels=32, width_mult=1.0) reproduces original channels:
      stem=32, s1=64, s2=128, s3=256, s4=512
    """

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.3,
        width_mult: float = 1.0,
        base_channels: int = 32,
        divisor: int = 8,
        eca_k: int = 5,
    ):
        super().__init__()

        def make_divisible(v: float, div: int = 8, min_value: int | None = None) -> int:
            if min_value is None:
                min_value = div
            new_v = max(min_value, int(v + div / 2) // div * div)
            # ensure we don't round down by more than 10%
            if new_v < 0.9 * v:
                new_v += div
            return int(new_v)

        # Compute scaled channels
        base = make_divisible(base_channels * width_mult, divisor)
        c_stem = base
        c1 = make_divisible(2 * base, divisor)   # stage1 out
        c2 = make_divisible(4 * base, divisor)   # stage2 out
        c3 = make_divisible(8 * base, divisor)   # stage3 out
        c4 = make_divisible(16 * base, divisor)  # stage4 out

        self._channels = {"stem": c_stem, "s1": c1, "s2": c2, "s3": c3, "s4": c4}

        # Stem
        self.stem = nn.Conv2d(3, c_stem, kernel_size=3, stride=2, padding=1)

        # Stage 1: Multi-scale @112x112 -> 56x56
        self.stage1 = MultiScaleBlock(c_stem, c1)
        self.stage1_down = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)

        # Stage 2: DW+PW @56x56 -> 28x28
        self.stage2 = DWConvBlock(c1, c2, stride=2)

        # Stage 3: DW+PW with dilation @28x28 -> 14x14
        self.stage3 = DWConvBlock(c2, c3, stride=2, dilation=2)

        # Stage 4: DW+PW @14x14 -> 7x7
        self.stage4 = DWConvBlock(c3, c4, stride=2)

        # Multi-scale fusion (project s2/s3/s4 -> c4 then gated sum)
        self.fusion_conv2 = nn.Conv2d(c2, c4, kernel_size=1)
        self.fusion_conv3 = nn.Conv2d(c3, c4, kernel_size=1)
        self.fusion_conv4 = nn.Conv2d(c4, c4, kernel_size=1)
        self.gates = nn.Parameter(torch.ones(3))  # alpha, beta, gamma

        self.eca = ECA(c4, k_size=eca_k)

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(c4, max(128, c4 // 2))
        self.fc2 = nn.Linear(max(128, c4 // 2), num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)  # 112x112

        # Stage 1
        x = self.stage1(x)        # 112x112
        x = self.stage1_down(x)   # 56x56

        # Stage 2
        s2 = self.stage2(x)       # 28x28

        # Stage 3
        s3 = self.stage3(s2)      # 14x14

        # Stage 4
        s4 = self.stage4(s3)      # 7x7

        # Fusion
        s2_up = F.interpolate(s2, size=(7, 7), mode="bilinear", align_corners=False)
        s3_up = F.interpolate(s3, size=(7, 7), mode="bilinear", align_corners=False)

        s2_f = self.fusion_conv2(s2_up)
        s3_f = self.fusion_conv3(s3_up)
        s4_f = self.fusion_conv4(s4)

        weights = F.softmax(self.gates, dim=0)
        fused = weights[0] * s2_f + weights[1] * s3_f + weights[2] * s4_f

        fused = self.eca(fused)

        # Head
        x = self.gap(fused).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable, dict(self._channels)
