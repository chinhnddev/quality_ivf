import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.dw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=padding,
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
        if not self.use_res:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.simam(x)
        if self.use_res:
            return x + identity
        return x + self.proj(identity)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, normed)[0]
        x = residual + attn_out
        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x


class LateMHSA(nn.Module):
    def __init__(self, dim, layers=1, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert layers >= 1, "mhsa_layers must be >= 1"
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(layers)]
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        for layer in self.layers:
            tokens = layer(tokens)
        out = tokens.mean(dim=1)
        assert out.shape == (b, c)
        return out


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p)


# -------------------------
# IVF_EffiMorphPP
# -------------------------
class IVF_EffiMorphPP(nn.Module):
    """
    Stable IVF_EffiMorphPP with optional Xception and MHSA enhancements.
    """

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        base_channels: int = 32,
        divisor: int = 8,
        eca_k: int = 5,
        task: str = "exp",
        use_coral: bool = False,
        use_xception_mid: bool = False,
        use_late_mhsa: bool = False,
        mhsa_layers: int = 1,
        mhsa_heads: int = 4,
        use_gem: bool = False,
        head_mlp: bool = False,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.0,
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

        stage_blocks = max(1, int(round(depth_mult)))

        self.stage1 = MultiScaleBlock(base, c1)
        self.stage1_down = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        block_mid = SeparableConvBlock if use_xception_mid else DWConvBlock
        self.stage2 = self._make_stage(block_mid, c1, c2, stage_blocks, stride=2)
        self.stage3 = self._make_stage(block_mid, c2, c3, stage_blocks, stride=2, dilation=2)
        self.stage4 = self._make_stage(DWConvBlock, c3, c4, stage_blocks, stride=2)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(c4, eca_k)

        self.use_late_mhsa = use_late_mhsa
        self.use_gem = use_gem
        self.head_mlp = head_mlp

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem = GeM() if use_gem else None
        self.dropout = nn.Dropout(dropout_p)

        if use_late_mhsa:
            self.mhsa = LateMHSA(c4, layers=max(1, mhsa_layers), heads=mhsa_heads)
        else:
            self.mhsa = None

        mlp_hidden = int(head_hidden_dim) if head_hidden_dim is not None else max(128, c4 // 2)
        if head_mlp:
            self.head_mlp_block = nn.Sequential(
                nn.Linear(c4, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(mlp_hidden, c4),
            )
        else:
            self.head_mlp_block = nn.Identity()

        output_dim = num_classes - 1 if (task == "exp" and use_coral) else num_classes
        self.head = nn.Sequential(
            nn.Linear(c4, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(mlp_hidden, output_dim),
        )

    def _make_stage(self, block_cls, in_channels, out_channels, blocks, stride=2, dilation=1):
        layers = []
        for idx in range(blocks):
            s = stride if idx == 0 else 1
            dil = dilation if idx == 0 else 1
            layers.append(block_cls(in_channels if idx == 0 else out_channels, out_channels, stride=s, dilation=dil))
        return nn.Sequential(*layers)

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

        if self.use_late_mhsa and self.mhsa is not None:
            x = self.mhsa(fused)
            if self.gem is not None:
                gem_vec = self.gem(fused).flatten(1)
                x = x + gem_vec
        else:
            pool = self.gem(fused) if self.gem is not None else self.gap(fused)
            x = pool.flatten(1)

        x = self.head_mlp_block(x)
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
