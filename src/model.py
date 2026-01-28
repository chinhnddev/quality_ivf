import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# Attention blocks
# -------------------------
class SimAM(nn.Module):
    """
    Simple Attention Module
    Paper: https://arxiv.org/abs/2106.03105
    """
    def __init__(self, channels):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = (x - mu).pow(2)
        # Tăng epsilon từ 1e-8 lên 1e-6 để tránh numerical instability
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + 1e-6)) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
    """
    Efficient Channel Attention
    Paper: https://arxiv.org/abs/1910.03151
    """
    def __init__(self, channels, k_size=None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Adaptive kernel size theo công thức trong paper
        if k_size is None:
            k_size = self._get_adaptive_kernel_size(channels)
        
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _get_adaptive_kernel_size(channels, gamma=2, b=1):
        """
        Adaptive kernel size theo ECA paper
        k = |log2(C) / γ + b / γ|_odd
        """
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 else t + 1  # Đảm bảo lẻ
        return max(3, k)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


# -------------------------
# Multi-scale block
# -------------------------
class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block với 3 branches (3x3, 5x5, 7x7)
    """
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

        # Residual connection
        self.use_res = in_channels == out_channels
        if not self.use_res:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

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
    """
    Depthwise Separable Convolution với residual connection
    Hỗ trợ cả stride=1 và stride=2
    """
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

        # Residual connection với projection khi cần
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x

        x = self.dw(x)
        x = self.pw(x)
        x = self.simam(x)

        x = x + identity
        return x


# -------------------------
# IVF_EffiMorphPP (FIXED VERSION)
# -------------------------
class IVF_EffiMorphPP(nn.Module):
    """
    Fixed IVF_EffiMorphPP với các cải tiến:
    - Dynamic interpolation size (không hardcode 7x7)
    - Dilation=1 ở stage3 để tránh conflict với stride=2
    - Residual projection cho tất cả DWConvBlock
    - Adaptive ECA kernel size
    - Hỗ trợ CORAL ordinal regression
    - BatchNorm + LayerNorm hybrid
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
    ):
        super().__init__()

        self.task = task
        self.use_coral = use_coral

        def make_divisible(v, d=8):
            return int((v + d / 2) // d * d)

        base = make_divisible(base_channels * width_mult, divisor)
        c1 = make_divisible(2 * base, divisor)
        c2 = make_divisible(4 * base, divisor)
        c3 = make_divisible(8 * base, divisor)
        c4 = make_divisible(16 * base, divisor)

        # Stem: 3 -> base, /2
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Stage 1: base -> c1, same spatial
        self.stage1 = MultiScaleBlock(base, c1)
        
        # Stage 1 downsample: c1 -> c1, /2
        self.stage1_down = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Stage 2: c1 -> c2, /2
        self.stage2 = DWConvBlock(c1, c2, stride=2)
        
        # Stage 3: c2 -> c3, /2 (FIX: dilation=1 thay vì 2)
        # Lý do: dilation=2 + stride=2 có thể gây vấn đề spatial size
        self.stage3 = DWConvBlock(c2, c3, stride=2, dilation=1)
        
        # Stage 4: c3 -> c4, /2
        self.stage4 = DWConvBlock(c3, c4, stride=2)

        # Fusion: concat c2, c3, c4 → c4
        # FIX: Dynamic interpolation size từ s4
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        # ECA với adaptive kernel size
        self.eca = ECA(c4, k_size=None)  # Auto-calculate

        # Head
        hidden = max(128, c4 // 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)

        # CORAL: output K-1 logits cho K classes
        # Regular: output K logits
        output_dim = num_classes - 1 if (task == "exp" and use_coral) else num_classes
        
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.LayerNorm(hidden),  # Batch-size safe
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        # Input: B x 3 x H x W
        x = self.stem(x)          # B x base x H/2 x W/2
        x = self.stage1(x)        # B x c1 x H/2 x W/2
        x = self.stage1_down(x)   # B x c1 x H/4 x W/4

        s2 = self.stage2(x)       # B x c2 x H/8 x W/8
        s3 = self.stage3(s2)      # B x c3 x H/16 x W/16
        s4 = self.stage4(s3)      # B x c4 x H/32 x W/32

        # FIX: Dynamic interpolation size
        target_size = s4.shape[2:]  # Lấy spatial size từ s4
        s2_up = F.interpolate(s2, size=target_size, mode="bilinear", align_corners=False)
        s3_up = F.interpolate(s3, size=target_size, mode="bilinear", align_corners=False)

        # Concat fusion
        fused = torch.cat([s2_up, s3_up, s4], dim=1)  # B x (c2+c3+c4) x H/32 x W/32
        fused = self.fusion(fused)                     # B x c4 x H/32 x W/32
        fused = self.eca(fused)                        # B x c4 x H/32 x W/32

        # Global pooling + head
        x = self.gap(fused).flatten(1)  # B x c4
        x = self.dropout(x)
        x = self.head(x)                # B x output_dim
        
        return x


# -------------------------
# Model info & testing
# -------------------------
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Comprehensive testing"""
    print("=" * 60)
    print("Testing IVF_EffiMorphPP (Fixed Version)")
    print("=" * 60)
    
    # Test 1: Different input sizes
    print("\n[Test 1] Multiple input sizes:")
    model = IVF_EffiMorphPP(num_classes=5, dropout_p=0.3)
    model.eval()
    
    for size in [224, 256, 320, 384]:
        x = torch.randn(2, 3, size, size)
        try:
            with torch.no_grad():
                y = model(x)
            print(f"  ✓ Input {size}x{size} → Output {y.shape} | OK")
        except Exception as e:
            print(f"  ✗ Input {size}x{size} → FAILED: {e}")
    
    # Test 2: CORAL mode
    print("\n[Test 2] CORAL mode:")
    model_coral = IVF_EffiMorphPP(num_classes=5, use_coral=True, task="exp")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model_coral(x)
    expected_dim = 4  # K-1 = 5-1 = 4
    assert y.shape[1] == expected_dim, f"Expected {expected_dim}, got {y.shape[1]}"
    print(f"  ✓ CORAL output: {y.shape} (expected K-1={expected_dim}) | OK")
    
    # Test 3: Gradient flow
    print("\n[Test 3] Gradient flow:")
    model.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    grad_issues = []
    for name, param in model.named_parameters():
        if param.grad is None:
            grad_issues.append(f"No gradient: {name}")
        elif torch.isnan(param.grad).any():
            grad_issues.append(f"NaN gradient: {name}")
    
    if grad_issues:
        print("  ✗ Gradient issues found:")
        for issue in grad_issues:
            print(f"    - {issue}")
    else:
        print("  ✓ All gradients OK")
    
    # Test 4: Model info
    print("\n[Test 4] Model info:")
    params = count_parameters(model)
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Test 5: Different width multipliers
    print("\n[Test 5] Width multipliers:")
    for wm in [0.5, 1.0, 1.5]:
        m = IVF_EffiMorphPP(num_classes=5, width_mult=wm)
        p = count_parameters(m)
        print(f"  width_mult={wm}: {p/1e6:.2f}M params")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()