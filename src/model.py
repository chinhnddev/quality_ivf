"""
IVF_EffiMorphPP - Tier 1 Improvements
- Fixed fusion spatial mismatch
- SE blocks in DWConv
- Spatial attention in fusion
- Per-channel learnable GeM pooling

Expected gain: +1.5-2.0% → 84.5-85.0%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
# ATTENTION BLOCKS
# ═══════════════════════════════════════════════════════════════

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
        y = var / (4 * (var.sum(dim=[2, 3], keepdim=True) / n + 1e-6)) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
    """
    Efficient Channel Attention
    Paper: https://arxiv.org/abs/1910.03151
    """
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
# NEW: SQUEEZE-EXCITATION BLOCK
# ═══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Paper: https://arxiv.org/abs/1709.01507
    
    Adaptive channel-wise feature recalibration
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


# ═══════════════════════════════════════════════════════════════
# NEW: SPATIAL ATTENTION BLOCK
# ═══════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM
    Paper: https://arxiv.org/abs/1807.06521
    
    Focuses on 'where' is important in the feature map
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and conv
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


# ═══════════════════════════════════════════════════════════════
# MULTI-SCALE BLOCK (UNCHANGED)
# ═════════════���═════════════════════════════════════════════════

class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block with dilated convolutions
    - branch1: 3x3 (dilation=1)
    - branch2: 3x3 (dilation=2)
    - branch3: 3x3 (dilation=3)
    """
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


# ═══════════════════════════════════════════════════════════════
# IMPROVED: DWCONV WITH SE
# ═══════════════════════════════════════════════════════════════

class DWConvBlock(nn.Module):
    """
    Depthwise Separable Convolution with SE and SimAM
    
    IMPROVEMENTS:
      ✓ Added SE block for channel-wise attention
      ✓ SE → SimAM cascade for better feature refinement
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        dilation: int = 1,
        use_se: bool = True,  # Enable SE by default
    ):
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

        # ════════════════════════════════════════════════════════
        # NEW: SE block
        # ════════════════════════════════════════════════════════
        self.se = SEBlock(out_channels, reduction=16) if use_se else nn.Identity()
        
        self.simam = SimAM(out_channels)
        self.use_res = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.dw(x)
        x = self.pw(x)
        
        # ════════════════════════════════════════════════════════
        # Attention cascade: SE → SimAM
        # ════════════════════════════════════════════════════════
        x = self.se(x)     # Channel-wise recalibration
        x = self.simam(x)  # Spatial-channel attention
        
        if self.use_res:
            x = x + identity
        return x


# ═══════════════════════════════════════════════════════════════
# IMPROVED: PER-CHANNEL GEM POOLING
# ═══════════════════════════════════════════════════════════════

class GeM(nn.Module):
    """
    Generalized Mean Pooling with per-channel learnable exponents
    
    IMPROVEMENTS:
      ✓ Per-channel p parameter (was global)
      ✓ Learned during training
      ✓ Clamped to valid range [1, 10]
    
    Paper: https://arxiv.org/abs/1711.02512
    """
    def __init__(
        self, 
        p: float = 3.0, 
        eps: float = 1e-6, 
        per_channel: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.per_channel = per_channel
        self.initial_p = p
        # register placeholder so state_dict keys exist even before first forward
        self.register_parameter("p", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ════════════════════════════════════════════════════════
        # Initialize p on first forward pass
        # ════════════════════════════════════════════════════════
        if self.p is None:
            if self.per_channel:
                # One p per channel
                channels = x.size(1)
                self.p = nn.Parameter(
                    torch.ones(1, channels, 1, 1, device=x.device) * self.initial_p
                )
            else:
                # Single global p
                self.p = nn.Parameter(
                    torch.ones(1, device=x.device) * self.initial_p
                )
        
        # ════════════════════════════════════════════════════════
        # Clamp p to valid range [1, 10]
        # ════════════════════════════════════════════════════════
        p = torch.clamp(self.p, min=1.0, max=10.0)
        
        # ════════════════════════════════════════════════════════
        # Generalized mean pooling
        # ════════════════════════════════════════════════════════
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / p)


# ═══════════════════════════════════════════════════════════════
# MAIN MODEL: IVF_EffiMorphPP (TIER 1 IMPROVED)
# ═══════════════════════════════════════════════════════════════

class IVF_EffiMorphPP(nn.Module):
    """
    IVF_EffiMorphPP with Tier 1 Improvements
    
    CHANGES FROM BASELINE:
      ✓ Fixed fusion spatial mismatch (critical bug fix)
      ✓ SE blocks in all DWConv layers
      ✓ Spatial attention after fusion
      ✓ Per-channel learnable GeM pooling
    
    EXPECTED IMPROVEMENT: +1.5-2.0% (83% → 84.5-85%)
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
        use_se: bool = True,           # Enable SE blocks
        use_spatial_att: bool = True,  # Enable spatial attention
        per_channel_gem: bool = True,  # Enable per-channel GeM
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

        # ════════════════════════════════════════════════════════
        # STEM
        # ════════════════════════════════════════════════════════
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # ════════════════════════════════════════════════════════
        # STAGE 1
        # ════════════════════════════════════════════════════════
        self.stage1 = MultiScaleBlock(base, c1)

        self.stage1_down = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # ════════════════════════════════════════════════════════
        # STAGES 2-4 (with SE blocks)
        # ════════════════════════════════════════════════════════
        self.stage2 = DWConvBlock(c1, c2, stride=2, dilation=1, use_se=use_se)
        self.stage3 = DWConvBlock(c2, c3, stride=1, dilation=2, use_se=use_se)
        self.stage4 = DWConvBlock(c3, c4, stride=1, dilation=4, use_se=use_se)

        # ════════════════════════════════════════════════════════
        # FUSION (with spatial attention)
        # ════════════════════════════════════════════════════════
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 + c3 + c4, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        # NEW: Spatial attention after fusion
        self.spatial_att = SpatialAttention(kernel_size=7) if use_spatial_att else nn.Identity()
        
        self.eca = ECA(c4, k_size=None)

        # ════════════════════════════════════════════════════════
        # HEAD
        # ════════════════════════════════════════════════════════
        hidden = max(128, c4 // 2)
        
        # NEW: Per-channel GeM pooling
        self.gap = GeM(p=3.0, eps=1e-6, per_channel=per_channel_gem)
        
        self.dropout = nn.Dropout(dropout_p)

        if use_coral and num_classes < 2:
            raise ValueError("CORAL requires at least 2 classes.")
        
        output_dim = num_classes - 1 if use_coral else num_classes
        
        self.head = nn.Sequential(
            nn.Linear(c4, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed fusion
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Output logits [B, num_classes-1] for CORAL or [B, num_classes]
        """
        # ════════════════════════════════════════════════════════
        # ENCODER
        # ════════════════════════════════════════════════════════
        x = self.stem(x)          # [B, base, 112, 112]
        x = self.stage1(x)        # [B, c1, 112, 112]
        x = self.stage1_down(x)   # [B, c1, 56, 56]

        s2 = self.stage2(x)       # [B, c2, 28, 28]
        s3 = self.stage3(s2)      # [B, c3, 28, 28]
        s4 = self.stage4(s3)      # [B, c4, 14, 14]

        # ════════════════════════════════════════════════════════
        # FIXED: Interpolate before fusion
        # ════════════════════════════════════════════════════════
        target_size = s4.shape[2:]  # Get s4 spatial size [14, 14]
        
        # Upsample s2 and s3 to match s4
        s2_up = F.interpolate(
            s2, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        s3_up = F.interpolate(
            s3, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )

        # ════════════════════════════════════════════════════════
        # FUSION with attention
        # ════════════════════════════════════════════════════════
        fused = torch.cat([s2_up, s3_up, s4], dim=1)  # [B, c2+c3+c4, 14, 14]
        fused = self.fusion(fused)                     # [B, c4, 14, 14]
        
        # NEW: Spatial attention
        fused = self.spatial_att(fused)                # [B, c4, 14, 14]
        
        # Channel attention (existing)
        fused = self.eca(fused)                        # [B, c4, 14, 14]

        # ════════════════════════════════════════════════════════
        # HEAD
        # ════════════════════════════════════════════════════════
        x = self.gap(fused).flatten(1)  # [B, c4]
        x = self.dropout(x)             # [B, c4]
        x = self.head(x)                # [B, output_dim]
        
        return x


# ═══════════════════════════════════════════════════════════════
# PARAMETER COUNT UTILITY
# ════════════════════��══════════════════════════════════════════

def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*70}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*70}")
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Non-trainable:         {total - trainable:,}")
    print(f"{'='*70}")
    
    return total, trainable


# ═══════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════

def create_model(
    num_classes: int = 5,
    dropout_p: float = 0.3,
    use_coral: bool = True,
    task: str = "exp",
    **kwargs
):
    """
    Factory function to create model
    
    Args:
        num_classes: Number of classes
        dropout_p: Dropout probability
        use_coral: Use CORAL ordinal regression
        task: Task name
        **kwargs: Additional model arguments
    
    Returns:
        model: IVF_EffiMorphPP instance
    """
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        dropout_p=dropout_p,
        use_coral=use_coral,
        task=task,
        **kwargs
    )
    
    # Print parameter count
    count_parameters(model)
    
    return model


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create model with all Tier 1 improvements
    model = create_model(
        num_classes=5,
        dropout_p=0.3,
        use_coral=True,
        task="exp",
        use_se=True,
        use_spatial_att=True,
        per_channel_gem=True,
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Forward pass successful!")
