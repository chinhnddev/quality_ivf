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
    def __init__(self, num_classes, dropout_p=0.3):
        super(IVF_EffiMorphPP, self).__init__()
        # Stem
        self.stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        # Stage 1: Multi-scale @112x112 -> 56x56, 64ch
        self.stage1 = MultiScaleBlock(32, 64)
        self.stage1_down = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Stage 2: DW+PW @56x56 -> 28x28, 128ch
        self.stage2 = DWConvBlock(64, 128, stride=2)

        # Stage 3: DW+PW with dilation @28x28 -> 14x14, 256ch
        self.stage3 = DWConvBlock(128, 256, stride=2, dilation=2)

        # Stage 4: DW+PW @14x14 -> 7x7, 512ch
        self.stage4 = DWConvBlock(256, 512, stride=2)

        # Multi-scale fusion
        self.fusion_conv2 = nn.Conv2d(128, 512, kernel_size=1)
        self.fusion_conv3 = nn.Conv2d(256, 512, kernel_size=1)
        self.fusion_conv4 = nn.Conv2d(512, 512, kernel_size=1)
        self.gates = nn.Parameter(torch.ones(3))  # alpha, beta, gamma

        self.eca = ECA(512, k_size=5)

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)  # 112x112, 32

        # Stage 1
        x = self.stage1(x)  # 112x112, 64
        x = self.stage1_down(x)  # 56x56, 64

        # Stage 2
        s2 = self.stage2(x)  # 28x28, 128

        # Stage 3
        s3 = self.stage3(s2)  # 14x14, 256

        # Stage 4
        s4 = self.stage4(s3)  # 7x7, 512

        # Fusion
        s2_up = F.interpolate(s2, size=(7, 7), mode='bilinear', align_corners=False)
        s3_up = F.interpolate(s3, size=(7, 7), mode='bilinear', align_corners=False)
        s2_f = self.fusion_conv2(s2_up)
        s3_f = self.fusion_conv3(s3_up)
        s4_f = self.fusion_conv4(s4)

        weights = F.softmax(self.gates, dim=0)
        fused = weights[0] * s2_f + weights[1] * s3_f + weights[2] * s4_f

        fused = self.eca(fused)

        # Head
        x = self.gap(fused).view(fused.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
