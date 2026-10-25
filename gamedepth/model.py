"""U-Net mit ResNet-18-Encoder für Tiefenschätzung aus einem Farbbild.

Encoder: ResNet-18 verkleinert das Bild schrittweise (1/2 … 1/32) und lernt
dabei immer abstraktere Merkmale. Decoder: vergrößert wieder bis zur vollen
Auflösung. Skip-Verbindungen reichen feine Details (Kanten von Hand und
Fingern) direkt vom Encoder an den Decoder weiter.
Ausgabe: log(Tiefe) für jedes Pixel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class UpBlock(nn.Module):
    """Vergrößern ×2, mit der Skip-Verbindung verketten, zwei 3×3-Faltungen."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        size = skip.shape[-2:] if skip is not None else [s * 2 for s in x.shape[-2:]]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DepthUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 1/2,  64 Kanäle
        self.pool = enc.maxpool                                    # 1/4
        self.layer1, self.layer2 = enc.layer1, enc.layer2          # 1/4 (64), 1/8 (128)
        self.layer3, self.layer4 = enc.layer3, enc.layer4          # 1/16 (256), 1/32 (512)
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = UpBlock(64, 0, 32)
        self.head = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d)
        return self.head(d)  # log(Tiefe), gleiche Größe wie das Eingabebild


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
