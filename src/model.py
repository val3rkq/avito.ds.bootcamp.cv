"""
Классификатор ориентации с антисимметричной головой:
    f: backbone (timm, один логит)
    z(x) = (f(x) - f(rot180 x)) / 2,   p_180 = sigmoid(z / T)

Свойство z(rot180 x) = -z(x) выполняется точно при любых весах, поэтому p(x) + p(rot180 x) = 1 встроено
в архитектуру: симметричные строки автоматически получают p около 0.5, а сеть не может опираться на признаки,
не связанные с ориентацией (фон, яркость).

Цена --- два прогона backbone, оба идут одним батчем.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class OrientNet(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_small_100", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx3xHxW -> B логитов класса 180 (антисимметричных)."""
        both = torch.cat([x, torch.flip(x, dims=(2, 3))], dim=0)  # поворот на 180°
        f = self.backbone(both).squeeze(1)
        b = x.shape[0]
        return (f[:b] - f[b:]) / 2


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
