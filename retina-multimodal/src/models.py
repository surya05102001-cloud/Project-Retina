"""
models.py
EfficientNet-B0 image encoder + MLP tabular encoder + fusion classifier.
"""
import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        # num_classes=0 -> feature extractor; global_pool="avg" -> (B, C)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        if in_dim <= 0:
            self.enabled = False
            self.out_dim = 0
            self.net = nn.Identity()
        else:
            self.enabled = True
            self.out_dim = hidden
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            # return zero-width embedding (handled at fusion)
            b = x.shape[0] if isinstance(x, torch.Tensor) else 1
            return torch.zeros(b, 0, device=x.device if isinstance(x, torch.Tensor) else None)
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, num_classes: int, tab_in_dim: int = 0,
                 backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.img_enc = ImageEncoder(backbone_name, pretrained)
        self.tab_enc = TabularEncoder(tab_in_dim)
        fusion_in = self.img_enc.out_dim + self.tab_enc.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        zi = self.img_enc(img)  # (B, Ci)
        zt = self.tab_enc(tab if tab is not None else torch.zeros(img.size(0), 0, device=img.device))
        z = torch.cat([zi, zt], dim=1)  # (B, Ci+Ct)
        return self.classifier(z)

