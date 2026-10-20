"""Hybrid CNN-Vision Transformer with multi-head self-attention, plus ablation baselines.

Architecture (``arch="hybrid"``)::

    image (3 x 224 x 224)
      └─ CNN stem (ResNet-18 up to layer3, or a light custom CNN)   -> C x 14 x 14 local features
           ├─ 1x1 conv projection -> 196 tokens of size D
           │     └─ [CLS] + learned positional embedding
           │          └─ L x pre-norm Transformer blocks (multi-head self-attention + MLP)
           │               └─ CLS embedding (global context)
           └─ global average pool (local texture summary)
      concat(CLS, GAP) -> MLP head -> tumour class

The CNN supplies inductive bias for local texture and edges, which matters on a
few thousand MRI slices. The transformer models long-range spatial
relationships, such as tumour position relative to midline structures. Fusing
CLS with GAP features keeps both views available to the classifier.

``arch="cnn"`` (stem + GAP + linear) and ``arch="vit"`` (16x16 patch embedding,
no CNN) are the ablation baselines.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CNN stems (output stride 16)
# ---------------------------------------------------------------------------
class LightCNN(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 256):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.GELU(),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.GELU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128), block(128, out_ch))
        self.out_channels = out_ch

    def forward(self, x):
        return self.features(x)


class ResNetStem(nn.Module):
    """ResNet-18 truncated after layer3 (256 channels, stride 16)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision import models

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = models.resnet18(weights=weights)
        self.features = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.out_channels = 256

    def forward(self, x):
        return self.features(x)


def make_stem(backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        return ResNetStem(pretrained)
    if backbone == "light":
        return LightCNN()
    raise ValueError(f"unknown backbone {backbone!r}")


# ---------------------------------------------------------------------------
# Transformer encoder with inspectable attention
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm transformer block. Stores attention maps when ``keep_attn`` is set."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        self.keep_attn = False
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=self.keep_attn, average_attn_weights=True)
        if self.keep_attn:
            self.attn_weights = w.detach()
        x = x + self.drop(a)
        return x + self.mlp(self.norm2(x))


class TokenEncoder(nn.Module):
    def __init__(self, n_tokens: int, dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        if tokens.shape[1] + 1 != self.pos.shape[1]:
            raise ValueError(f"got {tokens.shape[1]} tokens but positional embedding expects "
                             f"{self.pos.shape[1] - 1}; build the model with the training img_size")
        z = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1) + self.pos
        z = self.pos_drop(z)
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z[:, 0])

    def set_keep_attention(self, flag: bool) -> None:
        for blk in self.blocks:
            blk.keep_attn = flag


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class HybridViTCNN(nn.Module):
    def __init__(self, num_classes: int = 4, img_size: int = 224, backbone: str = "resnet18",
                 pretrained: bool = False, dim: int = 256, depth: int = 4, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, fuse_gap: bool = True):
        super().__init__()
        self.cnn = make_stem(backbone, pretrained)
        c = self.cnn.out_channels
        self.proj = nn.Conv2d(c, dim, kernel_size=1)
        self.grid = img_size // 16
        self.encoder = TokenEncoder(self.grid ** 2, dim, depth, heads, mlp_ratio, dropout)
        self.fuse_gap = fuse_gap
        in_head = dim + (c if fuse_gap else 0)
        self.head = nn.Sequential(nn.Linear(in_head, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, x):
        f = self.cnn(x)                                   # B x C x h x w
        tokens = self.proj(f).flatten(2).transpose(1, 2)  # B x hw x D
        z = self.encoder(tokens)
        if self.fuse_gap:
            z = torch.cat([z, f.mean(dim=(2, 3))], dim=1)
        return self.head(z)

    @property
    def cam_layer(self) -> nn.Module:
        return self.cnn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, backbone: str = "resnet18", pretrained: bool = False,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.cnn = make_stem(backbone, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.cnn.out_channels, num_classes))

    def forward(self, x):
        return self.head(self.cnn(x).mean(dim=(2, 3)))

    @property
    def cam_layer(self) -> nn.Module:
        return self.cnn


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, img_size: int = 224, dim: int = 256, depth: int = 6, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, **_):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.grid = img_size // 16
        self.encoder = TokenEncoder(self.grid ** 2, dim, depth, heads, mlp_ratio, dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(self.patch(x).flatten(2).transpose(1, 2)))

    @property
    def cam_layer(self) -> nn.Module:
        return self.patch


def build_model(arch: str = "hybrid", **kw) -> nn.Module:
    if arch == "hybrid":
        return HybridViTCNN(**kw)
    if arch == "cnn":
        kw.pop("img_size", None)
        return CNNClassifier(**kw)
    if arch == "vit":
        kw.pop("backbone", None)
        kw.pop("pretrained", None)
        return ViTClassifier(**kw)
    raise ValueError(f"unknown arch {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
