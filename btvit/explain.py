"""Visual explanations: Grad-CAM on the CNN stem and attention rollout on the transformer."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .data import denormalize  # noqa: E402


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on ``model.cam_layer``."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._h = model.cam_layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        self.acts = out
        if out.requires_grad:  # not the case under torch.no_grad(), e.g. during rollout
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, target: int | None = None) -> tuple[torch.Tensor, int]:
        """x: 1 x 3 x H x W. Returns (H x W heat-map in [0, 1], explained class)."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            target = int(logits.argmax(1)) if target is None else target
            self.model.zero_grad(set_to_none=True)
            logits[0, target].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach(), target

    def remove(self):
        self._h.remove()


@torch.no_grad()
def attention_rollout(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema, 2020): CLS-to-token relevance as an H x W map."""
    enc = model.encoder
    enc.set_keep_attention(True)
    model.eval()
    model(x)
    enc.set_keep_attention(False)
    n = enc.pos.shape[1]
    joint = torch.eye(n, device=x.device)
    for blk in enc.blocks:
        a = blk.attn_weights[0]                       # n x n, averaged over heads
        a = 0.5 * a + 0.5 * torch.eye(n, device=x.device)
        a = a / a.sum(-1, keepdim=True)
        joint = a @ joint
    g = model.grid
    m = joint[0, 1:].reshape(1, 1, g, g)
    m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    m = m - m.min()
    return m / (m.max() + 1e-8)


def save_gallery(model, dataset, classes, out: Path, n: int = 8, device: str = "cpu") -> Path:
    """Grid of test images with Grad-CAM (and attention rollout for transformer models)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    has_attn = hasattr(model, "encoder")
    cols = 3 if has_attn else 2
    idx = torch.linspace(0, len(dataset) - 1, steps=min(n, len(dataset))).long().tolist()
    fig, axes = plt.subplots(len(idx), cols, figsize=(3 * cols, 3 * len(idx)), squeeze=False)
    cam = GradCAM(model)
    for r, i in enumerate(idx):
        img, y = dataset[i]
        x = img.unsqueeze(0).to(device)
        heat, pred = cam(x)
        rgb = denormalize(img).permute(1, 2, 0).cpu().numpy()
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"true: {classes[y]}", fontsize=9)
        axes[r, 1].imshow(rgb)
        axes[r, 1].imshow(heat.cpu().numpy(), cmap="jet", alpha=0.45)
        axes[r, 1].set_title(f"Grad-CAM, pred: {classes[pred]}", fontsize=9)
        if has_attn:
            roll = attention_rollout(model, x)
            axes[r, 2].imshow(rgb)
            axes[r, 2].imshow(roll.cpu().numpy(), cmap="jet", alpha=0.45)
            axes[r, 2].set_title("attention rollout", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    cam.remove()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
