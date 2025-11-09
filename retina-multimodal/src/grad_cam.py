"""
grad_cam.py
Minimal Grad-CAM for FusionModel using a chosen conv layer from the image backbone.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import cv2


class GradCAM:
    """
    Example target layer for timm EfficientNet-B0:
        target_layer = model.img_enc.backbone.conv_head
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.acts = None
        self.grads = None

        def fwd_hook(_, __, output):
            self.acts = output.detach()

        def bwd_hook(_, grad_in, grad_out):
            self.grads = grad_out[0].detach()

        self._h1 = target_layer.register_forward_hook(fwd_hook)
        self._h2 = target_layer.register_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self._h1.remove(); self._h2.remove()
        except Exception:
            pass

    @torch.no_grad()
    def _normalize(self, cam: np.ndarray) -> np.ndarray:
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def generate(self, img_tensor: torch.Tensor, tab_tensor: Optional[torch.Tensor] = None,
                 class_idx: Optional[int] = None) -> np.ndarray:
        """
        img_tensor: (1, 3, H, W) normalized input.
        Return: heatmap in [0,1] with shape (H, W).
        """
        self.model.zero_grad(set_to_none=True)
        out = self.model(img_tensor, tab_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        score = out[0, class_idx]
        score.backward(retain_graph=True)

        weights = self.grads[0].mean(dim=(1, 2))      # (C,)
        cam = (weights[:, None, None] * self.acts[0]).sum(0).cpu().numpy()  # (Hc, Wc)
        cam = self._normalize(cam)

        H, W = img_tensor.shape[-2:]
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
        return cam

