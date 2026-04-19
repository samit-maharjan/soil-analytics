"""On-image FESEM annotations: Grad-CAM + label/arrow overlay (reference figure style)."""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def _pil_bottom_crop(img: Image.Image, fraction: float) -> Image.Image:
    """Match training/inference crop: remove bottom ``fraction`` of height."""
    if fraction <= 0:
        return img
    w, h = img.size
    new_h = max(1, int(h * (1.0 - fraction)))
    return img.crop((0, 0, w, new_h))


def _select_grad_cam_layer(model: nn.Module, backbone: str) -> nn.Module | None:
    b = backbone.lower()
    if "efficientnet" in b and hasattr(model, "blocks"):
        blk = model.blocks[-1]
        try:
            inv = blk[0]
            return inv.conv_pwl  # type: ignore[no-any-return]
        except (IndexError, AttributeError):
            return None
    if hasattr(model, "layer4"):
        lb = model.layer4[-1]
        if hasattr(lb, "conv3"):
            return lb.conv3  # Bottleneck
        if hasattr(lb, "conv2"):
            return lb.conv2  # BasicBlock
    return None


def _compute_grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    backbone: str,
) -> np.ndarray | None:
    """Return CAM in model input pixel space (H=W=img_size), float32 in [0, 1]."""
    layer = _select_grad_cam_layer(model, backbone)
    if layer is None:
        return None
    device = x.device
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def fwd_hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        out.retain_grad()
        captured["act"] = out

    h = layer.register_forward_hook(fwd_hook)
    try:
        logits = model(x)
        score = logits[0, class_idx]
        model.zero_grad(set_to_none=True)
        score.backward()
        act = captured.get("act")
        if act is None or act.grad is None:
            return None
        grad = act.grad
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0]
        cam = cam.detach().float().cpu().numpy()
        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin + 1e-8)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        _, _, Hx, Wx = x.shape
        cam_t = torch.from_numpy(cam)[None, None, ...].float().to(device)
        cam_up = (
            F.interpolate(cam_t, size=(Hx, Wx), mode="bilinear", align_corners=False)[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return cam_up
    except RuntimeError:
        return None
    finally:
        h.remove()


def _jet_rgb(cam: np.ndarray) -> np.ndarray:
    """HxW float [0,1] -> HxWx3 uint8 (matplotlib jet; lazy import)."""
    z = np.clip(cam, 0.0, 1.0)
    try:
        from matplotlib import colormaps

        cmap = colormaps["jet"]
    except Exception:  # pragma: no cover
        from matplotlib import pyplot as plt

        cmap = plt.cm.jet
    rgba = cmap(z)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _tip_from_cam(cam: np.ndarray | None, w: int, h: int) -> tuple[int, int]:
    """Peak / centroid of high activation; fallback points center-right."""
    if cam is None or cam.size == 0:
        return int(w * 0.62), int(h * 0.42)
    flat = cam.ravel()
    thr = float(np.percentile(flat, 92.0))
    ys, xs = np.where(cam >= thr)
    if ys.size == 0:
        iy, ix = np.unravel_index(int(np.argmax(cam)), cam.shape)
    else:
        iy, ix = int(ys.mean()), int(xs.mean())
    # Map from cam grid to display size if needed (caller resizes cam to display size)
    return int(ix), int(iy)


def _load_font(size: int) -> ImageFont.ImageFont:
    paths = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    )
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text_stroke(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "white",
    stroke: str = "black",
    sw: int = 2,
) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str = "white",
    width: int = 3,
    head: int = 14,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / length, dy / length
    bx, by = x1 - ux * head, y1 - uy * head
    perp_x, perp_y = -uy, ux
    left = (bx + perp_x * (head * 0.45), by + perp_y * (head * 0.45))
    right = (bx - perp_x * (head * 0.45), by - perp_y * (head * 0.45))
    draw.line((x0, y0, bx, by), fill=fill, width=width)
    draw.polygon([(x1, y1), left, right], fill=fill)


def render_annotated_fesem_png(
    pil_orig: Image.Image,
    predicted_class: str,
    confidence: float,
    model: nn.Module,
    meta: dict[str, Any],
    tfm,
    device: torch.device,
    *,
    blend_heatmap: bool = True,
    heatmap_alpha: float = 0.38,
) -> bytes:
    """
    Build a PNG with class label on the left, optional Grad-CAM tint, and an arrow toward
    the high-saliency region (Grad-CAM peak centroid), matching common FESEM annotation style.
    """
    backbone = str(meta.get("backbone", "resnet18"))
    crop_frac = float(meta.get("crop_bottom_fraction", 0.0))

    pil_base = _pil_bottom_crop(pil_orig.convert("RGB"), crop_frac)
    w, h = pil_base.size

    x = tfm(pil_orig).unsqueeze(0).to(device)
    class_list: list[str] = meta["classes"]
    pred_idx = int(class_list.index(predicted_class))

    cam = _compute_grad_cam(model, x, pred_idx, backbone)

    rgb = np.array(pil_base, dtype=np.uint8)
    cam_rs: np.ndarray | None = None
    if cam is not None:
        cam_rs = np.array(
            Image.fromarray((cam * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        if blend_heatmap and heatmap_alpha > 1e-6:
            jet = _jet_rgb(cam_rs)
            a = float(np.clip(heatmap_alpha, 0.0, 1.0))
            blended = rgb.astype(np.float32) * (1.0 - a) + jet.astype(np.float32) * a
            rgb = np.clip(blended, 0, 255).astype(np.uint8)
    tip_x, tip_y = _tip_from_cam(cam_rs, w, h)

    tip_x = int(np.clip(tip_x, w * 0.08, w * 0.92))
    tip_y = int(np.clip(tip_y, h * 0.08, h * 0.92))

    out = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(out)

    title = predicted_class.replace("_", " ")
    sub = f"p = {confidence:.2f}"

    fs = max(18, min(44, int(w / 22)))
    font_big = _load_font(fs)
    font_sm = _load_font(max(14, fs - 6))

    pad = max(12, fs // 2)
    tx, ty = pad, max(pad, int(h * 0.18))

    _draw_text_stroke(draw, (tx, ty), title, font_big)
    bbox = draw.textbbox((tx, ty), title, font=font_big)
    sy = bbox[3] + fs // 6
    _draw_text_stroke(draw, (tx, sy), sub, font_sm)

    bbox2 = draw.textbbox((tx, sy), sub, font=font_sm)
    text_right = float(bbox2[2])
    text_mid_y = (bbox[1] + bbox2[3]) / 2.0

    start_x = text_right + pad * 0.8
    start_y = text_mid_y
    if start_x > tip_x - 30:
        start_x = float(tx)
        start_y = float(bbox2[3] + pad)
    _draw_arrow(draw, (start_x, start_y), (float(tip_x), float(tip_y)))

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def annotate_prediction_dict(
    raw_bytes: bytes,
    result: dict[str, Any],
    model: nn.Module,
    meta: dict[str, Any],
    tfm,
    device: torch.device,
    *,
    blend_heatmap: bool = True,
    heatmap_alpha: float = 0.38,
) -> bytes:
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return render_annotated_fesem_png(
        pil,
        result["predicted_class"],
        float(result["confidence"]),
        model,
        meta,
        tfm,
        device,
        blend_heatmap=blend_heatmap,
        heatmap_alpha=heatmap_alpha,
    )
