"""Visual QA helpers for normal soft-mask alignment."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_BG = (28, 30, 34)
TEXT_COLOR = (245, 245, 245)


def _gray(values: np.ndarray) -> Image.Image:
    array = np.clip(np.asarray(values, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def _labeled(image: Image.Image, label: str, scale: int) -> Image.Image:
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    header = 30
    panel = Image.new("RGB", (image.width, image.height + header), PANEL_BG)
    panel.paste(image, (0, header))
    ImageDraw.Draw(panel).text((8, 7), label, fill=TEXT_COLOR, font=ImageFont.load_default())
    return panel


def make_contour_overlay(
    reference_soft: np.ndarray,
    aligned_soft: np.ndarray,
    threshold: float = 0.5,
) -> Image.Image:
    """Reference=green, aligned=red, coincident contours=yellow."""
    base = np.clip((reference_soft + aligned_soft) * 0.25 * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    ref_u8 = (reference_soft >= threshold).astype(np.uint8) * 255
    aligned_u8 = (aligned_soft >= threshold).astype(np.uint8) * 255
    ref_contours, _ = cv2.findContours(ref_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    aligned_contours, _ = cv2.findContours(aligned_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, ref_contours, -1, (30, 255, 30), 1, lineType=cv2.LINE_AA)
    cv2.drawContours(rgb, aligned_contours, -1, (255, 40, 40), 1, lineType=cv2.LINE_AA)
    return Image.fromarray(rgb)


def make_residual_heatmap(reference_soft: np.ndarray, aligned_soft: np.ndarray) -> Image.Image:
    """Blue=observed extra metal, red=missing metal, dark=agreement."""
    residual = np.asarray(reference_soft, np.float32) - np.asarray(aligned_soft, np.float32)
    magnitude = np.clip(np.abs(residual) * 3.0, 0.0, 1.0)
    rgb = np.zeros((*residual.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(residual > 0, magnitude * 255, 0).astype(np.uint8)
    rgb[..., 2] = np.where(residual < 0, magnitude * 255, 0).astype(np.uint8)
    rgb[..., 1] = (magnitude * 35).astype(np.uint8)
    return Image.fromarray(rgb)


def create_alignment_review(
    original_soft: np.ndarray,
    aligned_soft: np.ndarray,
    reference_soft: np.ndarray,
    output_path: str | Path,
    threshold: float = 0.5,
    scale: int = 2,
) -> None:
    panels = [
        _labeled(_gray(original_soft), "Original soft mask", scale),
        _labeled(_gray(reference_soft), "Normal reference", scale),
        _labeled(make_contour_overlay(reference_soft, aligned_soft, threshold), "Contours: ref green / aligned red", scale),
        _labeled(make_residual_heatmap(reference_soft, aligned_soft), "Residual: missing red / extra blue", scale),
    ]
    gap = 8
    width = panels[0].width * 2 + gap
    height = panels[0].height * 2 + gap
    sheet = Image.new("RGB", (width, height), PANEL_BG)
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (panels[0].width + gap, 0))
    sheet.paste(panels[2], (0, panels[0].height + gap))
    sheet.paste(panels[3], (panels[0].width + gap, panels[0].height + gap))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_anchor_preview(reference_soft: np.ndarray, anchor_weights: np.ndarray, path: str | Path) -> None:
    ref = np.clip(reference_soft * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(ref[..., None], 3, axis=2)
    weight = np.clip(anchor_weights, 0.0, 1.0)
    rgb[..., 1] = np.maximum(rgb[..., 1], (weight * 255).astype(np.uint8))
    rgb[..., 0] = (rgb[..., 0] * (1.0 - 0.55 * weight)).astype(np.uint8)
    rgb[..., 2] = (rgb[..., 2] * (1.0 - 0.55 * weight)).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
