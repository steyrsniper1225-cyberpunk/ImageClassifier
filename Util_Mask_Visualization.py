"""Visual review helpers for masks produced by Mask_Extracting_Process.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_BG = (28, 30, 34)
TEXT_COLOR = (245, 245, 245)


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2).astype(np.uint8)
    return array[..., :3].astype(np.uint8)


def make_boundary_overlay(
    image: Image.Image | np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int] = (255, 40, 40),
    thickness: int = 1,
) -> Image.Image:
    """Draw the extracted metal boundary without obscuring the source image."""
    rgb = _to_rgb_array(image).copy()
    mask_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, color[::-1], thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _gray_panel(values: np.ndarray) -> Image.Image:
    gray = np.clip(np.asarray(values, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _labeled_panel(image: Image.Image, label: str, scale: int) -> Image.Image:
    image = image.convert("RGB").resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )
    header = 30
    panel = Image.new("RGB", (image.width, image.height + header), PANEL_BG)
    panel.paste(image, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), label, fill=TEXT_COLOR, font=ImageFont.load_default())
    return panel


def create_review_panel(
    image: Image.Image | np.ndarray,
    soft_mask: np.ndarray,
    binary_mask: np.ndarray,
    output_path: str | Path | None = None,
    scale: int = 2,
) -> Image.Image:
    """Create a 2x2 panel: source, soft mask, binary mask, and boundary overlay."""
    rgb = Image.fromarray(_to_rgb_array(image))
    panels = [
        _labeled_panel(rgb, "Original", scale),
        _labeled_panel(_gray_panel(soft_mask), "Soft metal mask", scale),
        _labeled_panel(_gray_panel(np.asarray(binary_mask) > 0), "Binary metal mask", scale),
        _labeled_panel(make_boundary_overlay(rgb, binary_mask), "Boundary overlay", scale),
    ]
    gap = 8
    width = panels[0].width * 2 + gap
    height = panels[0].height * 2 + gap
    sheet = Image.new("RGB", (width, height), PANEL_BG)
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (panels[0].width + gap, 0))
    sheet.paste(panels[2], (0, panels[0].height + gap))
    sheet.paste(panels[3], (panels[0].width + gap, panels[0].height + gap))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual mask review panel.")
    parser.add_argument("--image", required=True, help="Original RGB image")
    parser.add_argument("--soft-mask", required=True, help="8-bit soft mask PNG")
    parser.add_argument("--binary-mask", required=True, help="Binary mask PNG")
    parser.add_argument("--output", required=True, help="Review panel output PNG")
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    soft = np.asarray(Image.open(args.soft_mask).convert("L"), dtype=np.float32) / 255.0
    binary = np.asarray(Image.open(args.binary_mask).convert("L"), dtype=np.uint8) > 127
    create_review_panel(image, soft, binary, args.output, max(1, args.scale))
    print(f"Saved review panel: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
