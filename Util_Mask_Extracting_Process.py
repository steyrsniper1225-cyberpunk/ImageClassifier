"""Extract reviewable metal masks from aligned industrial inspection ROIs.

The pipeline estimates metal/background colors independently for every image,
creates a continuous (soft) metal mask, thresholds it, and writes visual QA
artifacts. It does not build the normal-tolerance model yet; its output is the
input to that later stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from Mask_Visualization import create_review_panel, make_boundary_overlay


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskConfig:
    threshold: float = 0.50
    color_blur_sigma: float = 0.8
    min_component_area: int = 40
    metal_is_lighter: bool = True
    expected_width: int = 256
    expected_height: int = 256
    enforce_expected_size: bool = True
    kmeans_attempts: int = 5
    random_seed: int = 20260726


@dataclass
class MaskResult:
    soft_mask: np.ndarray
    binary_mask: np.ndarray
    metal_rgb: tuple[int, int, int]
    background_rgb: tuple[int, int, int]
    metal_fraction: float
    color_separation_lab: float


def discover_images(path: str | Path, recursive: bool = False) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input does not exist: {source}")
    iterator: Iterable[Path] = source.rglob("*") if recursive else source.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _validate_image(rgb: np.ndarray, path: Path, config: MaskConfig) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image: {path}")
    h, w = rgb.shape[:2]
    if config.enforce_expected_size and (w, h) != (config.expected_width, config.expected_height):
        raise ValueError(
            f"Expected {config.expected_width}x{config.expected_height}, got {w}x{h}: {path}"
        )


def _estimate_color_centers(rgb: np.ndarray, config: MaskConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Return metal/background centers in RGB and their Lab separation."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)
    # A deterministic stride keeps runtime bounded for larger-than-ROI images.
    if len(pixels) > 65536:
        pixels = pixels[:: max(1, len(pixels) // 65536)]
    cv2.setRNGSeed(config.random_seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.05)
    _, labels, centers = cv2.kmeans(
        pixels,
        2,
        None,
        criteria,
        config.kmeans_attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError("Color clustering produced an empty cluster")
    lighter = int(np.argmax(centers[:, 0]))
    metal_index = lighter if config.metal_is_lighter else 1 - lighter
    background_index = 1 - metal_index
    separation = float(np.linalg.norm(centers[metal_index] - centers[background_index]))

    centers_u8 = np.clip(centers.reshape(1, 2, 3), 0, 255).astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB).reshape(2, 3)
    return centers_rgb[metal_index], centers_rgb[background_index], separation


def _soft_projection(rgb: np.ndarray, metal_rgb: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    """Project Lab pixels onto the background-to-metal color axis."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    centers_rgb = np.array([[background_rgb, metal_rgb]], dtype=np.uint8)
    centers_lab = cv2.cvtColor(centers_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0]
    background_lab, metal_lab = centers_lab
    axis = metal_lab - background_lab
    denominator = float(np.dot(axis, axis))
    if denominator < 1.0:
        raise RuntimeError("Metal and background colors are not separable")
    soft = np.sum((lab - background_lab) * axis, axis=2) / denominator
    return np.clip(soft, 0.0, 1.0)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == label
    return keep


def extract_metal_mask(image: Image.Image | np.ndarray, config: MaskConfig | None = None) -> MaskResult:
    config = config or MaskConfig()
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
    metal_rgb, background_rgb, separation = _estimate_color_centers(rgb, config)
    soft = _soft_projection(rgb, metal_rgb, background_rgb)
    if config.color_blur_sigma > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), config.color_blur_sigma)
    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    binary = _remove_small_components(soft >= config.threshold, config.min_component_area)
    return MaskResult(
        soft_mask=soft,
        binary_mask=binary,
        metal_rgb=tuple(int(v) for v in metal_rgb),
        background_rgb=tuple(int(v) for v in background_rgb),
        metal_fraction=float(binary.mean()),
        color_separation_lab=separation,
    )


def _safe_stem(path: Path, input_root: Path | None) -> str:
    if input_root is None:
        return path.stem
    relative = path.relative_to(input_root).with_suffix("")
    return "__".join(relative.parts)


def process_image(path: Path, output_dir: Path, config: MaskConfig, input_root: Path | None) -> dict[str, object]:
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    _validate_image(rgb, path, config)
    result = extract_metal_mask(rgb, config)
    stem = _safe_stem(path, input_root)

    soft_path = output_dir / "soft_masks" / f"{stem}_soft.png"
    binary_path = output_dir / "binary_masks" / f"{stem}_mask.png"
    overlay_path = output_dir / "overlays" / f"{stem}_overlay.png"
    review_path = output_dir / "review_panels" / f"{stem}_review.png"
    for target in (soft_path, binary_path, overlay_path, review_path):
        target.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.round(result.soft_mask * 255).astype(np.uint8), mode="L").save(soft_path)
    Image.fromarray(result.binary_mask.astype(np.uint8) * 255, mode="L").save(binary_path)
    make_boundary_overlay(image, result.binary_mask).save(overlay_path)
    create_review_panel(image, result.soft_mask, result.binary_mask, review_path)

    return {
        "source_path": str(path.resolve()),
        "filename": path.name,
        "width": image.width,
        "height": image.height,
        "metal_rgb": json.dumps(result.metal_rgb),
        "background_rgb": json.dumps(result.background_rgb),
        "color_separation_lab": round(result.color_separation_lab, 6),
        "metal_fraction": round(result.metal_fraction, 8),
        "threshold": config.threshold,
        "soft_mask_path": str(soft_path.resolve()),
        "binary_mask_path": str(binary_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "review_panel_path": str(review_path.resolve()),
        "status": "ok",
        "error": "",
    }


def write_manifest(rows: list[dict[str, object]], output_dir: Path, config: MaskConfig) -> None:
    fields = list(rows[0].keys()) if rows else []
    with (output_dir / "mask_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Metal mask extraction and human visual QA",
        "config": asdict(config),
        "images_total": len(rows),
        "images_ok": sum(row["status"] == "ok" for row in rows),
        "images_failed": sum(row["status"] != "ok" for row in rows),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract metal masks and write human-review visualizations."
    )
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders")
    parser.add_argument("--threshold", type=float, default=0.50, help="Soft-mask threshold [0, 1]")
    parser.add_argument("--color-blur-sigma", type=float, default=0.8)
    parser.add_argument("--min-component-area", type=int, default=40)
    parser.add_argument("--metal-darker", action="store_true", help="Use when metal is darker than background")
    parser.add_argument("--allow-other-size", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    config = MaskConfig(
        threshold=args.threshold,
        color_blur_sigma=max(0.0, args.color_blur_sigma),
        min_component_area=max(0, args.min_component_area),
        metal_is_lighter=not args.metal_darker,
        enforce_expected_size=not args.allow_other_size,
    )
    source = Path(args.input)
    images = discover_images(source, args.recursive)
    if not images:
        raise FileNotFoundError(f"No supported images found: {source}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = source if source.is_dir() else None
    rows: list[dict[str, object]] = []
    for index, path in enumerate(images, start=1):
        try:
            row = process_image(path, output_dir, config, input_root)
            rows.append(row)
            print(f"[{index}/{len(images)}] OK: {path.name}")
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append({
                "source_path": str(path.resolve()),
                "filename": path.name,
                "width": "",
                "height": "",
                "metal_rgb": "",
                "background_rgb": "",
                "color_separation_lab": "",
                "metal_fraction": "",
                "threshold": config.threshold,
                "soft_mask_path": "",
                "binary_mask_path": "",
                "overlay_path": "",
                "review_panel_path": "",
                "status": "error",
                "error": str(exc),
            })
            print(f"[{index}/{len(images)}] ERROR: {path.name}: {exc}")
    write_manifest(rows, output_dir, config)
    ok_count = sum(row["status"] == "ok" for row in rows)
    print(f"Completed: {ok_count}/{len(rows)} images")
    print(f"Review panels: {(output_dir / 'review_panels').resolve()}")


if __name__ == "__main__":
    main()
