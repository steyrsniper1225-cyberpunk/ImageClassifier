"""Align per-equipment normal soft masks using tightly constrained transforms.

Allowed degrees of freedom:
  * X/Y translation
  * right-angle rotation (0, 90, 180, 270 degrees)
  * independent X/Y scale with absolute deviation strictly below 5 percent

The module intentionally excludes arbitrary-angle rotation, shear, perspective,
and non-rigid warping so that local defects cannot be normalized away.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from Mask_Alignment_Visualization import (
    create_alignment_review,
    save_anchor_preview,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RIGHT_ANGLE_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class AlignmentConfig:
    rotations: tuple[int, ...] = RIGHT_ANGLE_ROTATIONS
    max_scale_deviation: float = 0.04
    coarse_scale_step: float = 0.02
    refine_scale_step: float = 0.005
    max_shift_px: float = 12.0
    search_size: int = 64
    binary_threshold: float = 0.50
    anchor_band_px: int = 6
    reference_sample_size: int = 256
    reference_iterations: int = 1
    review_count: int = 40


@dataclass(frozen=True)
class Transform:
    rotation_degrees: int
    scale_x: float
    scale_y: float
    shift_x: float
    shift_y: float


@dataclass
class AlignmentResult:
    aligned_soft: np.ndarray
    transform: Transform
    anchor_rmse: float
    full_rmse: float
    binary_iou: float
    phase_response: float
    retained_metal_fraction: float


def discover_masks(path: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Soft-mask folder does not exist: {root}")
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No masks found: {root}")
    return paths


def load_soft_mask(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel mask: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Mask contains non-finite values: {path}")
    return np.clip(array, 0.0, 1.0)


def save_soft_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def _matrix_for_transform(shape: tuple[int, int], transform: Transform) -> np.ndarray:
    height, width = shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotation = np.vstack(
        [cv2.getRotationMatrix2D(center, transform.rotation_degrees, 1.0), [0.0, 0.0, 1.0]]
    )
    cx, cy = center
    scale = np.array(
        [
            [transform.scale_x, 0.0, cx * (1.0 - transform.scale_x)],
            [0.0, transform.scale_y, cy * (1.0 - transform.scale_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation = np.array(
        [[1.0, 0.0, transform.shift_x], [0.0, 1.0, transform.shift_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (translation @ scale @ rotation)[:2]


def warp_soft(mask: np.ndarray, transform: Transform, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    matrix = _matrix_for_transform(mask.shape, transform)
    return cv2.warpAffine(
        mask.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _weighted_rmse(reference: np.ndarray, candidate: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError("Anchor weights are empty")
    return float(np.sqrt(np.sum(weights * np.square(reference - candidate)) / denominator))


def _binary_iou(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    aa = a >= threshold
    bb = b >= threshold
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 1.0


def make_auto_anchor(reference: np.ndarray, threshold: float, band_px: int) -> np.ndarray:
    """Create a boundary-band anchor; variable regions can later be downweighted by MAD."""
    binary = (reference >= threshold).astype(np.uint8)
    radius = max(1, int(band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    anchor = (dilated != eroded).astype(np.float32)
    if not anchor.any():
        raise ValueError("Could not derive an anchor band from the reference mask")
    return anchor


def _scale_values(max_deviation: float, step: float) -> list[float]:
    count = int(np.floor(max_deviation / step + 1e-9))
    values = {1.0}
    for index in range(1, count + 1):
        values.add(round(1.0 - index * step, 6))
        values.add(round(1.0 + index * step, 6))
    return sorted(value for value in values if abs(value - 1.0) < 0.05)


def _phase_translation(reference: np.ndarray, candidate: np.ndarray, max_shift: float) -> tuple[float, float, float]:
    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    (relative_x, relative_y), response = cv2.phaseCorrelate(
        reference.astype(np.float32), candidate.astype(np.float32), window
    )
    # phaseCorrelate reports candidate displacement relative to reference.
    shift_x = float(np.clip(-relative_x, -max_shift, max_shift))
    shift_y = float(np.clip(-relative_y, -max_shift, max_shift))
    return shift_x, shift_y, float(response)


def _evaluate_transform(
    query: np.ndarray,
    reference: np.ndarray,
    anchor: np.ndarray,
    transform: Transform,
    threshold: float,
    phase_response: float,
) -> AlignmentResult:
    aligned = warp_soft(query, transform, reference.shape)
    expected_mass = max(float(query.sum()) * transform.scale_x * transform.scale_y, 1e-6)
    retained = min(1.0, float(aligned.sum()) / expected_mass)
    return AlignmentResult(
        aligned_soft=aligned,
        transform=transform,
        anchor_rmse=_weighted_rmse(reference, aligned, anchor),
        full_rmse=float(np.sqrt(np.mean(np.square(reference - aligned)))),
        binary_iou=_binary_iou(reference, aligned, threshold),
        phase_response=phase_response,
        retained_metal_fraction=retained,
    )


def align_mask(
    query: np.ndarray,
    reference: np.ndarray,
    anchor: np.ndarray,
    config: AlignmentConfig,
) -> AlignmentResult:
    if query.shape != reference.shape or anchor.shape != reference.shape:
        raise ValueError("Query, reference, and anchor must have identical shapes")
    search_h = min(config.search_size, reference.shape[0])
    search_w = min(config.search_size, reference.shape[1])
    small_shape = (search_h, search_w)
    small_ref = cv2.resize(reference, (search_w, search_h), interpolation=cv2.INTER_AREA)
    small_anchor = cv2.resize(anchor, (search_w, search_h), interpolation=cv2.INTER_AREA)
    small_query = cv2.resize(query, (search_w, search_h), interpolation=cv2.INTER_AREA)
    scale_ratio_x = reference.shape[1] / search_w
    scale_ratio_y = reference.shape[0] / search_h
    coarse_scales = _scale_values(config.max_scale_deviation, config.coarse_scale_step)
    best: tuple[float, Transform, float] | None = None

    for rotation in config.rotations:
        for scale_x in coarse_scales:
            for scale_y in coarse_scales:
                base = Transform(rotation, scale_x, scale_y, 0.0, 0.0)
                candidate = warp_soft(small_query, base, small_shape)
                tx, ty, response = _phase_translation(
                    small_ref, candidate, config.max_shift_px / max(scale_ratio_x, scale_ratio_y)
                )
                transform_small = Transform(rotation, scale_x, scale_y, tx, ty)
                shifted = warp_soft(small_query, transform_small, small_shape)
                score = _weighted_rmse(small_ref, shifted, small_anchor)
                transform_full = Transform(
                    rotation, scale_x, scale_y, tx * scale_ratio_x, ty * scale_ratio_y
                )
                if best is None or score < best[0]:
                    best = (score, transform_full, response)

    assert best is not None
    _, coarse, _ = best
    refine_radius = config.coarse_scale_step / 2.0
    refine_offsets = np.arange(
        -refine_radius,
        refine_radius + config.refine_scale_step * 0.5,
        config.refine_scale_step,
    )
    best_result: AlignmentResult | None = None
    for dx_scale in refine_offsets:
        for dy_scale in refine_offsets:
            scale_x = round(coarse.scale_x + float(dx_scale), 6)
            scale_y = round(coarse.scale_y + float(dy_scale), 6)
            if (
                abs(scale_x - 1.0) > config.max_scale_deviation + 1e-9
                or abs(scale_y - 1.0) > config.max_scale_deviation + 1e-9
                or abs(scale_x - 1.0) >= 0.05
                or abs(scale_y - 1.0) >= 0.05
            ):
                continue
            unshifted = Transform(coarse.rotation_degrees, scale_x, scale_y, 0.0, 0.0)
            candidate = warp_soft(query, unshifted, reference.shape)
            tx, ty, response = _phase_translation(reference, candidate, config.max_shift_px)
            # Test a one-pixel neighborhood because phase correlation can be subpixel-biased by scale.
            for offset_x in (-1.0, 0.0, 1.0):
                for offset_y in (-1.0, 0.0, 1.0):
                    transform = Transform(
                        coarse.rotation_degrees,
                        scale_x,
                        scale_y,
                        float(np.clip(tx + offset_x, -config.max_shift_px, config.max_shift_px)),
                        float(np.clip(ty + offset_y, -config.max_shift_px, config.max_shift_px)),
                    )
                    result = _evaluate_transform(
                        query, reference, anchor, transform, config.binary_threshold, response
                    )
                    if best_result is None or result.anchor_rmse < best_result.anchor_rmse:
                        best_result = result
    assert best_result is not None
    return best_result


def _sample_paths(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[int(index)] for index in indices]


def _median_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required")
    return np.median(np.stack(masks, axis=0), axis=0).astype(np.float32)


def build_reference(
    sample_paths: list[Path],
    explicit_reference: Path | None,
    explicit_anchor: Path | None,
    config: AlignmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    masks = [load_soft_mask(path) for path in sample_paths]
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"All masks must have one shape, got: {sorted(shapes)}")
    # A raw pixel median is invalid when the folder contains mixed right-angle
    # orientations. Start from one real normal mask, align the deterministic
    # sample to it, and only then form the robust median reference.
    reference = load_soft_mask(explicit_reference) if explicit_reference else masks[0].copy()
    if reference.shape != masks[0].shape:
        raise ValueError("Explicit reference shape does not match input masks")
    if explicit_anchor:
        anchor = load_soft_mask(explicit_anchor)
        if anchor.shape != reference.shape:
            raise ValueError("Anchor shape does not match reference")
        anchor = np.clip(anchor, 0.0, 1.0)
    else:
        anchor = make_auto_anchor(reference, config.binary_threshold, config.anchor_band_px)

    if explicit_reference:
        return reference, anchor

    for _ in range(config.reference_iterations):
        aligned = [align_mask(mask, reference, anchor, config).aligned_soft for mask in masks]
        reference = _median_masks(aligned)
        if explicit_anchor is None:
            base_anchor = make_auto_anchor(reference, config.binary_threshold, config.anchor_band_px)
            deviations = np.median(np.abs(np.stack(aligned) - reference), axis=0)
            stability = 1.0 / (deviations + 0.02)
            stability /= max(float(stability.max()), 1e-6)
            anchor = (base_anchor * stability).astype(np.float32)
    return reference, anchor


def _safe_stem(path: Path, root: Path) -> str:
    return "__".join(path.relative_to(root).with_suffix("").parts)


def _summary_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def parse_rotations(raw: str) -> tuple[int, ...]:
    rotations = tuple(dict.fromkeys(int(item.strip()) % 360 for item in raw.split(",")))
    if not rotations or any(value not in RIGHT_ANGLE_ROTATIONS for value in rotations):
        raise ValueError("--rotations may contain only 0,90,180,270")
    return rotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align normal soft masks with constrained transforms.")
    parser.add_argument("--input-soft-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-soft")
    parser.add_argument("--anchor-mask")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--max-scale-deviation", type=float, default=0.04)
    parser.add_argument("--coarse-scale-step", type=float, default=0.02)
    parser.add_argument("--refine-scale-step", type=float, default=0.005)
    parser.add_argument("--max-shift-px", type=float, default=12.0)
    parser.add_argument("--search-size", type=int, default=64)
    parser.add_argument("--binary-threshold", type=float, default=0.50)
    parser.add_argument("--anchor-band-px", type=int, default=6)
    parser.add_argument("--reference-sample-size", type=int, default=256)
    parser.add_argument("--reference-iterations", type=int, default=1)
    parser.add_argument("--review-count", type=int, default=40)
    return parser.parse_known_args()[0]


def validate_config(config: AlignmentConfig) -> None:
    if not 0.0 <= config.max_scale_deviation < 0.05:
        raise ValueError("max_scale_deviation must be >= 0 and strictly below 0.05")
    if config.coarse_scale_step <= 0 or config.refine_scale_step <= 0:
        raise ValueError("Scale steps must be positive")
    if config.max_shift_px < 0:
        raise ValueError("max_shift_px must be non-negative")
    if not 0.0 < config.binary_threshold < 1.0:
        raise ValueError("binary_threshold must be between 0 and 1")
    if config.reference_sample_size < 1 or config.reference_iterations < 0:
        raise ValueError("Reference sample size/iterations are invalid")
    if config.search_size < 16 or config.anchor_band_px < 1 or config.review_count < 0:
        raise ValueError("Search size, anchor band, or review count is invalid")


def main() -> None:
    args = parse_args()
    config = AlignmentConfig(
        rotations=parse_rotations(args.rotations),
        max_scale_deviation=args.max_scale_deviation,
        coarse_scale_step=args.coarse_scale_step,
        refine_scale_step=args.refine_scale_step,
        max_shift_px=args.max_shift_px,
        search_size=args.search_size,
        binary_threshold=args.binary_threshold,
        anchor_band_px=args.anchor_band_px,
        reference_sample_size=args.reference_sample_size,
        reference_iterations=args.reference_iterations,
        review_count=args.review_count,
    )
    validate_config(config)
    input_root = Path(args.input_soft_dir)
    paths = discover_masks(input_root, args.recursive)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty; use a new versioned folder to avoid overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    sample_paths = _sample_paths(paths, config.reference_sample_size)
    reference, anchor = build_reference(
        sample_paths,
        Path(args.reference_soft) if args.reference_soft else None,
        Path(args.anchor_mask) if args.anchor_mask else None,
        config,
    )
    save_soft_mask(reference, output / "reference" / "reference_soft_median.png")
    save_soft_mask(anchor, output / "reference" / "anchor_weights.png")
    save_anchor_preview(reference, anchor, output / "reference" / "anchor_preview.png")

    review_indices = set(
        int(index)
        for index in np.linspace(0, len(paths) - 1, min(config.review_count, len(paths)), dtype=int)
    )
    rows: list[dict[str, object]] = []
    best_medoid: tuple[float, np.ndarray] | None = None
    for index, path in enumerate(
        tqdm(paths, desc="Aligining masks", unit="mask")
      ):
        query = load_soft_mask(path)
        result = align_mask(query, reference, anchor, config)
        stem = _safe_stem(path, input_root)
        aligned_soft_path = output / "aligned_soft_masks" / f"{stem}_aligned_soft.png"
        aligned_binary_path = output / "aligned_binary_masks" / f"{stem}_aligned_mask.png"
        save_soft_mask(result.aligned_soft, aligned_soft_path)
        save_soft_mask(
            (result.aligned_soft >= config.binary_threshold).astype(np.float32), aligned_binary_path
        )
        review_path = output / "alignment_reviews" / f"{stem}_alignment_review.png"
        if index in review_indices:
            create_alignment_review(
                query,
                result.aligned_soft,
                reference,
                review_path,
                config.binary_threshold,
            )
        transform = result.transform
        rows.append(
            {
                "filename": path.name,
                "source_path": str(path.resolve()),
                "rotation_degrees_ccw": transform.rotation_degrees,
                "scale_x": transform.scale_x,
                "scale_y": transform.scale_y,
                "shift_x": round(transform.shift_x, 6),
                "shift_y": round(transform.shift_y, 6),
                "anchor_rmse": round(result.anchor_rmse, 8),
                "full_rmse": round(result.full_rmse, 8),
                "binary_iou": round(result.binary_iou, 8),
                "phase_response": round(result.phase_response, 8),
                "retained_metal_fraction": round(result.retained_metal_fraction, 8),
                "aligned_soft_path": str(aligned_soft_path.resolve()),
                "aligned_binary_path": str(aligned_binary_path.resolve()),
                "review_path": str(review_path.resolve()) if index in review_indices else "",
                "worst_review_path": "",
            }
        )
        if best_medoid is None or result.full_rmse < best_medoid[0]:
            best_medoid = (result.full_rmse, result.aligned_soft.copy())
        # print(f"[{index + 1}/{len(paths)}] {path.name} anchor_rmse={result.anchor_rmse:.6f}")

    assert best_medoid is not None
    save_soft_mask(best_medoid[1], output / "reference" / "reference_medoid_actual.png")
    worst_rows = sorted(rows, key=lambda row: float(row["anchor_rmse"]), reverse=True)[
        : min(config.review_count, len(rows))
    ]
    for row in worst_rows:
        source_path = Path(str(row["source_path"]))
        aligned_path = Path(str(row["aligned_soft_path"]))
        worst_path = output / "alignment_reviews_worst" / (
            f"{aligned_path.stem}_worst_alignment_review.png"
        )
        create_alignment_review(
            load_soft_mask(source_path),
            load_soft_mask(aligned_path),
            reference,
            worst_path,
            config.binary_threshold,
        )
        row["worst_review_path"] = str(worst_path.resolve())
    fields = list(rows[0])
    with (output / "alignment_transforms.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        key: _summary_stats([float(row[key]) for row in rows])
        for key in ("anchor_rmse", "full_rmse", "binary_iou", "retained_metal_fraction")
    }
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Per-equipment normal soft-mask constrained alignment QA",
        "input_mask_count": len(paths),
        "reference_sample_count": len(sample_paths),
        "config": asdict(config),
        "allowed_transform_policy": {
            "translation_xy": True,
            "right_angle_rotation_only": list(config.rotations),
            "scale_xy_absolute_deviation_strictly_below": 0.05,
            "arbitrary_rotation": False,
            "shear": False,
            "perspective": False,
            "elastic_warp": False,
        },
        "metrics": metrics,
        "artifacts": {
            "transforms_csv": str((output / "alignment_transforms.csv").resolve()),
            "reference_soft": str((output / "reference" / "reference_soft_median.png").resolve()),
            "reference_medoid_actual": str((output / "reference" / "reference_medoid_actual.png").resolve()),
            "anchor_preview": str((output / "reference" / "anchor_preview.png").resolve()),
            "worst_review_dir": str((output / "alignment_reviews_worst").resolve()),
        },
    }
    (output / "alignment_qa.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Alignment complete: {len(rows)} masks")
    print(f"QA summary: {(output / 'alignment_qa.json').resolve()}")


if __name__ == "__main__":
    main()
