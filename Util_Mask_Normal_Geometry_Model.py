"""Build a robust COMMON_V1 normal-geometry model from aligned soft masks.

Only masks supplied through ``--fit-soft-dir`` affect the model. An optional
``--qa-soft-dir`` is scored against the frozen model without entering fitting.
The v001 model is deliberately non-parametric and pixelwise: median, empirical
quantile envelopes, and MAD. It does not perform elastic or local deformation,
so it cannot actively reconstruct a local notch as normal geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Mask_Normal_Alignment import discover_masks, load_soft_mask, save_soft_mask


@dataclass(frozen=True)
class ModelConfig:
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    binary_threshold: float = 0.50
    residual_pixel_threshold: float = 0.05
    robust_sigma_floor: float = 0.01
    review_count: int = 40
    minimum_fit_count: int = 20


@dataclass(frozen=True)
class GeometryModel:
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mad: np.ndarray
    robust_sigma: np.ndarray


def validate_config(config: ModelConfig) -> None:
    if not 0.0 < config.lower_quantile < 0.5:
        raise ValueError("lower_quantile must be in (0, 0.5)")
    if not 0.5 < config.upper_quantile < 1.0:
        raise ValueError("upper_quantile must be in (0.5, 1)")
    if config.lower_quantile >= config.upper_quantile:
        raise ValueError("lower_quantile must be below upper_quantile")
    if not 0.0 < config.binary_threshold < 1.0:
        raise ValueError("binary_threshold must be in (0, 1)")
    if not 0.0 < config.residual_pixel_threshold < 1.0:
        raise ValueError("residual_pixel_threshold must be in (0, 1)")
    if config.robust_sigma_floor <= 0 or config.review_count < 0:
        raise ValueError("sigma floor must be positive and review_count non-negative")
    if config.minimum_fit_count < 3:
        raise ValueError("minimum_fit_count must be at least 3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(path: Path, root: Path) -> str:
    return "__".join(path.relative_to(root).with_suffix("").parts)


def _check_unique_names(fit_paths: list[Path], qa_paths: list[Path]) -> None:
    fit_names = {path.name for path in fit_paths}
    overlap = sorted(fit_names.intersection(path.name for path in qa_paths))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"FIT/QA filename overlap detected ({len(overlap)}): {preview}")


def _stack_masks(paths: list[Path], stack_path: Path) -> np.memmap:
    first = load_soft_mask(paths[0])
    stack = np.lib.format.open_memmap(
        stack_path, mode="w+", dtype=np.float32, shape=(len(paths), *first.shape)
    )
    stack[0] = first
    for index, path in enumerate(paths[1:], start=1):
        mask = load_soft_mask(path)
        if mask.shape != first.shape:
            raise ValueError(f"Shape mismatch: {path} has {mask.shape}, expected {first.shape}")
        stack[index] = mask
    stack.flush()
    return stack


def fit_geometry_model(stack: np.ndarray, config: ModelConfig) -> GeometryModel:
    if stack.ndim != 3 or stack.shape[0] < config.minimum_fit_count:
        raise ValueError(
            f"Need at least {config.minimum_fit_count} aligned masks, got {stack.shape[0]}"
        )
    median = np.median(stack, axis=0).astype(np.float32)
    lower = np.quantile(stack, config.lower_quantile, axis=0).astype(np.float32)
    upper = np.quantile(stack, config.upper_quantile, axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - median[None, ...]), axis=0).astype(np.float32)
    robust_sigma = np.maximum(1.4826 * mad, config.robust_sigma_floor).astype(np.float32)
    if not (np.all(lower <= median + 1e-7) and np.all(median <= upper + 1e-7)):
        raise RuntimeError("Invalid fitted envelope ordering")
    return GeometryModel(median, lower, upper, mad, robust_sigma)


def _largest_component(values: np.ndarray, threshold: float) -> tuple[int, float]:
    binary = (values >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0, 0.0
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    selected = labels == label
    return int(selected.sum()), float(values[selected].sum())


def _robust_z_global_metrics(
    mask: np.ndarray,
    model: GeometryModel,
    defect_prone_mask: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    if defect_prone_mask.shape != mask.shape:
        raise ValueError(
            f"Defect-prone mask shape mismatch: "
            f"{defect_prone_mask.shape} vs {mask.shape}"
        )

    zone = defect_prone_mask.astype(bool)

    robust_z_missing = np.maximum(
        (model.median - mask)
        / np.maximum(model.robust_sigma, 1e-6),
        0.0,
    )

    # Defect-prone 영역 밖은 완전히 제거
    masked_robust_z = np.where(
        zone,
        robust_z_missing,
        0.0,
    ).astype(np.float32)

    values = robust_z_missing[zone]

    largest_area, largest_sum = _largest_component(
        masked_robust_z,
        threshold,
    )

    return {
        "robust_z_max": float(values.max(initial=0.0)),
        "robust_z_sum": float(values.sum()),
        "robust_z_area_ge_threshold": int(
            np.count_nonzero(values >= threshold)
        ),
        "robust_z_largest_component_sum": largest_sum,
        "robust_z_largest_component_area": largest_area,
    }


def score_mask(mask: np.ndarray, model: GeometryModel, config: ModelConfig) -> dict[str, float | int]:
    if mask.shape != model.median.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match model {model.median.shape}")
    missing = np.maximum(model.lower - mask, 0.0)
    extra = np.maximum(mask - model.upper, 0.0)
    absolute = np.abs(mask - model.median)
    standardized = absolute / model.robust_sigma
    missing_largest_area, missing_largest_sum = _largest_component(
        missing, config.residual_pixel_threshold
    )
    extra_largest_area, extra_largest_sum = _largest_component(extra, config.residual_pixel_threshold)
    transition_fraction = float(np.mean((mask > 0.10) & (mask < 0.90)))
    return {
        "center_rmse": float(np.sqrt(np.mean(np.square(mask - model.median)))),
        "mean_robust_z": float(np.mean(standardized)),
        "p99_robust_z": float(np.quantile(standardized, 0.99)),
        "missing_sum": float(missing.sum()),
        "missing_max": float(missing.max()),
        "missing_area": int(np.count_nonzero(missing >= config.residual_pixel_threshold)),
        "missing_largest_component_area": missing_largest_area,
        "missing_largest_component_sum": missing_largest_sum,
        "extra_sum": float(extra.sum()),
        "extra_max": float(extra.max()),
        "extra_area": int(np.count_nonzero(extra >= config.residual_pixel_threshold)),
        "extra_largest_component_area": extra_largest_area,
        "extra_largest_component_sum": extra_largest_sum,
        "transition_fraction": transition_fraction,
    }


def _rank01(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return np.zeros_like(array)
    order = np.argsort(array, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / (len(array) - 1)


def add_outlier_ranks(rows: list[dict[str, Any]]) -> None:
    """Rank for review only; never remove a fitting image automatically."""
    for role in sorted({str(row["dataset_role"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["dataset_role"] == role]
        metrics = (
            "center_rmse",
            "missing_largest_component_sum",
            "extra_largest_component_sum",
            "transition_fraction",
        )
        ranks = [
            _rank01([float(rows[index][metric]) for index in indices]) for metric in metrics
        ]
        combined = np.mean(np.stack(ranks), axis=0)
        for local_index, row_index in enumerate(indices):
            rows[row_index]["review_outlier_rank"] = float(combined[local_index])


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _gray(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")


def _heatmap(array: np.ndarray, gain: float = 4.0) -> Image.Image:
    values = np.round(np.clip(array * gain, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _panel(image: Image.Image, label: str, scale: int = 2) -> Image.Image:
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (image.width, image.height + 28), (28, 30, 34))
    panel.paste(image, (0, 28))
    ImageDraw.Draw(panel).text((7, 7), label, fill="white", font=ImageFont.load_default())
    return panel


def save_model_overview(model: GeometryModel, path: Path, binary_threshold: float = 0.5) -> None:
    panels = [
        _panel(_gray(model.median), "Normal median"),
        _panel(_gray(model.lower), "Empirical lower envelope"),
        _panel(_gray(model.upper), "Empirical upper envelope"),
        _panel(_heatmap(model.mad, 8.0), "MAD heatmap (x8)"),
        _panel(_heatmap(model.upper - model.lower, 4.0), "Tolerance width (x4)"),
        _panel(_gray((model.median >= binary_threshold).astype(np.float32)), "Median binary QA"),
    ]
    gap = 8
    width = panels[0].width * 3 + gap * 2
    height = panels[0].height * 2 + gap
    sheet = Image.new("RGB", (width, height), (20, 22, 25))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * (panel.width + gap), (index // 3) * (panel.height + gap)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_outlier_review(mask: np.ndarray, model: GeometryModel, row: dict[str, Any], path: Path) -> None:
    missing = np.maximum(model.lower - mask, 0.0)
    extra = np.maximum(mask - model.upper, 0.0)
    deviation = np.abs(mask - model.median)
    panels = [
        _panel(_gray(mask), "Observed aligned soft mask"),
        _panel(_gray(model.median), "Normal median"),
        _panel(_heatmap(missing), "Missing beyond tolerance"),
        _panel(_heatmap(extra), "Extra beyond tolerance"),
        _panel(_heatmap(deviation), "Absolute center deviation"),
        _panel(_heatmap(model.upper - model.lower), "Normal tolerance width"),
    ]
    gap, footer = 8, 42
    width = panels[0].width * 3 + gap * 2
    height = panels[0].height * 2 + gap + footer
    sheet = Image.new("RGB", (width, height), (20, 22, 25))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * (panel.width + gap), (index // 3) * (panel.height + gap)))
    ImageDraw.Draw(sheet).text(
        (8, height - footer + 10),
        f"{row['dataset_role']} | {row['source_name']} | missing component={row['missing_largest_component_sum']:.3f} | "
        f"RMSE={row['center_rmse']:.4f} | transition={row['transition_fraction']:.4f}",
        fill="white",
        font=ImageFont.load_default(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_float_artifact(array: np.ndarray, npy_path: Path, png_path: Path) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, array.astype(np.float32), allow_pickle=False)
    save_soft_mask(array, png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-soft-dir", type=Path, required=True)
    parser.add_argument("--qa-soft-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--lower-quantile", type=float, default=0.005)
    parser.add_argument("--upper-quantile", type=float, default=0.995)
    parser.add_argument("--binary-threshold", type=float, default=0.50)
    parser.add_argument("--residual-pixel-threshold", type=float, default=0.05)
    parser.add_argument("--robust-sigma-floor", type=float, default=0.01)
    parser.add_argument("--review-count", type=int, default=40)
    parser.add_argument("--minimum-fit-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        binary_threshold=args.binary_threshold,
        residual_pixel_threshold=args.residual_pixel_threshold,
        robust_sigma_floor=args.robust_sigma_floor,
        review_count=args.review_count,
        minimum_fit_count=args.minimum_fit_count,
    )
    validate_config(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    fit_paths = discover_masks(args.fit_soft_dir, args.recursive)
    qa_paths = discover_masks(args.qa_soft_dir, args.recursive) if args.qa_soft_dir else []
    _check_unique_names(fit_paths, qa_paths)
    if len(fit_paths) < config.minimum_fit_count:
        raise ValueError(f"Need at least {config.minimum_fit_count} FIT masks, got {len(fit_paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="normal_geometry_") as temp_dir:
        stack = _stack_masks(fit_paths, Path(temp_dir) / "fit_stack.npy")
        model = fit_geometry_model(stack, config)
        del stack

    model_dir = args.output_dir / "model"
    artifacts = {
        "median": (model.median, "normal_median"),
        "lower": (model.lower, "normal_lower_envelope"),
        "upper": (model.upper, "normal_upper_envelope"),
        "mad": (model.mad, "normal_mad"),
        "robust_sigma": (model.robust_sigma, "normal_robust_sigma"),
    }
    artifact_paths: dict[str, str] = {}
    for key, (array, stem) in artifacts.items():
        npy_path, png_path = model_dir / f"{stem}.npy", model_dir / f"{stem}.png"
        _save_float_artifact(array, npy_path, png_path)
        artifact_paths[f"{key}_npy"] = str(npy_path.resolve())
        artifact_paths[f"{key}_png"] = str(png_path.resolve())
    overview_path = args.output_dir / "normal_model_overview.png"
    save_model_overview(model, overview_path, config.binary_threshold)

    rows: list[dict[str, Any]] = []
    row_paths: list[Path] = []
    for role, root, paths in (("FIT", args.fit_soft_dir, fit_paths), ("QA", args.qa_soft_dir, qa_paths)):
        if root is None:
            continue
        for index, path in enumerate(paths, start=1):
            mask = load_soft_mask(path)
            
            robust_metrics = _robust_z_global_metrics(
                mask,
                model,
                threshold=3.0,
            )
            
            row = {
                "dataset_role": role,
                "source_name": path.name,
                "source_relative_path": str(path.relative_to(root)),
                "source_sha256": _sha256(path),
                **score_mask(mask, model, config),
                **robust_metrics,
            }
            rows.append(row)
            row_paths.append(path)
            print(f"[{role} {index}/{len(paths)}] {path.name}")
    add_outlier_ranks(rows)

    csv_path = args.output_dir / "normal_geometry_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = sorted(range(len(rows)), key=lambda index: float(rows[index]["review_outlier_rank"]), reverse=True)
    for review_index, row_index in enumerate(selected[: config.review_count], start=1):
        row = rows[row_index]
        safe = _safe_name(
            (args.fit_soft_dir if row["dataset_role"] == "FIT" else args.qa_soft_dir)
            / str(row["source_relative_path"]),
            args.fit_soft_dir if row["dataset_role"] == "FIT" else args.qa_soft_dir,
        )
        save_outlier_review(
            load_soft_mask(row_paths[row_index]), model, row,
            args.output_dir / "outlier_reviews_worst" / f"{review_index:03d}_{row['dataset_role']}_{safe}.png",
        )

    metric_names = [
        "center_rmse", "mean_robust_z", "p99_robust_z", "missing_sum",
        "missing_largest_component_sum", "extra_sum", "extra_largest_component_sum",
        "transition_fraction",
        "robust_z_max",
        "robust_z_sum",
        "robust_z_area_ge_threshold",
        "robust_z_largest_component_sum",
        "robust_z_largest_compinent_area",
    ]
    metrics_by_role: dict[str, Any] = {}
    for role in sorted({str(row["dataset_role"]) for row in rows}):
        subset = [row for row in rows if row["dataset_role"] == role]
        metrics_by_role[role] = {
            "count": len(subset),
            **{metric: _stats(float(row[metric]) for row in subset) for metric in metric_names},
        }
    model_json = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_type": "COMMON_V1 robust pixelwise normal geometry envelope v001",
        "fit_mask_count": len(fit_paths),
        "qa_mask_count": len(qa_paths),
        "image_shape": list(model.median.shape),
        "config": asdict(config),
        "policy": {
            "soft_mask_primary": True,
            "qa_data_used_for_fitting": False,
            "automatic_fit_outlier_removal": False,
            "local_or_elastic_fitting": False,
            "missing_residual_definition": "max(normal_lower_envelope - observed_soft_mask, 0)",
            "status": "candidate model; requires outlier review and defect-preservation validation",
        },
        "metrics_by_role": metrics_by_role,
        "artifacts": {
            **artifact_paths,
            "scores_csv": str(csv_path.resolve()),
            "overview": str(overview_path.resolve()),
            "outlier_review_dir": str((args.output_dir / "outlier_reviews_worst").resolve()),
        },
    }
    json_path = args.output_dir / "normal_geometry_model.json"
    json_path.write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Normal geometry candidate built: {json_path.resolve()}")


if __name__ == "__main__":
    main()
