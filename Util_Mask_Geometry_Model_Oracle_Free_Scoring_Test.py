"""Run a Tip1 oracle-free local-relative scoring POC.

Synthetic alpha is used only to create controlled defects and to evaluate the
selected location after scoring.  It is never passed to the oracle-free scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


from Mask_Geometry_Model_Oracle_Free_Scoring import (
    LocalScanConfig,
    LocalZoneScanResult,
    compute_signed_z,
    score_local_candidate_at_center,
    score_mask_oracle_free,
)
from Mask_Normal_Alignment import discover_masks, load_soft_mask
from Normal_Geometry_Model import (
    GeometryModel,
    ModelConfig,
    validate_config,
)
from Paired_Defect_Preservation_Test import (
    DefectSpec,
    inject_missing_metal,
    load_defect_config,
    make_defect_alpha,
)


@dataclass(frozen=True)
class LoadedGeometry:
    model: GeometryModel
    config: ModelConfig
    model_json_path: Path


@dataclass(frozen=True)
class ReviewItem:
    source_path: Path
    defect_id: str
    normal_score: float
    defect_score: float
    best_candidate_hits_alpha: bool
    best_distance_to_alpha_px: float


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)

    if array.size == 0:
        return {
            key: float("nan")
            for key in (
                "min",
                "p01",
                "p05",
                "median",
                "p95",
                "p99",
                "max",
                "mean",
            )
        }

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


def _load_array(model_dir: Path, stem: str) -> np.ndarray:
    path = model_dir / f"{stem}.npy"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing geometry model artifact: {path}"
        )

    array = np.load(
        path,
        allow_pickle=False,
    ).astype(np.float32)

    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(
            f"Invalid geometry model array: {path}"
        )

    return array


def load_frozen_geometry(
    geometry_output_dir: Path,
) -> LoadedGeometry:
    model_json_path = (
        geometry_output_dir
        / "normal_geometry_model.json"
    )

    if not model_json_path.is_file():
        raise FileNotFoundError(
            f"Normal geometry JSON not found: {model_json_path}"
        )

    payload = json.loads(
        model_json_path.read_text(encoding="utf-8")
    )

    raw_config = payload.get("config", {})

    config = ModelConfig(
        lower_quantile=float(
            raw_config.get("lower_quantile", 0.005)
        ),
        upper_quantile=float(
            raw_config.get("upper_quantile", 0.995)
        ),
        binary_threshold=float(
            raw_config.get("binary_threshold", 0.50)
        ),
        residual_pixel_threshold=float(
            raw_config.get("residual_pixel_threshold", 0.05)
        ),
        robust_sigma_floor=float(
            raw_config.get("robust_sigma_floor", 0.01)
        ),
        review_count=int(
            raw_config.get("review_count", 40)
        ),
        minimum_fit_count=int(
            raw_config.get("minimum_fit_count", 20)
        ),
    )

    validate_config(config)

    model_dir = geometry_output_dir / "model"

    model = GeometryModel(
        median=_load_array(model_dir, "normal_median"),
        lower=_load_array(
            model_dir,
            "normal_lower_envelope",
        ),
        upper=_load_array(
            model_dir,
            "normal_upper_envelope",
        ),
        mad=_load_array(model_dir, "normal_mad"),
        robust_sigma=_load_array(
            model_dir,
            "normal_robust_sigma",
        ),
    )

    shapes = {
        model.median.shape,
        model.lower.shape,
        model.upper.shape,
        model.mad.shape,
        model.robust_sigma.shape,
    }

    if len(shapes) != 1:
        raise ValueError(
            "Geometry model artifact shapes do not match: "
            f"{sorted(shapes)}"
        )

    return LoadedGeometry(
        model=model,
        config=config,
        model_json_path=model_json_path,
    )


def _load_zone(
    zones_dir: Path,
    zone_name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    zone_path = (
        zones_dir
        / zone_name
        / f"{zone_name}.npy"
    )

    if not zone_path.is_file():
        raise FileNotFoundError(
            f"Missing local-zone mask: {zone_path}"
        )

    zone = np.load(
        zone_path,
        allow_pickle=False,
    ).astype(bool)

    if zone.shape != expected_shape:
        raise ValueError(
            f"{zone_name}: zone shape mismatch: "
            f"{zone.shape} vs {expected_shape}"
        )

    if not np.any(zone):
        raise ValueError(
            f"{zone_name}: zone mask is empty"
        )

    return zone


def _best_candidate_fields(
    prefix: str,
    result: LocalZoneScanResult,
) -> dict[str, Any]:
    candidate = result.best_candidate

    output: dict[str, Any] = {
        f"{prefix}_zone_score": result.zone_score,
        f"{prefix}_scanned_candidate_count": (
            result.scanned_candidate_count
        ),
        f"{prefix}_valid_candidate_count": (
            result.valid_candidate_count
        ),
    }

    if candidate is None:
        output.update(
            {
                f"{prefix}_best_y": -1,
                f"{prefix}_best_x": -1,
                f"{prefix}_candidate_pixel_count": 0,
                f"{prefix}_reference_patch_count": 0,
                f"{prefix}_candidate_signed_z_median": float("nan"),
                f"{prefix}_reference_signed_z_median": float("nan"),
                f"{prefix}_local_signed_excess": float("nan"),
                f"{prefix}_local_corrected_top3_sum": float("nan"),
            }
        )
        return output

    output.update(
        {
            f"{prefix}_best_y": candidate.center_y,
            f"{prefix}_best_x": candidate.center_x,
            f"{prefix}_candidate_pixel_count": (
                candidate.candidate_pixel_count
            ),
            f"{prefix}_reference_patch_count": (
                candidate.reference_patch_count
            ),
            f"{prefix}_candidate_signed_z_median": (
                candidate.candidate_signed_z_median
            ),
            f"{prefix}_reference_signed_z_median": (
                candidate.reference_signed_z_median
            ),
            f"{prefix}_local_signed_excess": (
                candidate.local_signed_excess
            ),
            f"{prefix}_local_corrected_top3_sum": (
                candidate.local_corrected_top3_sum
            ),
        }
    )

    return output


def _alpha_localization_metrics(
    alpha: np.ndarray,
    zone: np.ndarray,
    result: LocalZoneScanResult,
    patch_radius: int,
    alpha_support_fraction: float = 0.05,
) -> dict[str, Any]:
    candidate = result.best_candidate
    alpha_max = float(alpha.max())

    if alpha_max <= 0.0:
        return {
            "alpha_center_y": -1.0,
            "alpha_center_x": -1.0,
            "best_distance_to_alpha_px": float("nan"),
            "best_candidate_hits_alpha": False,
        }

    alpha_support = (
        alpha >= alpha_max * alpha_support_fraction
    ) & zone

    support_pixels = np.argwhere(alpha_support)

    if support_pixels.size == 0:
        return {
            "alpha_center_y": -1.0,
            "alpha_center_x": -1.0,
            "best_distance_to_alpha_px": float("nan"),
            "best_candidate_hits_alpha": False,
        }

    support_weights = alpha[alpha_support].astype(np.float64)
    alpha_center_y = float(
        np.average(
            support_pixels[:, 0],
            weights=support_weights,
        )
    )
    alpha_center_x = float(
        np.average(
            support_pixels[:, 1],
            weights=support_weights,
        )
    )

    if candidate is None:
        return {
            "alpha_center_y": alpha_center_y,
            "alpha_center_x": alpha_center_x,
            "best_distance_to_alpha_px": float("nan"),
            "best_candidate_hits_alpha": False,
        }

    distance = float(
        np.hypot(
            candidate.center_y - alpha_center_y,
            candidate.center_x - alpha_center_x,
        )
    )

    height, width = alpha.shape
    y0 = max(0, candidate.center_y - patch_radius)
    y1 = min(height, candidate.center_y + patch_radius + 1)
    x0 = max(0, candidate.center_x - patch_radius)
    x1 = min(width, candidate.center_x + patch_radius + 1)

    hits_alpha = bool(
        np.any(alpha_support[y0:y1, x0:x1])
    )

    return {
        "alpha_center_y": alpha_center_y,
        "alpha_center_x": alpha_center_x,
        "best_distance_to_alpha_px": distance,
        "best_candidate_hits_alpha": hits_alpha,
    }


def _alpha_near_candidate_diagnostic(
    normal: np.ndarray,
    defective: np.ndarray,
    alpha: np.ndarray,
    zone: np.ndarray,
    model: GeometryModel,
    robust_sigma_floor: float,
    defect_result: LocalZoneScanResult,
    scan_config: LocalScanConfig,
) -> dict[str, Any]:
    normal_signed_z = compute_signed_z(
        observed=normal,
        model=model,
        robust_sigma_floor=robust_sigma_floor,
    )

    defect_signed_z = compute_signed_z(
        observed=defective,
        model=model,
        robust_sigma_floor=robust_sigma_floor,
    )

    alpha_max = float(
        alpha.max()
    )

    if alpha_max > 0.0:
        alpha_support = (
            alpha
            >= alpha_max * 0.05
        )
    else:
        alpha_support = np.zeros_like(
            alpha,
            dtype=bool,
        )

    alpha_zone_support = (
        alpha_support
        & zone
    )

    delta_signed_z = (
        defect_signed_z
        - normal_signed_z
    )

    delta_values = delta_signed_z[
        alpha_zone_support
    ]

    if delta_values.size:
        alpha_zone_delta_median = float(
            np.median(delta_values)
        )

        alpha_zone_delta_max = float(
            delta_values.max()
        )
    else:
        alpha_zone_delta_median = float(
            "nan"
        )

        alpha_zone_delta_max = float(
            "nan"
        )

    radius = (
        scan_config.patch_radius
    )

    kernel_size = (
        2 * radius + 1
    )

    near_alpha = cv2.dilate(
        alpha_support.astype(
            np.uint8
        ),
        np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        ),
        iterations=1,
    ).astype(bool)

    valid_score_pixels = (
        near_alpha
        & np.isfinite(
            defect_result.score_map
        )
    )

    valid_candidate_count = int(
        np.count_nonzero(
            valid_score_pixels
        )
    )

    output: dict[str, Any] = {
        "alpha_zone_support_pixel_count": int(
            np.count_nonzero(
                alpha_zone_support
            )
        ),
        "alpha_zone_delta_signed_z_median": (
            alpha_zone_delta_median
        ),
        "alpha_zone_delta_signed_z_max": (
            alpha_zone_delta_max
        ),
        "alpha_near_valid_candidate_count": (
            valid_candidate_count
        ),
        "alpha_near_zone_score": float(
            "nan"
        ),
        "alpha_near_best_y": -1,
        "alpha_near_best_x": -1,
        "alpha_near_candidate_pixel_count": 0,
        "alpha_near_reference_patch_count": 0,
        "alpha_near_candidate_signed_z_median": float(
            "nan"
        ),
        "alpha_near_reference_signed_z_median": float(
            "nan"
        ),
        "alpha_near_local_signed_excess": float(
            "nan"
        ),
        "alpha_near_local_corrected_top3_sum": float(
            "nan"
        ),
    }

    if valid_candidate_count == 0:
        return output

    restricted_scores = np.where(
        valid_score_pixels,
        defect_result.score_map,
        -np.inf,
    )

    flat_index = int(
        np.argmax(
            restricted_scores
        )
    )

    center_y, center_x = np.unravel_index(
        flat_index,
        restricted_scores.shape,
    )

    candidate = (
        score_local_candidate_at_center(
            signed_z=defect_signed_z,
            zone=zone,
            center_y=int(center_y),
            center_x=int(center_x),
            config=scan_config,
        )
    )

    if candidate is None:
        return output

    output.update(
        {
            "alpha_near_zone_score": float(
                restricted_scores[
                    center_y,
                    center_x,
                ]
            ),
            "alpha_near_best_y": int(
                center_y
            ),
            "alpha_near_best_x": int(
                center_x
            ),
            "alpha_near_candidate_pixel_count": (
                candidate.candidate_pixel_count
            ),
            "alpha_near_reference_patch_count": (
                candidate.reference_patch_count
            ),
            "alpha_near_candidate_signed_z_median": (
                candidate.candidate_signed_z_median
            ),
            "alpha_near_reference_signed_z_median": (
                candidate.reference_signed_z_median
            ),
            "alpha_near_local_signed_excess": (
                candidate.local_signed_excess
            ),
            "alpha_near_local_corrected_top3_sum": (
                candidate.local_corrected_top3_sum
            ),
        }
    )

    return output


def _gray_rgb(array: np.ndarray) -> np.ndarray:
    gray = np.round(
        np.clip(array, 0.0, 1.0) * 255
    ).astype(np.uint8)

    return np.repeat(gray[..., None], 3, axis=2)


def _score_map_rgb(
    score_map: np.ndarray,
    maximum: float,
) -> np.ndarray:
    valid = np.isfinite(score_map)
    normalized = np.zeros(score_map.shape, dtype=np.float32)

    if np.any(valid) and maximum > 0.0:
        normalized[valid] = np.clip(
            score_map[valid] / maximum,
            0.0,
            1.0,
        )

    heat = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )

    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat[~valid] = 0
    return heat


def _draw_candidate_marker(
    draw: ImageDraw.ImageDraw,
    offset_x: int,
    offset_y: int,
    result: LocalZoneScanResult,
    patch_radius: int,
) -> None:
    candidate = result.best_candidate

    if candidate is None:
        return

    x = offset_x + candidate.center_x
    y = offset_y + candidate.center_y
    radius = patch_radius + 2

    draw.rectangle(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        ),
        outline=(255, 255, 0),
        width=2,
    )


def save_pair_review(
    normal: np.ndarray,
    defective: np.ndarray,
    alpha: np.ndarray,
    zone: np.ndarray,
    normal_result: LocalZoneScanResult,
    defect_result: LocalZoneScanResult,
    defect_id: str,
    source_name: str,
    path: Path,
    patch_radius: int,
) -> None:
    normal_panel = _gray_rgb(normal).astype(np.float32)
    defect_panel = _gray_rgb(defective).astype(np.float32)

    normal_panel[zone] = (
        0.75 * normal_panel[zone]
        + 0.25 * np.array([40, 180, 255], dtype=np.float32)
    )
    defect_panel[zone] = (
        0.75 * defect_panel[zone]
        + 0.25 * np.array([40, 180, 255], dtype=np.float32)
    )

    alpha_support = alpha > max(float(alpha.max()) * 0.05, 1e-6)
    defect_panel[alpha_support] = (
        0.45 * defect_panel[alpha_support]
        + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    )

    common_maximum = max(
        float(normal_result.zone_score)
        if np.isfinite(normal_result.zone_score)
        else 0.0,
        float(defect_result.zone_score)
        if np.isfinite(defect_result.zone_score)
        else 0.0,
        1e-6,
    )

    normal_heat = _score_map_rgb(
        normal_result.score_map,
        common_maximum,
    )
    defect_heat = _score_map_rgb(
        defect_result.score_map,
        common_maximum,
    )

    height, width = normal.shape
    label_height = 26
    footer_height = 74
    canvas = Image.new(
        "RGB",
        (
            width * 2,
            (height + label_height) * 2 + footer_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    panels = (
        ("Aligned normal", normal_panel.astype(np.uint8)),
        ("Injected defect", defect_panel.astype(np.uint8)),
        ("Normal candidate score map", normal_heat),
        ("Defect candidate score map", defect_heat),
    )

    for index, (title, panel) in enumerate(panels):
        column = index % 2
        row = index // 2
        offset_x = column * width
        offset_y = row * (height + label_height)

        draw.text(
            (offset_x + 6, offset_y + 6),
            title,
            fill="black",
        )
        canvas.paste(
            Image.fromarray(panel),
            (offset_x, offset_y + label_height),
        )

        result = normal_result if column == 0 else defect_result
        _draw_candidate_marker(
            draw,
            offset_x,
            offset_y + label_height,
            result,
            patch_radius,
        )

    localization = _alpha_localization_metrics(
        alpha,
        zone,
        defect_result,
        patch_radius,
    )

    footer_y = (height + label_height) * 2 + 6
    draw.text(
        (6, footer_y),
        f"Source: {source_name} | Defect: {defect_id}",
        fill="black",
    )
    draw.text(
        (6, footer_y + 20),
        "Oracle-free zone maximum | "
        f"Normal: {normal_result.zone_score:.3f} | "
        f"Defect: {defect_result.zone_score:.3f} | "
        f"Shift: {defect_result.zone_score - normal_result.zone_score:.3f}",
        fill="black",
    )
    draw.text(
        (6, footer_y + 40),
        "Defect best candidate | "
        f"hit_alpha={localization['best_candidate_hits_alpha']} | "
        f"distance={float(localization['best_distance_to_alpha_px']):.2f}px",
        fill="black",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    safe = value
    for character in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        safe = safe.replace(character, "_")
    return safe


def _select_unique_sources(
    items: list[ReviewItem],
    count: int,
) -> list[ReviewItem]:
    selected: list[ReviewItem] = []
    seen: set[Path] = set()

    for item in items:
        if item.source_path in seen:
            continue
        seen.add(item.source_path)
        selected.append(item)
        if len(selected) >= count:
            break

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned-soft-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--geometry-output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--defect-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--defect-zones-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--zone-name",
        choices=("tip1, tip2",),
        default="tip1",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--review-count", type=int, default=40)
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument(
        "--max-reference-patches",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--minimum-candidate-pixels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--minimum-reference-patches",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--reference-guard-px",
        type=int,
        default=9,
    )

    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    if args.review_count < 0:
        raise ValueError("--review-count must be non-negative")

    minimum_non_overlap_distance = (
    2 * args.patch_radius + 1
    )
    
    if (
        args.reference_guard_px
        < minimum_non_overlap_distance
    ):
        raise ValueError(
            "--reference-guard-px must be at least "
            f"{minimum_non_overlap_distance} "
            "for the selected patch radius"
        )

    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )

    paths = discover_masks(
        args.aligned_soft_dir,
        args.recursive,
    )

    if args.max_images is not None:
        paths = paths[:args.max_images]

    loaded = load_frozen_geometry(
        args.geometry_output_dir
    )
    model = loaded.model
    model_config = loaded.config

    zone = _load_zone(
        args.defect_zones_dir,
        args.zone_name,
        model.median.shape,
    )
    
        candidate_center_bounds = {
            "tip1": (
                68,
                73,
                164,
                169,
            ),
            "tip2": (
                138,
                143,
                163,
                168,
            ),
        }
    
        (
            candidate_y0,
            candidate_y1,
            candidate_x0,
            candidate_x1,
        ) = candidate_center_bounds[
            args.zone_name
        ]
    
        candidate_center_mask = np.zeros_like(
            zone,
            dtype=bool,
        )
    
        candidate_center_mask[
            candidate_y0:candidate_y1,
            candidate_x0:candidate_x1,
        ] = True
    
        candidate_center_mask &= zone
    
        if not np.any(candidate_center_mask):
            raise RuntimeError(
                f"{args.zone_name}: candidate-center "
                "mask does not overlap the local zone"
            )

    defects, _ = load_defect_config(args.defect_config)

    applicable_specs: list[DefectSpec] = []

    for spec in defects:
        alpha = make_defect_alpha(model.median.shape, spec)
        alpha_max = float(alpha.max())
        alpha_support = (
            alpha >= alpha_max * 0.05
            if alpha_max > 0.0
            else np.zeros_like(alpha, dtype=bool)
        )

        if np.any(alpha_support & zone):
            applicable_specs.append(spec)

    if not applicable_specs:
        raise RuntimeError(
            f"No configured defects overlap the {args.zone_name} zone"
        )

    scan_config = LocalScanConfig(
        zone_name=args.zone_name,
        score_feature="local_corrected_top3_sum",
        patch_radius=args.patch_radius,
        max_reference_patches=args.max_reference_patches,
        minimum_candidate_pixels=args.minimum_candidate_pixels,
        minimum_reference_patches=args.minimum_reference_patches,
        top_k=3,
        keep_top_candidates=10,
        minimum_reference_center_distance=(
            args.reference_guard_px
        ),
        minimum_caninical_patch_mass=0.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    normal_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    review_items: list[ReviewItem] = []

    for path in tqdm(
        paths,
        desc=f"Oracle-free {args.zone_name} scan",
        unit="mask",
    ):
        normal = load_soft_mask(path)

        if normal.shape != model.median.shape:
            raise ValueError(
                f"Shape mismatch: {path} has {normal.shape}, "
                f"expected {model.median.shape}"
            )

        normal_result = score_mask_oracle_free(
            observed=normal,
            model=model,
            robust_sigma_floor=model_config.robust_sigma_floor,
            zone=zone,
            config=scan_config,
            candidate_center_mask = (candidate_center_mask),
        )

        if normal_result.best_candidate is None:
            raise RuntimeError(
                f"No valid oracle-free candidate for normal mask: {path}"
            )

        normal_row: dict[str, Any] = {
            "source_name": path.name,
            "source_path": str(path.resolve()),
            "zone_name": args.zone_name,
            "score_feature": scan_config.score_feature,
        }
        normal_row.update(
            _best_candidate_fields("normal", normal_result)
        )
        normal_rows.append(normal_row)

        for spec in applicable_specs:
            alpha = make_defect_alpha(normal.shape, spec)
            defective = inject_missing_metal(normal, alpha)

            injected_removed_energy = float(
                np.maximum(normal - defective, 0.0).sum()
            )

            if injected_removed_energy <= 0.05:
                raise ValueError(
                    f"{spec.defect_id}: configured defect does not "
                    "overlap observable metal"
                )

            defect_result = score_mask_oracle_free(
                observed=defective,
                model=model,
                robust_sigma_floor=model_config.robust_sigma_floor,
                zone=zone,
                config=scan_config,
                candidate_center_mask = (candidate_center_mask),
            )

            if defect_result.best_candidate is None:
                raise RuntimeError(
                    "No valid oracle-free candidate for injected mask: "
                    f"{path} / {spec.defect_id}"
                )

            localization = _alpha_localization_metrics(
                alpha=alpha,
                zone=zone,
                result=defect_result,
                patch_radius=scan_config.patch_radius,
            )
            
            alpha_near_diagnostic = (
                _alpha_near_candidate_diagnostic(
                    normal=normal,
                    defective=defective,
                    alpha=alpha,
                    zone=zone,
                    model=model,
                    robust_sigma_floor=(
                        model_config.robust_sigma_floor
                    ),
                    defect_result=defect_result,
                    scan_config=scan_config,
                )
            )

            pair_row: dict[str, Any] = {
                "source_name": path.name,
                "source_path": str(path.resolve()),
                "defect_id": spec.defect_id,
                "defect_shape": spec.shape,
                "zone_name": args.zone_name,
                "score_feature": scan_config.score_feature,
                "injected_removed_energy": injected_removed_energy,
            }

            pair_row.update(
                _best_candidate_fields("normal", normal_result)
            )
            pair_row.update(
                _best_candidate_fields("defect", defect_result)
            )
            pair_row.update(
                localization
            )
            
            pair_row.update(
                alpha_near_diagnostic
            )
            
            pair_row[
                "paired_zone_score_shift"
            ] = float(
                defect_result.zone_score
                - normal_result.zone_score
            )

            pair_rows.append(pair_row)

            review_items.append(
                ReviewItem(
                    source_path=path,
                    defect_id=spec.defect_id,
                    normal_score=float(normal_result.zone_score),
                    defect_score=float(defect_result.zone_score),
                    best_candidate_hits_alpha=bool(
                        localization["best_candidate_hits_alpha"]
                    ),
                    best_distance_to_alpha_px=float(
                        localization["best_distance_to_alpha_px"]
                    ),
                )
            )

    if not normal_rows or not pair_rows:
        raise RuntimeError("No oracle-free scoring rows were generated")

    normal_csv_path = args.output_dir / "oracle_free_normal_results.csv"
    pair_csv_path = args.output_dir / "oracle_free_pair_results.csv"
    _write_csv(normal_rows, normal_csv_path)
    _write_csv(pair_rows, pair_csv_path)

    normal_scores = [
        float(row["normal_zone_score"])
        for row in normal_rows
    ]
    defect_scores = [
        float(row["defect_zone_score"])
        for row in pair_rows
    ]
    paired_shifts = [
        float(row["paired_zone_score_shift"])
        for row in pair_rows
    ]
    localization_distances = [
        float(row["best_distance_to_alpha_px"])
        for row in pair_rows
        if np.isfinite(float(row["best_distance_to_alpha_px"]))
    ]

    by_defect: dict[str, Any] = {}

    for defect_id in sorted(
        {str(row["defect_id"]) for row in pair_rows}
    ):
        subset = [
            row
            for row in pair_rows
            if str(row["defect_id"]) == defect_id
        ]

        by_defect[defect_id] = {
            "count": len(subset),
            "defect_zone_score": _stats(
                float(row["defect_zone_score"])
                for row in subset
            ),
            "paired_zone_score_shift": _stats(
                float(row["paired_zone_score_shift"])
                for row in subset
            ),
            "best_candidate_hit_fraction": float(
                np.mean(
                    [
                        bool(row["best_candidate_hits_alpha"])
                        for row in subset
                    ]
                )
            ),
            "best_distance_to_alpha_px": _stats(
                float(row["best_distance_to_alpha_px"])
                for row in subset
                if np.isfinite(
                    float(row["best_distance_to_alpha_px"])
                )
            ),
        }

    summary = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "purpose": "Tip1 oracle-free local-relative scoring POC",
        "zone_name": args.zone_name,
        "score_feature": scan_config.score_feature,
        "normal_mask_count": len(normal_rows),
        "applicable_defect_count": len(applicable_specs),
        "pair_count": len(pair_rows),
        "scan_config": {
            "patch_radius": scan_config.patch_radius,
            "max_reference_patches": (
                scan_config.max_reference_patches
            ),
            "minimum_candidate_pixels": (
                scan_config.minimum_candidate_pixels
            ),
            "minimum_reference_patches": (
                scan_config.minimum_reference_patches
            ),
            "top_k": scan_config.top_k,
            "minimum_reference_center_distance": (
                2 * scan_config.patch_radius + 1
                if scan_config.minimum_reference_center_distance is None
                else scan_config.minimum_reference_center_distance
            ),
        },
        "normal_zone_score": _stats(normal_scores),
        "defect_zone_score": _stats(defect_scores),
        "paired_zone_score_shift": _stats(paired_shifts),
        "best_candidate_hit_fraction": float(
            np.mean(
                [
                    bool(row["best_candidate_hits_alpha"])
                    for row in pair_rows
                ]
            )
        ),
        "best_distance_to_alpha_px": _stats(
            localization_distances
        ),
        "by_defect": by_defect,
        "interpretation_note": (
            "Scores are maxima over all valid Tip1 candidates. "
            "Do not reuse oracle-based thresholds. Recalibrate on "
            "independent normal masks and validate on actual defects."
        ),
        "artifacts": {
            "normal_results_csv": str(normal_csv_path.resolve()),
            "pair_results_csv": str(pair_csv_path.resolve()),
        },
    }

    summary_path = args.output_dir / "oracle_free_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.review_count > 0:
        defect_by_id = {
            spec.defect_id: spec
            for spec in applicable_specs
        }

        normal_high = _select_unique_sources(
            sorted(
                review_items,
                key=lambda item: item.normal_score,
                reverse=True,
            ),
            args.review_count,
        )

        defect_low = sorted(
            review_items,
            key=lambda item: item.defect_score,
        )[:args.review_count]

        localization_failures = sorted(
            [
                item
                for item in review_items
                if not item.best_candidate_hits_alpha
            ],
            key=lambda item: item.best_distance_to_alpha_px,
            reverse=True,
        )[:args.review_count]

        review_groups = (
            ("review_normal_high_scores", normal_high),
            ("review_defect_low_scores", defect_low),
            ("review_localization_failures", localization_failures),
        )

        for directory_name, selected_items in review_groups:
            review_dir = args.output_dir / directory_name

            for rank, item in enumerate(selected_items, start=1):
                normal = load_soft_mask(item.source_path)
                spec = defect_by_id[item.defect_id]
                alpha = make_defect_alpha(normal.shape, spec)
                defective = inject_missing_metal(normal, alpha)

                normal_result = score_mask_oracle_free(
                    observed=normal,
                    model=model,
                    robust_sigma_floor=(
                        model_config.robust_sigma_floor
                    ),
                    zone=zone,
                    config=scan_config,
                    candidate_center_mask = (candidate_center_mask),
                )
                defect_result = score_mask_oracle_free(
                    observed=defective,
                    model=model,
                    robust_sigma_floor=(
                        model_config.robust_sigma_floor
                    ),
                    zone=zone,
                    config=scan_config,
                    candidate_center_mask = (candidate_center_mask),
                )

                filename = (
                    f"{rank:03d}_"
                    f"normal_{normal_result.zone_score:07.3f}_"
                    f"defect_{defect_result.zone_score:07.3f}_"
                    f"{_safe_name(item.source_path.stem)}_"
                    f"{_safe_name(item.defect_id)}.png"
                )

                save_pair_review(
                    normal=normal,
                    defective=defective,
                    alpha=alpha,
                    zone=zone,
                    normal_result=normal_result,
                    defect_result=defect_result,
                    defect_id=item.defect_id,
                    source_name=item.source_path.name,
                    path=review_dir / filename,
                    patch_radius=scan_config.patch_radius,
                )

    print(
        "Oracle-free scoring complete: "
        f"{len(normal_rows)} normal masks x "
        f"{len(applicable_specs)} Tip1 defects = "
        f"{len(pair_rows)} pairs"
    )
    print(f"Summary: {summary_path.resolve()}")
    print(
        "Normal zone score p99 / max: "
        f"{summary['normal_zone_score']['p99']:.3f} / "
        f"{summary['normal_zone_score']['max']:.3f}"
    )
    print(
        "Defect zone score min / p01: "
        f"{summary['defect_zone_score']['min']:.3f} / "
        f"{summary['defect_zone_score']['p01']:.3f}"
    )
    print(
        "Best candidate hit fraction: "
        f"{summary['best_candidate_hit_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
