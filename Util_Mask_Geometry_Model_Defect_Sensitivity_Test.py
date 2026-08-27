"""Test defect sensitivity of a frozen Normal Geometry Model.

This test starts from already-aligned trusted-normal soft masks. It does NOT
realign the injected copy and it does NOT refit the geometry model.

For every aligned normal mask and every configured canonical defect:
  1. Score the untouched aligned normal against the frozen geometry model.
  2. Inject the known missing-metal defect directly in canonical coordinates.
  3. Score the injected copy against the same frozen geometry model.
  4. Measure global score changes and local changes inside the defect evaluation zone.
  5. Save CSV/JSON summaries and the weakest-sensitivity review panels.

The purpose is to verify that the frozen normal tolerance envelope does not
absorb minimum defects such as a 1 px tip retreat or a thin boundary notch.
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

from Mask_Normal_Alignment import discover_masks, load_soft_mask
from Normal_Geometry_Model import GeometryModel, ModelConfig, score_mask, validate_config
from Paired_Defect_Preservation_Test import (
    DefectSpec,
    evaluation_zone,
    inject_missing_metal,
    load_defect_config,
    make_defect_alpha,
)


@dataclass(frozen=True)
class LoadedGeometry:
    model: GeometryModel
    config: ModelConfig
    model_json_path: Path


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            key: float("nan")
            for key in ("min", "p01", "p05", "median", "p95", "p99", "max", "mean")
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
        raise FileNotFoundError(f"Missing geometry model artifact: {path}")
    array = np.load(path, allow_pickle=False).astype(np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid geometry model array: {path}")
    return array


def _load_defect_zones(
    zones_dir: Path,
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    zone_names = (
        "charge1",
        "charge2",
        "charge3",
        "tip1",
        "tip2",
    )

    zones: dict[str, np.ndarray] = {}

    for zone_name in zone_names:
        zone_path = zones_dir / zone_name / f"{zone_name}.npy"

        if not zone_path.is_file():
            raise FileNotFoundError(
                f"Missing defect zone mask: {zone_path}"
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

        if not zone.any():
            raise ValueError(
                f"{zone_name}: zone mask is empty"
            )

        zones[zone_name] = zone

    return zones


def load_frozen_geometry(geometry_output_dir: Path) -> LoadedGeometry:
    model_json_path = geometry_output_dir / "normal_geometry_model.json"
    if not model_json_path.is_file():
        raise FileNotFoundError(f"Normal geometry JSON not found: {model_json_path}")

    payload = json.loads(model_json_path.read_text(encoding="utf-8"))
    raw_config = payload.get("config", {})
    config = ModelConfig(
        lower_quantile=float(raw_config.get("lower_quantile", 0.005)),
        upper_quantile=float(raw_config.get("upper_quantile", 0.995)),
        binary_threshold=float(raw_config.get("binary_threshold", 0.50)),
        residual_pixel_threshold=float(raw_config.get("residual_pixel_threshold", 0.05)),
        robust_sigma_floor=float(raw_config.get("robust_sigma_floor", 0.01)),
        review_count=int(raw_config.get("review_count", 40)),
        minimum_fit_count=int(raw_config.get("minimum_fit_count", 20)),
    )
    validate_config(config)

    model_dir = geometry_output_dir / "model"
    model = GeometryModel(
        median=_load_array(model_dir, "normal_median"),
        lower=_load_array(model_dir, "normal_lower_envelope"),
        upper=_load_array(model_dir, "normal_upper_envelope"),
        mad=_load_array(model_dir, "normal_mad"),
        robust_sigma=_load_array(model_dir, "normal_robust_sigma"),
    )

    shapes = {
        model.median.shape,
        model.lower.shape,
        model.upper.shape,
        model.mad.shape,
        model.robust_sigma.shape,
    }
    if len(shapes) != 1:
        raise ValueError(f"Geometry model artifact shapes do not match: {sorted(shapes)}")
    if not (np.all(model.lower <= model.median + 1e-7) and np.all(model.median <= model.upper + 1e-7)):
        raise ValueError("Frozen geometry model has invalid lower/median/upper ordering")

    return LoadedGeometry(model=model, config=config, model_json_path=model_json_path)


def _largest_component(values: np.ndarray, threshold: float) -> tuple[int, float]:
    binary = (values >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0, 0.0
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    selected = labels == label
    return int(selected.sum()), float(values[selected].sum())


def _right_edge_crossing_x(
    mask: np.ndarray,
    zone: np.ndarray,
    y: int,
    level: float,
) -> float | None:
    xs = np.flatnonzero(zone[y])

    if xs.size < 2:
        return None

    values = mask[y, xs]

    above = values >= level

    if not np.any(above):
        return None

    indices = np.flatnonzero(above)
    last_index = int(indices[-1])

    if last_index >= xs.size - 1:
        return None

    x1 = float(xs[last_index])
    x2 = float(xs[last_index + 1])

    v1 = float(values[last_index])
    v2 = float(values[last_index + 1])

    if abs(v2 - v1) < 1e-8:
        return x1

    fraction = (level - v1) / (v2 - v1)

    return x1 + fraction * (x2 - x1)


def _local_signed_boundary_metrics(
    signed_z: np.ndarray,
    alpha: np.ndarray,
    zone: np.ndarray,
    patch_radius: int = 1,
    max_reference_patches: int = 8,
    alpha_support_fraction: float = 0.05,
) -> dict[str, float | int]:
    if signed_z.shape != alpha.shape:
        raise ValueError(
            "signed_z and alpha must have the same shape"
        )

    if signed_z.shape != zone.shape:
        raise ValueError(
            "signed_z and zone must have the same shape"
        )

    zone_weights = np.where(
        zone,
        alpha,
        0.0,
    ).astype(np.float32)

    total_weight = float(
        zone_weights.sum()
    )

    candidate_pixel_count = int(
        np.count_nonzero(
            (alpha > 0.0) & zone
        )
    )

    if total_weight <= 0.0:
        return {
            "candidate_pixel_count": candidate_pixel_count,
            "reference_patch_count": 0,
            "candidate_center_y": -1,
            "candidate_center_x": -1,
            "candidate_signed_z_median": 0.0,
            "reference_signed_z_median": 0.0,
            "local_signed_excess": 0.0,
            "local_corrected_top3_sum": 0.0,
        }

    yy, xx = np.indices(
        signed_z.shape,
        dtype=np.float32,
    )

    center_y = int(
        np.round(
            float(
                (yy * zone_weights).sum()
                / total_weight
            )
        )
    )

    center_x = int(
        np.round(
            float(
                (xx * zone_weights).sum()
                / total_weight
            )
        )
    )

    height, width = signed_z.shape

    def patch_bounds(
        cy: int,
        cx: int,
    ) -> tuple[int, int, int, int]:
        y0 = max(
            0,
            cy - patch_radius,
        )

        y1 = min(
            height,
            cy + patch_radius + 1,
        )

        x0 = max(
            0,
            cx - patch_radius,
        )

        x1 = min(
            width,
            cx + patch_radius + 1,
        )

        return (
            y0,
            y1,
            x0,
            x1,
        )

    (
        candidate_y0,
        candidate_y1,
        candidate_x0,
        candidate_x1,
    ) = patch_bounds(
        center_y,
        center_x,
    )

    candidate_zone_patch = zone[
        candidate_y0:candidate_y1,
        candidate_x0:candidate_x1,
    ]

    candidate_signed_patch = signed_z[
        candidate_y0:candidate_y1,
        candidate_x0:candidate_x1,
    ]

    candidate_values = (
        candidate_signed_patch[
            candidate_zone_patch
        ]
    )

    if candidate_values.size == 0:
        return {
            "candidate_pixel_count": candidate_pixel_count,
            "reference_patch_count": 0,
            "candidate_center_y": center_y,
            "candidate_center_x": center_x,
            "candidate_signed_z_median": 0.0,
            "reference_signed_z_median": 0.0,
            "local_signed_excess": 0.0,
            "local_corrected_top3_sum": 0.0,
        }

    candidate_signed_z_median = float(
        np.median(
            candidate_values
        )
    )

    alpha_max = float(
        alpha.max()
    )

    if alpha_max > 0.0:
        alpha_support = (
            alpha
            >= (
                alpha_max
                * alpha_support_fraction
            )
        )
    else:
        alpha_support = np.zeros_like(
            alpha,
            dtype=bool,
        )

    reference_candidates: list[
        tuple[int, float]
    ] = []

    reference_rows = np.flatnonzero(
        zone[:, center_x]
    )

    for ref_y_raw in reference_rows:
        ref_y = int(
            ref_y_raw
        )

        if (
            abs(
                ref_y - center_y
            )
            <= patch_radius
        ):
            continue

        (
            y0,
            y1,
            x0,
            x1,
        ) = patch_bounds(
            ref_y,
            center_x,
        )

        reference_zone_patch = zone[
            y0:y1,
            x0:x1,
        ]

        if not np.any(
            reference_zone_patch
        ):
            continue

        reference_alpha_patch = (
            alpha_support[
                y0:y1,
                x0:x1,
            ]
        )

        if np.any(
            reference_alpha_patch
            & reference_zone_patch
        ):
            continue

        reference_signed_patch = (
            signed_z[
                y0:y1,
                x0:x1,
            ]
        )

        reference_values = (
            reference_signed_patch[
                reference_zone_patch
            ]
        )

        if reference_values.size == 0:
            continue

        reference_patch_median = float(
            np.median(
                reference_values
            )
        )

        distance = abs(
            ref_y - center_y
        )

        reference_candidates.append(
            (
                distance,
                reference_patch_median,
            )
        )

    reference_candidates.sort(
        key=lambda item: item[0]
    )

    selected_references = (
        reference_candidates[
            :max_reference_patches
        ]
    )

    reference_patch_count = len(
        selected_references
    )

    if reference_patch_count == 0:
        return {
            "candidate_pixel_count": candidate_pixel_count,
            "reference_patch_count": 0,
            "candidate_center_y": center_y,
            "candidate_center_x": center_x,
            "candidate_signed_z_median": (
                candidate_signed_z_median
            ),
            "reference_signed_z_median": 0.0,
            "local_signed_excess": (
                candidate_signed_z_median
            ),
            "local_corrected_top3_sum": 0.0,
        }

    reference_signed_z_median = float(
        np.median(
            [
                value
                for _, value
                in selected_references
            ]
        )
    )

    local_signed_excess = float(
        candidate_signed_z_median
        - reference_signed_z_median
    )

    corrected_candidate_values = (
        np.maximum(
            candidate_values
            - reference_signed_z_median,
            0.0,
        )
    )

    k = min(
        3,
        corrected_candidate_values.size,
    )

    if k > 0:
        top_values = np.partition(
            corrected_candidate_values,
            corrected_candidate_values.size - k,
        )[-k:]

        local_corrected_top3_sum = float(
            top_values.sum()
        )
    else:
        local_corrected_top3_sum = 0.0

    return {
        "candidate_pixel_count": (
            candidate_pixel_count
        ),
        "reference_patch_count": (
            reference_patch_count
        ),
        "candidate_center_y": (
            center_y
        ),
        "candidate_center_x": (
            center_x
        ),
        "candidate_signed_z_median": (
            candidate_signed_z_median
        ),
        "reference_signed_z_median": (
            reference_signed_z_median
        ),
        "local_signed_excess": (
            local_signed_excess
        ),
        "local_corrected_top3_sum": (
            local_corrected_top3_sum
        ),
    }


def _patch_top3_sum(
    robust_z: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int = 1,
) -> float:
    h, w = robust_z.shape

    y0 = max(0, center_y - radius)
    y1 = min(h, center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(w, center_x + radius + 1)

    patch = robust_z[y0:y1, x0:x1].ravel()

    if patch.size == 0:
        return 0.0

    k = min(3, patch.size)

    top_values = np.partition(
        patch,
        patch.size - k,
    )[-k:]

    return float(top_values.sum())


def _robust_z_metrics(
    robust_z_missing: np.ndarray,
    zone: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    masked = np.where(
        zone,
        robust_z_missing,
        0.0,
    ).astype(np.float32)

    values = robust_z_missing[zone]

    largest_area, largest_sum = _largest_component(
        masked,
        threshold,
    )

    if values.size:
        max_value = float(values.max())
        median_value = float(np.median(values))
        p90_value = float(np.quantile(values, 0.90))

        sorted_values = np.sort(values)[::-1]

        top3_sum = float(
            sorted_values[:3].sum()
        )

        top5_sum = float(
            sorted_values[:5].sum()
        )
    else:
        max_value = 0.0
        median_value = 0.0
        p90_value = 0.0
        top3_sum = 0.0
        top5_sum = 0.0

    local_3x3_sum = cv2.boxFilter(
        masked,
        ddepth=-1,
        ksize=(3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )

    valid_centers = zone.astype(bool)

    if np.any(valid_centers):
        max_3x3_sum = float(
            local_3x3_sum[valid_centers].max()
        )
    else:
        max_3x3_sum = 0.0

    return {
        "max": max_value,
        "median": median_value,
        "p90": p90_value,
        "peak_minus_median": max_value - median_value,
        "peak_minus_p90": max_value - p90_value,

        "top3_sum": top3_sum,
        "top5_sum": top5_sum,
        "max_3x3_sum": max_3x3_sum,

        "sum": float(values.sum()),
        "area_ge_threshold": int(
            np.count_nonzero(values >= threshold)
        ),
        "largest_component_sum": largest_sum,
        "largest_component_area": largest_area,
    }


def _zone_metrics(
    missing: np.ndarray,
    zone: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    masked = np.where(zone, missing, 0.0).astype(np.float32)
    values = missing[zone]
    largest_area, largest_sum = _largest_component(masked, threshold)
    return {
        "sum": float(values.sum()),
        "max": float(values.max(initial=0.0)),
        "area": int(np.count_nonzero(values >= threshold)),
        "largest_component_area": largest_area,
        "largest_component_sum": largest_sum,
    }


def evaluate_sensitivity(
    normal: np.ndarray,
    model: GeometryModel,
    config: ModelConfig,
    spec: DefectSpec,
    defect_zones: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
]:
    if normal.shape != model.median.shape:
        raise ValueError(
            f"Mask shape {normal.shape} "
            f"does not match model "
            f"{model.median.shape}"
        )

    alpha = make_defect_alpha(
        normal.shape,
        spec,
    )

    defective = inject_missing_metal(
        normal,
        alpha,
    )

    injected_removed_energy = float(
        np.maximum(
            normal - defective,
            0.0,
        ).sum()
    )

    if injected_removed_energy <= 0.05:
        raise ValueError(
            f"{spec.defect_id}: configured defect "
            "does not overlap observable metal; "
            "check canonical coordinates"
        )

    normal_score = score_mask(
        normal,
        model,
        config,
    )

    defect_score = score_mask(
        defective,
        model,
        config,
    )

    normal_missing = np.maximum(
        model.lower - normal,
        0.0,
    )

    defect_missing = np.maximum(
        model.lower - defective,
        0.0,
    )

    added_missing = np.maximum(
        defect_missing - normal_missing,
        0.0,
    )

    median_missing = np.maximum(
        model.median - defective,
        0.0,
    )

    sigma_floor = max(
        float(
            config.robust_sigma_floor
        ),
        1e-6,
    )

    sigma = np.maximum(
        model.robust_sigma,
        sigma_floor,
    )

    normal_signed_z = (
        model.median - normal
    ) / sigma

    defect_signed_z = (
        model.median - defective
    ) / sigma

    normal_robust_z_missing = (
        np.maximum(
            normal_signed_z,
            0.0,
        )
    )

    defect_robust_z_missing = (
        np.maximum(
            defect_signed_z,
            0.0,
        )
    )

    delta_robust_z_missing = (
        np.maximum(
            defect_robust_z_missing
            - normal_robust_z_missing,
            0.0,
        )
    )

    zone = evaluation_zone(
        alpha,
        spec.evaluation_margin_px,
    )

    robust_z_threshold = 3.0

    normal_zone = _zone_metrics(
        normal_missing,
        zone,
        config.residual_pixel_threshold,
    )

    defect_zone = _zone_metrics(
        defect_missing,
        zone,
        config.residual_pixel_threshold,
    )

    added_zone = _zone_metrics(
        added_missing,
        zone,
        config.residual_pixel_threshold,
    )

    row: dict[str, Any] = {
        "defect_id": (
            spec.defect_id
        ),
        "shape": (
            spec.shape
        ),
        "injected_removed_energy": (
            injected_removed_energy
        ),
    }

    for tip_name in (
        "tip1",
        "tip2",
    ):
        tip_zone = defect_zones[
            tip_name
        ]

        normal_tip_metrics = (
            _local_signed_boundary_metrics(
                normal_signed_z,
                alpha,
                tip_zone,
            )
        )

        defect_tip_metrics = (
            _local_signed_boundary_metrics(
                defect_signed_z,
                alpha,
                tip_zone,
            )
        )

        for (
            metric_name,
            value,
        ) in normal_tip_metrics.items():
            row[
                f"normal_{tip_name}_"
                f"{metric_name}"
            ] = value

        for (
            metric_name,
            value,
        ) in defect_tip_metrics.items():
            row[
                f"defect_{tip_name}_"
                f"{metric_name}"
            ] = value

        row[
            f"paired_{tip_name}_"
            "candidate_signed_z_shift"
        ] = (
            float(
                defect_tip_metrics[
                    "candidate_signed_z_median"
                ]
            )
            - float(
                normal_tip_metrics[
                    "candidate_signed_z_median"
                ]
            )
        )

        row[
            f"paired_{tip_name}_"
            "local_signed_excess_shift"
        ] = (
            float(
                defect_tip_metrics[
                    "local_signed_excess"
                ]
            )
            - float(
                normal_tip_metrics[
                    "local_signed_excess"
                ]
            )
        )

        row[
            f"paired_{tip_name}_"
            "local_corrected_top3_shift"
        ] = (
            float(
                defect_tip_metrics[
                    "local_corrected_top3_sum"
                ]
            )
            - float(
                normal_tip_metrics[
                    "local_corrected_top3_sum"
                ]
            )
        )

    for (
        zone_name,
        zone_mask,
    ) in defect_zones.items():
        normal_zone_robust_z = (
            _robust_z_metrics(
                normal_robust_z_missing,
                zone_mask,
                robust_z_threshold,
            )
        )

        defect_zone_robust_z = (
            _robust_z_metrics(
                defect_robust_z_missing,
                zone_mask,
                robust_z_threshold,
            )
        )

        delta_zone_robust_z = (
            _robust_z_metrics(
                delta_robust_z_missing,
                zone_mask,
                robust_z_threshold,
            )
        )

        for (
            metric_name,
            value,
        ) in normal_zone_robust_z.items():
            row[
                f"normal_{zone_name}"
                f"_robust_z_{metric_name}"
            ] = value

        for (
            metric_name,
            value,
        ) in defect_zone_robust_z.items():
            row[
                f"defect_{zone_name}"
                f"_robust_z_{metric_name}"
            ] = value

        for (
            metric_name,
            value,
        ) in delta_zone_robust_z.items():
            row[
                f"delta_{zone_name}"
                f"_robust_z_{metric_name}"
            ] = value

    score_fields = (
        "center_rmse",
        "mean_robust_z",
        "p99_robust_z",
        "missing_sum",
        "missing_max",
        "missing_area",
        "missing_largest_component_area",
        "missing_largest_component_sum",
        "extra_sum",
        "extra_max",
        "extra_area",
        "extra_largest_component_area",
        "extra_largest_component_sum",
        "transition_fraction",
    )

    for field in score_fields:
        normal_value = (
            normal_score[
                field
            ]
        )

        defect_value = (
            defect_score[
                field
            ]
        )

        row[
            f"normal_{field}"
        ] = normal_value

        row[
            f"defect_{field}"
        ] = defect_value

        row[
            f"delta_{field}"
        ] = (
            float(defect_value)
            - float(normal_value)
        )

    for prefix, metrics in (
        (
            "normal_zone_missing",
            normal_zone,
        ),
        (
            "defect_zone_missing",
            defect_zone,
        ),
        (
            "added_zone_missing",
            added_zone,
        ),
    ):
        for (
            name,
            value,
        ) in metrics.items():
            row[
                f"{prefix}_{name}"
            ] = value

    row[
        "delta_zone_missing_sum"
    ] = (
        float(
            defect_zone["sum"]
        )
        - float(
            normal_zone["sum"]
        )
    )

    row[
        "delta_zone_missing_max"
    ] = (
        float(
            defect_zone["max"]
        )
        - float(
            normal_zone["max"]
        )
    )

    row[
        "delta_zone_missing_area"
    ] = (
        int(
            defect_zone["area"]
        )
        - int(
            normal_zone["area"]
        )
    )

    row[
        "delta_zone_missing_"
        "largest_component_sum"
    ] = (
        float(
            defect_zone[
                "largest_component_sum"
            ]
        )
        - float(
            normal_zone[
                "largest_component_sum"
            ]
        )
    )

    row[
        "delta_zone_missing_"
        "largest_component_area"
    ] = (
        int(
            defect_zone[
                "largest_component_area"
            ]
        )
        - int(
            normal_zone[
                "largest_component_area"
            ]
        )
    )

    artifacts = {
        "normal": (
            normal
        ),
        "defective": (
            defective
        ),
        "normal_missing": (
            normal_missing
        ),
        "defect_missing": (
            defect_missing
        ),
        "added_missing": (
            added_missing
        ),
        "median_missing": (
            median_missing
        ),
        "normal_signed_z": (
            normal_signed_z.astype(
                np.float32
            )
        ),
        "defect_signed_z": (
            defect_signed_z.astype(
                np.float32
            )
        ),
        "normal_robust_z_missing": (
            normal_robust_z_missing
        ),
        "defect_robust_z_missing": (
            defect_robust_z_missing
        ),
        "delta_robust_z_missing": (
            delta_robust_z_missing
        ),
        "zone": (
            zone.astype(
                np.float32
            )
        ),
    }

    return row, artifacts


def _gray_rgb(array: np.ndarray) -> np.ndarray:
    gray = np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _heat_rgb(array: np.ndarray, gain: float = 4.0) -> np.ndarray:
    heat = cv2.applyColorMap(
        np.round(np.clip(array * gain, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def save_review(artifacts: dict[str, np.ndarray], row: dict[str, Any], path: Path) -> None:
    panels: list[tuple[str, np.ndarray]] = [
        ("Aligned normal", _gray_rgb(artifacts["normal"])),
        ("Injected defect", _gray_rgb(artifacts["defective"])),
        ("Normal missing beyond tolerance", _heat_rgb(artifacts["normal_missing"])),
        ("Defect missing beyond tolerance", _heat_rgb(artifacts["defect_missing"])),
        ("Added missing beyond tolerance", _heat_rgb(artifacts["added_missing"])),
        ("Median missing beyond tolerance", _heatmap_rgb(artifacts["median_missing"])),
        ("Robust-Z missing", _heatmap_rgb(artifacts["defect_robust_z_missing"], gain=0.2)),
    ]

    overlay = _gray_rgb(artifacts["normal"]).astype(np.float32)
    zone = artifacts["zone"] > 0
    overlay[zone] = 0.45 * overlay[zone] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    panels.append(("Evaluation zone", overlay.astype(np.uint8)))

    h, w = artifacts["normal"].shape
    label_h, footer_h = 24, 64
    canvas = Image.new("RGB", (w * 3, (h + label_h) * 2 + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = (index % 3) * w
        y = (index // 3) * (h + label_h)
        canvas.paste(Image.fromarray(panel), (x, y + label_h))
        draw.text((x + 6, y + 5), title, fill="black")

    footer_y = (h + label_h) * 2 + 5
    draw.text(
        (6, footer_y),
        f"{row['source_name']} | {row['defect_id']} | "
        f"dZoneSum={float(row['delta_zone_missing_sum']):.3f} | "
        f"addedLargest={float(row['added_zone_missing_largest_component_sum']):.3f} | "
        f"dGlobalLargest={float(row['delta_missing_largest_component_sum']):.3f}",
        fill="black",
    )
    draw.text(
        (6, footer_y + 20),
        f"removed={float(row['injected_removed_energy']):.3f} | "
        f"defectZoneArea={int(row['defect_zone_missing_area'])} | "
        f"addedZoneArea={int(row['added_zone_missing_area'])}",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_score_grade_review(
    artifacts: dict[str, np.ndarray],
    row: dict[str, Any],
    path: Path,
) -> None:
    normal_panel = _gray_rgb(
        artifacts["normal"]
    )

    defect_panel = _gray_rgb(
        artifacts["defective"]
    )

    height, width = artifacts["normal"].shape

    label_height = 32
    footer_height = 90

    canvas_width = width * 2
    canvas_height = (
        height
        + label_height
        + footer_height
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (8, 8),
        "Aligned normal",
        fill="black",
    )

    draw.text(
        (width + 8, 8),
        "Injected defect",
        fill="black",
    )

    canvas.paste(
        Image.fromarray(normal_panel),
        (
            0,
            label_height,
        ),
    )

    canvas.paste(
        Image.fromarray(defect_panel),
        (
            width,
            label_height,
        ),
    )

    separator_x = width

    draw.line(
        (
            separator_x,
            0,
            separator_x,
            label_height + height,
        ),
        fill=(180, 180, 180),
        width=1,
    )

    score_zone_name = str(
    row.get(
        "score_grade_zone",
        "tip1",
        )
    )
    
    normal_score = float(
        row[
            f"normal_{score_zone_name}_"
            "local_corrected_top3_sum"
        ]
    )
    
    defect_score = float(
        row[
            f"defect_{score_zone_name}_"
            "local_corrected_top3_sum"
        ]
    )
    
    normal_local_signed = float(
        row[
            f"normal_{score_zone_name}_"
            "local_signed_excess"
        ]
    )
    
    defect_local_signed = float(
        row[
            f"defect_{score_zone_name}_"
            "local_signed_excess"
        ]
    )

    footer_y = (
        label_height
        + height
        + 8
    )

    draw.text(
        (
            8,
            footer_y,
        ),
        (
            f"Source: {row['source_name']}  |  "
            f"Defect: {row['defect_id']}"
            f"Zine: {score_zone_name}"
        ),
        fill="black",
    )

    draw.text(
        (
            8,
            footer_y + 22,
        ),
        (
            "local_corrected_top3_sum  |  "
            f"Normal: {normal_score:.3f}  |  "
            f"Defect: {defect_score:.3f}"
        ),
        fill="black",
    )

    draw.text(
        (
            8,
            footer_y + 44,
        ),
        (
            "local_signed_excess  |  "
            f"Normal: {normal_local_signed:.3f}  |  "
            f"Defect: {defect_local_signed:.3f}"
        ),
        fill="black",
    )

    draw.text(
        (
            8,
            footer_y + 66,
        ),
        "Visual grade:  NORMAL  /  AMBIGUOUS  /  DEFECT",
        fill="black",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(path)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "injected_removed_energy",
        "delta_missing_sum",
        "delta_missing_max",
        "delta_missing_area",
        "delta_missing_largest_component_sum",
        "delta_missing_largest_component_area",
        "delta_zone_missing_sum",
        "delta_zone_missing_max",
        "delta_zone_missing_area",
        "added_zone_missing_sum",
        "added_zone_missing_max",
        "added_zone_missing_area",
        "added_zone_missing_largest_component_sum",
        "added_zone_missing_largest_component_area",
        "defect_zone_missing_sum",
        "defect_zone_missing_max",
        "defect_zone_missing_area",
        "defect_zone_missing_largest_component_sum",
        "defect_zone_missing_largest_component_area",
    )
    
    local_zone_metric_names: list[str] = []

    for zone_name in (
        "charge1",
        "charge2",
        "charge3",
        "tip1",
        "tip2",
    ):
        for prefix in (
            "normal",
            "defect",
            "delta",
        ):
            local_zone_metric_names.extend(
                [
                    f"{prefix}_{zone_name}_robust_z_max",
                    f"{prefix}_{zone_name}_robust_z_largest_component_sum",
                ]
            )
    
    metric_names = (
        *metric_names,
        *local_zone_metric_names,
    )

    by_defect: dict[str, Any] = {}
    for defect_id in sorted({str(row["defect_id"]) for row in rows}):
        subset = [row for row in rows if row["defect_id"] == defect_id]
        by_defect[defect_id] = {
            "count": len(subset),
            **{
                metric: _stats(float(row[metric]) for row in subset)
                for metric in metric_names
            },
            "fraction_with_added_component": float(
                np.mean(
                    [
                        float(row["added_zone_missing_largest_component_sum"]) > 0.0
                        for row in subset
                    ]
                )
            ),
            "fraction_with_added_pixels_ge_threshold": float(
                np.mean([int(row["added_zone_missing_area"]) > 0 for row in subset])
            ),
        }

    overall = {
        metric: _stats(float(row[metric]) for row in rows)
        for metric in metric_names
    }
    return {
        "overall": overall,
        "by_defect": by_defect,
        "interpretation_note": (
            "No hard pass/fail gate is imposed in v001. Inspect p05/median distributions and "
            "weakest-sensitivity reviews first, then freeze an operating threshold from normal "
            "QA and controlled-defect separation."
        ),
    }


def save_local_zone_summary_tables(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    zone_names = (
        "charge1",
        "charge2",
        "charge3",
        "tip1",
        "tip2",
    )

    metric_names = (
        "robust_z_max",
        "robust_z_largest_component_sum",
    )

    detail_rows: list[dict[str, Any]] = []

    defect_ids = sorted(
        {str(row["defect_id"]) for row in rows}
    )

    for defect_id in defect_ids:
        subset = [
            row
            for row in rows
            if str(row["defect_id"]) == defect_id
        ]

        for zone_name in zone_names:
            for metric_name in metric_names:
                for kind in (
                    "normal",
                    "defect",
                    "delta",
                ):
                    key = (
                        f"{kind}_{zone_name}_{metric_name}"
                    )

                    values = [
                        float(row[key])
                        for row in subset
                    ]

                    stats = _stats(values)

                    detail_rows.append(
                        {
                            "defect_id": defect_id,
                            "zone": zone_name,
                            "kind": kind,
                            "metric": metric_name,
                            "count": len(values),
                            "min": stats["min"],
                            "p05": stats["p05"],
                            "median": stats["median"],
                            "p95": stats["p95"],
                            "p99": stats["p99"],
                            "max": stats["max"],
                        }
                    )

    detail_path = (
        output_dir
        / "local_zone_robust_z_summary.csv"
    )

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(detail_rows[0]),
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    comparison_rows: list[dict[str, Any]] = []

    for defect_id in defect_ids:
        subset = [
            row
            for row in rows
            if str(row["defect_id"]) == defect_id
        ]

        for zone_name in zone_names:
            row_out: dict[str, Any] = {
                "defect_id": defect_id,
                "zone": zone_name,
                "count": len(subset),
            }

            for metric_name in metric_names:
                normal_values = [
                    float(
                        row[
                            f"normal_{zone_name}_{metric_name}"
                        ]
                    )
                    for row in subset
                ]

                defect_values = [
                    float(
                        row[
                            f"defect_{zone_name}_{metric_name}"
                        ]
                    )
                    for row in subset
                ]

                normal_stats = _stats(normal_values)
                defect_stats = _stats(defect_values)

                row_out[
                    f"normal_{metric_name}_p95"
                ] = normal_stats["p95"]

                row_out[
                    f"normal_{metric_name}_p99"
                ] = normal_stats["p99"]

                row_out[
                    f"normal_{metric_name}_max"
                ] = normal_stats["max"]

                row_out[
                    f"defect_{metric_name}_p05"
                ] = defect_stats["p05"]

                row_out[
                    f"defect_{metric_name}_median"
                ] = defect_stats["median"]

                row_out[
                    f"defect_{metric_name}_min"
                ] = defect_stats["min"]

            comparison_rows.append(row_out)

    comparison_path = (
        output_dir
        / "local_zone_normal_vs_defect.csv"
    )

    with comparison_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(comparison_rows[0]),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    return detail_path, comparison_path
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-soft-dir", type=Path, required=True)
    parser.add_argument("--geometry-output-dir", type=Path, required=True)
    parser.add_argument("--defect-config", type=Path, required=True)
    parser.add_argument("--defect-zones-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--review-count", type=int, default=40)
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    if args.review_count < 0:
        raise ValueError(
            "--review-count must be non-negative"
        )

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            "Output directory is not empty: "
            f"{args.output_dir}"
        )

    paths = discover_masks(
        args.aligned_soft_dir,
        args.recursive,
    )

    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(
                "--max-images must be positive"
            )

        paths = paths[
            :args.max_images
        ]

    loaded = load_frozen_geometry(
        args.geometry_output_dir
    )

    model = loaded.model
    config = loaded.config

    defects, _ = load_defect_config(
        args.defect_config
    )

    defect_zones = _load_defect_zones(
        args.defect_zones_dir,
        model.median.shape,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    review_candidates: list[
        tuple[
            float,
            dict[str, Any],
            dict[str, np.ndarray],
        ]
    ] = []

    score_grade_candidates: list[
        tuple[
            float,
            dict[str, Any],
            dict[str, np.ndarray],
        ]
    ] = []

    for path in tqdm(
        paths,
        desc="Geometry defect sensitivity",
        unit="mask",
    ):
        normal = load_soft_mask(
            path
        )

        if normal.shape != model.median.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"{path} has {normal.shape}, "
                f"expected {model.median.shape}"
            )

        for spec in defects:
            row, artifacts = (
                evaluate_sensitivity(
                    normal,
                    model,
                    config,
                    spec,
                    defect_zones,
                )
            )

            row["source_name"] = path.name
            row["source_path"] = str(
                path.resolve()
            )

            rows.append(
                row
            )

            review_candidates.append(
                (
                    float(
                        row[
                            "added_zone_missing_sum"
                        ]
                    ),
                    row,
                    artifacts,
                )
            )

            score_zone_name: str | None = None
            
            for tip_name in (
                "tip1",
                "tip2",
            ):
                candidate_pixel_count = int(
                    row.get(
                        f"defect_{tip_name}_candidate_pixel_count",
                        0,
                    )
                )
            
                if candidate_pixel_count > 0:
                    score_zone_name = tip_name
                    break
            
            if score_zone_name is not None:
                row["score_grade_zone"] = score_zone_name
            
                score_grade_candidates.append(
                    (
                        float(
                            row[
                                f"defect_{score_zone_name}_"
                                "local_corrected_top3_sum"
                            ]
                        ),
                        row,
                        artifacts,
                    )
                )

    if not rows:
        raise RuntimeError(
            "No sensitivity rows were generated"
        )

    csv_path = (
        args.output_dir
        / "geometry_model_defect_sensitivity_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    review_dir = (
        args.output_dir
        / "reviews_weakest_sensitivity"
    )

    if args.review_count > 0:
        defect_ids = sorted(
            {
                str(row["defect_id"])
                for row in rows
            }
        )

        per_defect = max(
            1,
            args.review_count
            // max(
                len(defect_ids),
                1,
            ),
        )

        selected: list[
            tuple[
                float,
                dict[str, Any],
                dict[str, np.ndarray],
            ]
        ] = []

        for defect_id in defect_ids:
            subset = [
                item
                for item in review_candidates
                if item[1]["defect_id"]
                == defect_id
            ]

            selected.extend(
                sorted(
                    subset,
                    key=lambda item: item[0],
                )[
                    :per_defect
                ]
            )

        selected = sorted(
            selected,
            key=lambda item: item[0],
        )[
            :args.review_count
        ]

        for index, (
            _,
            row,
            artifacts,
        ) in enumerate(
            selected,
            start=1,
        ):
            safe_defect = str(
                row["defect_id"]
            ).replace(
                "/",
                "_",
            )

            stem = Path(
                str(
                    row["source_name"]
                )
            ).stem

            save_review(
                artifacts,
                row,
                review_dir
                / (
                    f"{index:03d}_"
                    f"{stem}_"
                    f"{safe_defect}.png"
                ),
            )

    score_grade_dir = (
        args.output_dir
        / "review_score_grade"
    )

    score_grade_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_grade_candidates = sorted(
        score_grade_candidates,
        key=lambda item: item[0],
    )

    score_grade_rows: list[
        dict[str, Any]
    ] = []

    for rank, (
        defect_score,
        row,
        artifacts,
    ) in enumerate(
        score_grade_candidates,
        start=1,
    ):
        safe_defect = str(
            row["defect_id"]
        ).replace(
            "/",
            "_",
        )

        stem = Path(
            str(
                row["source_name"]
            )
        ).stem

        filename = (
            f"{rank:04d}_"
            f"score_{defect_score:07.3f}_"
            f"{stem}_"
            f"{safe_defect}.png"
        )

        save_score_grade_review(
            artifacts,
            row,
            score_grade_dir
            / filename,
        )

        score_grade_rows.append(
            {
                "rank": rank,
                "source_name": (
                    row["source_name"]
                ),
                "defect_id": (
                    row["defect_id"]
                ),
                "normal_local_signed_excess": (
                    row[
                        "normal_tip1_"
                        "local_signed_excess"
                    ]
                ),
                "defect_local_signed_excess": (
                    row[
                        "defect_tip1_"
                        "local_signed_excess"
                    ]
                ),
                "normal_local_corrected_top3_sum": (
                    row[
                        "normal_tip1_"
                        "local_corrected_top3_sum"
                    ]
                ),
                "defect_local_corrected_top3_sum": (
                    row[
                        "defect_tip1_"
                        "local_corrected_top3_sum"
                    ]
                ),
                "visual_grade": "",
                "review_png": filename,
            }
        )

    score_grade_csv_path = (
        score_grade_dir
        / "score_grade_index.csv"
    )

    if score_grade_rows:
        with score_grade_csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    score_grade_rows[0]
                ),
            )

            writer.writeheader()
            writer.writerows(
                score_grade_rows
            )

    analysis = build_summary(
        rows
    )

    (
        local_zone_summary_path,
        local_zone_comparison_path,
    ) = save_local_zone_summary_tables(
        rows,
        args.output_dir,
    )

    summary = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "purpose": (
            "Frozen Normal Geometry Model "
            "defect sensitivity test"
        ),
        "aligned_mask_count": len(
            paths
        ),
        "defect_count_per_mask": len(
            defects
        ),
        "pair_count": len(
            rows
        ),
        "geometry_output_dir": str(
            args.geometry_output_dir.resolve()
        ),
        "geometry_model_json": str(
            loaded.model_json_path.resolve()
        ),
        "geometry_model_config": {
            "lower_quantile": (
                config.lower_quantile
            ),
            "upper_quantile": (
                config.upper_quantile
            ),
            "binary_threshold": (
                config.binary_threshold
            ),
            "residual_pixel_threshold": (
                config.residual_pixel_threshold
            ),
            "robust_sigma_floor": (
                config.robust_sigma_floor
            ),
        },
        "defect_config": str(
            args.defect_config.resolve()
        ),
        "measurement_definition": {
            "normal_missing": (
                "max(frozen_lower_envelope "
                "- aligned_normal, 0)"
            ),
            "defect_missing": (
                "max(frozen_lower_envelope "
                "- injected_defective, 0)"
            ),
            "added_missing": (
                "max(defect_missing "
                "- normal_missing, 0)"
            ),
            "zone": (
                "configured defect support "
                "dilated by "
                "evaluation_margin_px"
            ),
            "tip1_local_corrected_top3": (
                "signed-Z -> nearby Tip1 "
                "reference median subtraction "
                "-> positive clipping "
                "-> candidate top3 sum"
            ),
        },
        **analysis,
        "artifacts": {
            "results_csv": str(
                csv_path.resolve()
            ),
            "local_zone_robust_z_summary_csv": str(
                local_zone_summary_path.resolve()
            ),
            "local_zone_normal_vs_defect_csv": str(
                local_zone_comparison_path.resolve()
            ),
            "weakest_review_dir": str(
                review_dir.resolve()
            ),
            "score_grade_review_dir": str(
                score_grade_dir.resolve()
            ),
            "score_grade_index_csv": (
                str(
                    score_grade_csv_path.resolve()
                )
                if score_grade_rows
                else None
            ),
        },
    }

    summary_path = (
        args.output_dir
        / "geometry_model_defect_sensitivity_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Sensitivity complete: "
        f"{len(paths)} masks x "
        f"{len(defects)} defects = "
        f"{len(rows)} pairs"
    )

    print(
        f"Summary: "
        f"{summary_path.resolve()}"
    )

    print(
        "Score-grade review: "
        f"{score_grade_dir.resolve()} "
        f"({len(score_grade_rows)} images)"
    )


if __name__ == "__main__":
    main()
