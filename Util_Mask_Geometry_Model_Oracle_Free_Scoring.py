"""Oracle-free local-relative scoring for aligned soft masks.

This module contains inference-safe scoring logic only.  It does not accept a
synthetic defect alpha mask, a paired normal image, or a DefectSpec.  Candidate
centres are scanned inside a fixed canonical local zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from Normal_Geometry_Model import GeometryModel


ScoreFeature = Literal[
    "local_signed_excess",
    "local_corrected_top3_sum",
]


@dataclass(frozen=True)
class LocalScanConfig:
    zone_name: str = "tip1"
    score_feature: ScoreFeature = "local_corrected_top3_sum"
    patch_radius: int = 1
    max_reference_patches: int = 8
    minimum_candidate_pixels: int = 3
    minimum_reference_patches: int = 1
    top_k: int = 3
    keep_top_candidates: int = 10
    minimum_reference_center_distance: int | None = None
    minimum_canonical_patch_mass: float = 0.0


@dataclass(frozen=True)
class LocalCandidateScore:
    center_y: int
    center_x: int
    candidate_pixel_count: int
    reference_patch_count: int
    candidate_signed_z_median: float
    reference_signed_z_median: float
    local_signed_excess: float
    local_corrected_top3_sum: float


@dataclass(frozen=True)
class LocalZoneScanResult:
    zone_name: str
    score_feature: ScoreFeature
    scanned_candidate_count: int
    valid_candidate_count: int
    zone_score: float
    best_candidate: LocalCandidateScore | None
    top_candidates: tuple[LocalCandidateScore, ...]
    score_map: np.ndarray


def _validate_scan_config(config: LocalScanConfig) -> None:
    if config.score_feature not in {
        "local_signed_excess",
        "local_corrected_top3_sum",
    }:
        raise ValueError(
            f"Unsupported score_feature: {config.score_feature}"
        )

    if config.patch_radius < 0:
        raise ValueError("patch_radius must be non-negative")

    if config.max_reference_patches <= 0:
        raise ValueError("max_reference_patches must be positive")

    if config.minimum_candidate_pixels <= 0:
        raise ValueError("minimum_candidate_pixels must be positive")

    if config.minimum_reference_patches <= 0:
        raise ValueError("minimum_reference_patches must be positive")

    if (
        config.minimum_reference_patches
        > config.max_reference_patches
    ):
        raise ValueError(
            "minimum_reference_patches cannot exceed "
            "max_reference_patches"
        )

    if config.top_k <= 0:
        raise ValueError("top_k must be positive")

    if config.keep_top_candidates <= 0:
        raise ValueError("keep_top_candidates must be positive")

    if (
        config.minimum_reference_center_distance is not None
        and config.minimum_reference_center_distance <= 0
    ):
        raise ValueError(
            "minimum_reference_center_distance must be positive"
        )
        
    if not np.isfinite(
        config.minimum_canonical_patch_mass
    ):
        raise ValueError(
            "minimum_canonical_patch_mass must be finite"
        )

    if config.minimum_canonical_patch_mass < 0.0:
        raise ValueError(
            "minimum_canonical_patch_mass "
            "must be non-negative"
        )


def compute_signed_z(
    observed: np.ndarray,
    model: GeometryModel,
    robust_sigma_floor: float,
) -> np.ndarray:
    """Express missing-metal residual in normal-population sigma units."""
    if observed.shape != model.median.shape:
        raise ValueError(
            f"Mask shape {observed.shape} does not match "
            f"model shape {model.median.shape}"
        )

    if observed.ndim != 2:
        raise ValueError("observed must be a two-dimensional soft mask")

    if not np.isfinite(observed).all():
        raise ValueError("observed contains non-finite values")

    if robust_sigma_floor <= 0.0:
        raise ValueError("robust_sigma_floor must be positive")

    sigma = np.maximum(
        model.robust_sigma,
        max(float(robust_sigma_floor), 1e-6),
    )

    return (
        (model.median - observed) / sigma
    ).astype(np.float32)


def _patch_bounds(
    shape: tuple[int, int],
    center_y: int,
    center_x: int,
    patch_radius: int,
) -> tuple[int, int, int, int]:
    height, width = shape

    return (
        max(0, center_y - patch_radius),
        min(height, center_y + patch_radius + 1),
        max(0, center_x - patch_radius),
        min(width, center_x + patch_radius + 1),
    )


def _candidate_feature_value(
    candidate: LocalCandidateScore,
    score_feature: ScoreFeature,
) -> float:
    if score_feature == "local_signed_excess":
        return candidate.local_signed_excess

    return candidate.local_corrected_top3_sum


def score_local_candidate_at_center(
    signed_z: np.ndarray,
    zone: np.ndarray,
    center_y: int,
    center_x: int,
    config: LocalScanConfig,
) -> LocalCandidateScore | None:
    """Score one candidate using same-X, Y-direction reference patches."""
    _validate_scan_config(config)

    if signed_z.shape != zone.shape:
        raise ValueError("signed_z and zone must have the same shape")

    if signed_z.ndim != 2:
        raise ValueError("signed_z and zone must be two-dimensional")

    if not np.isfinite(signed_z).all():
        raise ValueError("signed_z contains non-finite values")

    height, width = signed_z.shape

    if not (
        0 <= center_y < height
        and 0 <= center_x < width
    ):
        return None

    y0, y1, x0, x1 = _patch_bounds(
        signed_z.shape,
        center_y,
        center_x,
        config.patch_radius,
    )

    candidate_zone_patch = zone[y0:y1, x0:x1]
    candidate_signed_patch = signed_z[y0:y1, x0:x1]
    candidate_values = candidate_signed_patch[candidate_zone_patch]
    candidate_pixel_count = int(candidate_values.size)

    if candidate_pixel_count < config.minimum_candidate_pixels:
        return None

    candidate_signed_z_median = float(
        np.median(candidate_values)
    )

    minimum_distance = config.minimum_reference_center_distance

    if minimum_distance is None:
        # Two square patches do not overlap when their centre distance is
        # at least 2 * radius + 1 pixels.
        minimum_distance = 2 * config.patch_radius + 1

    reference_candidates: list[tuple[int, int, float]] = []
    reference_rows = np.flatnonzero(zone[:, center_x])

    for ref_y_raw in reference_rows:
        ref_y = int(ref_y_raw)
        distance = abs(ref_y - center_y)

        if distance < minimum_distance:
            continue

        ref_y0, ref_y1, ref_x0, ref_x1 = _patch_bounds(
            signed_z.shape,
            ref_y,
            center_x,
            config.patch_radius,
        )

        reference_zone_patch = zone[
            ref_y0:ref_y1,
            ref_x0:ref_x1,
        ]

        reference_signed_patch = signed_z[
            ref_y0:ref_y1,
            ref_x0:ref_x1,
        ]

        reference_values = reference_signed_patch[
            reference_zone_patch
        ]

        if reference_values.size < config.minimum_candidate_pixels:
            continue

        reference_patch_median = float(
            np.median(reference_values)
        )

        reference_candidates.append(
            (
                distance,
                ref_y,
                reference_patch_median,
            )
        )

    reference_candidates.sort(
        key=lambda item: (item[0], item[1])
    )

    selected_references = reference_candidates[
        :config.max_reference_patches
    ]

    reference_patch_count = len(selected_references)

    if reference_patch_count < config.minimum_reference_patches:
        return None

    reference_signed_z_median = float(
        np.median(
            [
                value
                for _, _, value in selected_references
            ]
        )
    )

    local_signed_excess = float(
        candidate_signed_z_median
        - reference_signed_z_median
    )

    corrected_candidate_values = np.maximum(
        candidate_values - reference_signed_z_median,
        0.0,
    )

    k = min(config.top_k, corrected_candidate_values.size)

    if k > 0:
        top_values = np.partition(
            corrected_candidate_values,
            corrected_candidate_values.size - k,
        )[-k:]
        local_corrected_top3_sum = float(top_values.sum())
    else:
        local_corrected_top3_sum = 0.0

    return LocalCandidateScore(
        center_y=center_y,
        center_x=center_x,
        candidate_pixel_count=candidate_pixel_count,
        reference_patch_count=reference_patch_count,
        candidate_signed_z_median=candidate_signed_z_median,
        reference_signed_z_median=reference_signed_z_median,
        local_signed_excess=local_signed_excess,
        local_corrected_top3_sum=local_corrected_top3_sum,
    )


def scan_local_zone(
    signed_z: np.ndarray,
    zone: np.ndarray,
    canonical_median: np.ndarray,
    config: LocalScanConfig,
    candidate_center_mask = (np.ndarray | None) = None,
) -> LocalZoneScanResult:
    """Scan all eligible centres and return the maximum local score."""
    _validate_scan_config(config)

    if signed_z.shape != zone.shape:
        raise ValueError("signed_z and zone must have the same shape")
    
    if canonical_median.shape != signed_z.shape:
        raise ValueError(
            "canonical_median and signed_z "
            "must have the same shape"
        )

    if not np.isfinite(canonical_median).all():
        raise ValueError(
            "canonical_median contains non-finite values"
        )

    zone = zone.astype(bool, copy=False)

    if not np.any(zone):
        raise ValueError(f"{config.zone_name}: zone mask is empty")

    if candidate_center_mask is None:
        scan_mask = zone
    else:
        if (
            candidate_center_mask.shape
            != signed_z.shape
        ):
            raise ValueError(
                "candidate_center_mask and "
                "signed_z must have the same shape"
            )
    
        scan_mask = (
            candidate_center_mask.astype(
                bool,
                copy=False,
            )
        )
    
        if not np.any(
            scan_mask
        ):
            raise ValueError(
                f"{config.zone_name}: "
                "candidate center mask is empty"
            )
    
    candidate_centers = np.argwhere(
        scan_mask
    )
    score_map = np.full(
        signed_z.shape,
        np.nan,
        dtype=np.float32,
    )

    valid_candidates: list[LocalCandidateScore] = []

    for center_y_raw, center_x_raw in candidate_centers:
        center_y = int(center_y_raw)
        center_x = int(center_x_raw)

        y0, y1, x0, x1 = _patch_bounds(
            signed_z.shape,
            center_y,
            center_x,
            config.patch_radius,
        )

        candidate_zone_patch = zone[
            y0:y1,
            x0:x1,
        ]

        canonical_patch = canonical_median[
            y0:y1,
            x0:x1,
        ]

        canonical_patch_mass = float(
            canonical_patch[
                candidate_zone_patch
            ].sum()
        )

        if (
            canonical_patch_mass
            < config.minimum_canonical_patch_mass
        ):
            continue
        
        candidate = score_local_candidate_at_center(
            signed_z=signed_z,
            zone=zone,
            center_y=center_y
            center_x=center_x,
            config=config,
        )

        if candidate is None:
            continue

        score = _candidate_feature_value(
            candidate,
            config.score_feature,
        )

        score_map[
            candidate.center_y,
            candidate.center_x,
        ] = score

        valid_candidates.append(candidate)

    valid_candidates.sort(
        key=lambda candidate: (
            -_candidate_feature_value(
                candidate,
                config.score_feature,
            ),
            candidate.center_y,
            candidate.center_x,
        )
    )

    if valid_candidates:
        best_candidate = valid_candidates[0]
        zone_score = _candidate_feature_value(
            best_candidate,
            config.score_feature,
        )
        top_candidates = tuple(
            valid_candidates[:config.keep_top_candidates]
        )
    else:
        best_candidate = None
        zone_score = float("nan")
        top_candidates = ()

    return LocalZoneScanResult(
        zone_name=config.zone_name,
        score_feature=config.score_feature,
        scanned_candidate_count=int(candidate_centers.shape[0]),
        valid_candidate_count=len(valid_candidates),
        zone_score=float(zone_score),
        best_candidate=best_candidate,
        top_candidates=top_candidates,
        score_map=score_map,
    )


def score_mask_oracle_free(
    observed: np.ndarray,
    model: GeometryModel,
    robust_sigma_floor: float,
    zone: np.ndarray,
    config: LocalScanConfig,
    candidate_center_mask : (np.ndarray | None) = None,
) -> LocalZoneScanResult:
    """Compute signed-Z and perform inference-safe local-zone scanning."""
    signed_z = compute_signed_z(
        observed=observed,
        model=model,
        robust_sigma_floor=robust_sigma_floor,
    )

    return scan_local_zone(
        signed_z=signed_z,
        zone=zone,
        canonical_median=model.median,
        config=config,
        candidate_center_mask = (candidate_center_mask),
    )


def score_mask_all_zones(
    observed: np.ndarray,
    model: GeometryModel,
    robust_sigma_floor: float,
    zones: dict[str, np.ndarray],
    configs: dict[str, LocalScanConfig],
    candidate_center_masks: dict[
        str,
        np.ndarray,
    ],
) -> dict[str, LocalZoneScanResult]:
    """Score all configured zones from one shared signed-Z map."""
    if not configs:
        raise ValueError(
            "configs must contain at least one zone"
        )

    config_names = set(configs)
    zone_names = set(zones)
    candidate_mask_names = set(
        candidate_center_masks
    )

    missing_zones = (
        config_names
        - zone_names
    )

    missing_candidate_masks = (
        config_names
        - candidate_mask_names
    )

    if missing_zones:
        raise ValueError(
            "Missing zone masks: "
            f"{sorted(missing_zones)}"
        )

    if missing_candidate_masks:
        raise ValueError(
            "Missing candidate-center masks: "
            f"{sorted(missing_candidate_masks)}"
        )

    signed_z = compute_signed_z(
        observed=observed,
        model=model,
        robust_sigma_floor=(
            robust_sigma_floor
        ),
    )

    results: dict[
        str,
        LocalZoneScanResult,
    ] = {}

    for zone_name, config in configs.items():
        if config.zone_name != zone_name:
            raise ValueError(
                "Zone config name mismatch: "
                f"dictionary key={zone_name}, "
                f"config.zone_name="
                f"{config.zone_name}"
            )

        results[zone_name] = (
            scan_local_zone(
                signed_z=signed_z,
                zone=zones[zone_name],
                canonical_median=(
                    model.median
                ),
                config=config,
                candidate_center_mask=(
                    candidate_center_masks[
                        zone_name
                    ]
                ),
            )
        )

    return results