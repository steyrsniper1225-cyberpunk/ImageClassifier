"""Run oracle-free scoring for all five local geometry zones.

This inference runner does not use synthetic alpha masks, defect
configuration files, injected images, or paired normal images.

For each already-aligned soft mask:
  1. Compute one shared signed-Z map.
  2. Scan Tip1, Tip2, Charge1, Charge2, and Charge3.
  3. Save zone scores and best candidate diagnostics.
  4. Save top-score review images for visual QA.

No OK/NG threshold is applied in this stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from Mask_Geometry_Model_Oracle_Free_Scoring import (
    LocalScanConfig,
    LocalZoneScanResult,
    score_mask_all_zones,
)
from Mask_Normal_Alignment import (
    discover_masks,
    load_soft_mask,
)
from Normal_Geometry_Model import (
    GeometryModel,
)
from Paired_Defect_Preservation_Test import (
    DefectSpec,
    inject_missing_metal,
    load_defect_config,
    make_defect_alpha,
)

ZONE_NAMES = (
    "tip1",
    "tip2",
    "charge1",
    "charge2",
    "charge3",
)


SCORE_FEATURE_BY_ZONE = {
    "tip1": (
        "local_corrected_top3_sum"
    ),
    "tip2": (
        "local_corrected_top3_sum"
    ),
    "charge1": (
        "local_signed_excess"
    ),
    "charge2": (
        "local_signed_excess"
    ),
    "charge3": (
        "local_corrected_top3_sum"
    ),
}


CANDIDATE_CENTER_BOUNDS = {
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
    "charge1": (
        68,
        73,
        186,
        191,
    ),
    "charge2": (
        128,
        133,
        186,
        191,
    ),
    "charge3": (
        248,
        253,
        188,
        193,
    ),
}


MINIMUM_CANONICAL_PATCH_MASS = {
    "tip1": 0.75,
    "tip2": 0.0,
    "charge1": 0.0,
    "charge2": 0.0,
    "charge3": 0.0,
}


@dataclass(frozen=True)
class LoadedGeometry:
    model: GeometryModel
    robust_sigma_floor: float
    model_json_path: Path


@dataclass(frozen=True)
class ReviewCandidate:
    source_path: Path
    zone_name: str
    score_feature: str
    zone_score: float
    best_y: int
    best_x: int
    patch_radius: int


def _stats(
    values: Iterable[float],
) -> dict[str, float]:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

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
        "p01": float(
            np.quantile(array, 0.01)
        ),
        "p05": float(
            np.quantile(array, 0.05)
        ),
        "median": float(
            np.quantile(array, 0.50)
        ),
        "p95": float(
            np.quantile(array, 0.95)
        ),
        "p99": float(
            np.quantile(array, 0.99)
        ),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _load_array(
    model_dir: Path,
    stem: str,
) -> np.ndarray:
    path = model_dir / f"{stem}.npy"

    if not path.is_file():
        raise FileNotFoundError(
            "Missing geometry model artifact: "
            f"{path}"
        )

    array = np.load(
        path,
        allow_pickle=False,
    ).astype(np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2-D array"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{path}: array contains "
            "non-finite values"
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
            "Normal geometry model JSON "
            f"not found: {model_json_path}"
        )

    payload = json.loads(
        model_json_path.read_text(
            encoding="utf-8"
        )
    )

    raw_config = payload.get(
        "config",
        {},
    )

    robust_sigma_floor = float(
        raw_config.get(
            "robust_sigma_floor",
            0.01,
        )
    )

    if (
        not np.isfinite(
            robust_sigma_floor
        )
        or robust_sigma_floor <= 0.0
    ):
        raise ValueError(
            "Invalid robust_sigma_floor: "
            f"{robust_sigma_floor}"
        )

    model_dir = (
        geometry_output_dir
        / "model"
    )

    model = GeometryModel(
        median=_load_array(
            model_dir,
            "normal_median",
        ),
        lower=_load_array(
            model_dir,
            "normal_lower_envelope",
        ),
        upper=_load_array(
            model_dir,
            "normal_upper_envelope",
        ),
        mad=_load_array(
            model_dir,
            "normal_mad",
        ),
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
            "Geometry model artifact "
            f"shapes differ: {sorted(shapes)}"
        )

    if not (
        np.all(
            model.lower
            <= model.median + 1e-7
        )
        and np.all(
            model.median
            <= model.upper + 1e-7
        )
    ):
        raise ValueError(
            "Frozen geometry model has "
            "invalid lower/median/upper ordering"
        )

    return LoadedGeometry(
        model=model,
        robust_sigma_floor=(
            robust_sigma_floor
        ),
        model_json_path=(
            model_json_path
        ),
    )


def _load_zones(
    zones_dir: Path,
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    zones: dict[
        str,
        np.ndarray,
    ] = {}

    for zone_name in ZONE_NAMES:
        path = (
            zones_dir
            / zone_name
            / f"{zone_name}.npy"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {zone_name} zone: "
                f"{path}"
            )

        zone = np.load(
            path,
            allow_pickle=False,
        ).astype(bool)

        if zone.shape != expected_shape:
            raise ValueError(
                f"{zone_name}: zone shape "
                f"{zone.shape} does not match "
                f"{expected_shape}"
            )

        if not np.any(zone):
            raise ValueError(
                f"{zone_name}: zone is empty"
            )

        zones[zone_name] = zone

    return zones


def _build_candidate_center_masks(
    zones: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidate_masks: dict[
        str,
        np.ndarray,
    ] = {}

    for zone_name in ZONE_NAMES:
        zone = zones[zone_name]

        (
            y0,
            y1,
            x0,
            x1,
        ) = CANDIDATE_CENTER_BOUNDS[
            zone_name
        ]

        height, width = zone.shape

        if not (
            0 <= y0 < y1 <= height
            and 0 <= x0 < x1 <= width
        ):
            raise ValueError(
                f"{zone_name}: invalid "
                "candidate bounds "
                f"{(y0, y1, x0, x1)} "
                f"for shape {zone.shape}"
            )

        candidate_mask = np.zeros_like(
            zone,
            dtype=bool,
        )

        candidate_mask[
            y0:y1,
            x0:x1,
        ] = True

        candidate_mask &= zone

        if not np.any(candidate_mask):
            raise RuntimeError(
                f"{zone_name}: candidate-center "
                "mask does not overlap its zone"
            )

        candidate_masks[
            zone_name
        ] = candidate_mask

    return candidate_masks


def _build_scan_configs(
) -> dict[str, LocalScanConfig]:
    configs: dict[
        str,
        LocalScanConfig,
    ] = {}

    for zone_name in ZONE_NAMES:
        configs[zone_name] = (
            LocalScanConfig(
                zone_name=zone_name,
                score_feature=(
                    SCORE_FEATURE_BY_ZONE[
                        zone_name
                    ]
                ),
                patch_radius=1,
                max_reference_patches=8,
                minimum_candidate_pixels=3,
                minimum_reference_patches=1,
                top_k=3,
                keep_top_candidates=10,
                minimum_reference_center_distance=9,
                minimum_canonical_patch_mass=(
                    MINIMUM_CANONICAL_PATCH_MASS[
                        zone_name
                    ]
                ),
            )
        )

    return configs


def _find_target_zone(
    alpha: np.ndarray,
    zones: dict[str, np.ndarray],
) -> str:
    """Find which frozen local zone contains most injected defect mass."""
    alpha_max = float(alpha.max())

    if alpha_max <= 0.0:
        raise ValueError(
            "Injected alpha mask is empty"
        )

    alpha_support = (
        alpha >= alpha_max * 0.05
    )

    overlaps: dict[str, float] = {}

    for zone_name in ZONE_NAMES:
        overlaps[zone_name] = float(
            alpha[
                alpha_support
                & zones[zone_name]
            ].sum()
        )

    target_zone = max(
        overlaps,
        key=overlaps.get,
    )

    if overlaps[target_zone] <= 0.0:
        raise ValueError(
            "Injected defect does not overlap "
            "any configured local zone"
        )

    return target_zone


def _result_fields(
    zone_name: str,
    result: LocalZoneScanResult,
) -> dict[str, Any]:
    prefix = f"{zone_name}_"

    fields: dict[str, Any] = {
        f"{prefix}score_feature": (
            result.score_feature
        ),
        f"{prefix}zone_score": float(
            result.zone_score
        ),
        f"{prefix}scanned_candidate_count": int(
            result.scanned_candidate_count
        ),
        f"{prefix}valid_candidate_count": int(
            result.valid_candidate_count
        ),
    }

    candidate = result.best_candidate

    if candidate is None:
        fields.update(
            {
                f"{prefix}best_y": -1,
                f"{prefix}best_x": -1,
                f"{prefix}candidate_pixel_count": 0,
                f"{prefix}reference_patch_count": 0,
                f"{prefix}candidate_signed_z_median": (
                    float("nan")
                ),
                f"{prefix}reference_signed_z_median": (
                    float("nan")
                ),
                f"{prefix}local_signed_excess": (
                    float("nan")
                ),
                f"{prefix}local_corrected_top3_sum": (
                    float("nan")
                ),
            }
        )

        return fields

    fields.update(
        {
            f"{prefix}best_y": int(
                candidate.center_y
            ),
            f"{prefix}best_x": int(
                candidate.center_x
            ),
            f"{prefix}candidate_pixel_count": int(
                candidate.candidate_pixel_count
            ),
            f"{prefix}reference_patch_count": int(
                candidate.reference_patch_count
            ),
            f"{prefix}candidate_signed_z_median": float(
                candidate.candidate_signed_z_median
            ),
            f"{prefix}reference_signed_z_median": float(
                candidate.reference_signed_z_median
            ),
            f"{prefix}local_signed_excess": float(
                candidate.local_signed_excess
            ),
            f"{prefix}local_corrected_top3_sum": float(
                candidate.local_corrected_top3_sum
            ),
        }
    )

    return fields


def _prefixed_result_fields(
    state: str,
    zone_name: str,
    result: LocalZoneScanResult,
) -> dict[str, Any]:
    raw = _result_fields(
        zone_name,
        result,
    )

    prefix = f"{zone_name}_"

    return {
        f"{state}_{key}": value
        for key, value in raw.items()
    }


def _gray_rgb(
    array: np.ndarray,
) -> np.ndarray:
    gray = np.round(
        np.clip(array, 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)

    return np.repeat(
        gray[..., None],
        3,
        axis=2,
    )


def _save_zone_review(
    mask: np.ndarray,
    zone: np.ndarray,
    candidate_center_mask: np.ndarray,
    item: ReviewCandidate,
    path: Path,
) -> None:
    panel = _gray_rgb(
        mask
    ).astype(np.float32)

    zone_pixels = zone.astype(bool)

    panel[zone_pixels] = (
        0.65 * panel[zone_pixels]
        + 0.35 * np.array(
            [255, 50, 50],
            dtype=np.float32,
        )
    )

    candidate_pixels = (
        candidate_center_mask.astype(bool)
    )

    panel[candidate_pixels] = (
        0.55 * panel[candidate_pixels]
        + 0.45 * np.array(
            [30, 180, 255],
            dtype=np.float32,
        )
    )

    panel = np.clip(
        panel,
        0,
        255,
    ).astype(np.uint8)

    height, width = mask.shape
    header_height = 30
    footer_height = 54

    canvas = Image.new(
        "RGB",
        (
            width,
            header_height
            + height
            + footer_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (8, 8),
        (
            f"{item.zone_name} | "
            f"{item.score_feature}"
        ),
        fill="black",
    )

    canvas.paste(
        Image.fromarray(panel),
        (
            0,
            header_height,
        ),
    )

    radius = item.patch_radius

    x0 = max(
        0,
        item.best_x - radius,
    )

    y0 = max(
        0,
        item.best_y - radius,
    )

    x1 = min(
        width - 1,
        item.best_x + radius,
    )

    y1 = min(
        height - 1,
        item.best_y + radius,
    )

    draw.rectangle(
        (
            x0,
            header_height + y0,
            x1,
            header_height + y1,
        ),
        outline=(255, 230, 0),
        width=2,
    )

    footer_y = (
        header_height
        + height
        + 7
    )

    draw.text(
        (8, footer_y),
        (
            f"Score: {item.zone_score:.6f}  |  "
            f"Best: y={item.best_y}, "
            f"x={item.best_x}"
        ),
        fill="black",
    )

    draw.text(
        (8, footer_y + 21),
        item.source_path.name,
        fill="black",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

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
        "--defect-zones-dir",
        type=Path,
        required=True,
    )
    
    parser.add_argument(
        "--defect-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--review-count",
        type=int,
        default=20,
    )

    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    if args.review_count < 0:
        raise ValueError(
            "--review-count must be "
            "non-negative"
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

    if not paths:
        raise RuntimeError(
            "No aligned soft masks found"
        )

    loaded = load_frozen_geometry(
        args.geometry_output_dir
    )

    model = loaded.model

    zones = _load_zones(
        args.defect_zones_dir,
        model.median.shape,
    )

    candidate_center_masks = (
        _build_candidate_center_masks(
            zones
        )
    )

    configs = _build_scan_configs()
    
    defects, _ = load_defect_config(
        args.defect_config
    )
    
    if not defects:
        raise RuntimeError(
            "No injected defects found "
            "in defect config"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_rows: list[
        dict[str, Any]
    ] = []

    review_candidates: dict[
        str,
        list[ReviewCandidate],
    ] = {
        zone_name: []
        for zone_name in ZONE_NAMES
    }

    for path in tqdm(
        paths,
        desc="Multi-zone injected-defect scoring",
        unit="mask",
    ):
        normal = load_soft_mask(
            path
        )
    
        if normal.shape != model.median.shape:
            raise ValueError(
                f"{path}: mask shape "
                f"{normal.shape} does not "
                f"match {model.median.shape}"
            )
    
        # --------------------------------------------------------
        # 1. Normal을 현재 동결된 5-zone scorer로 1회 평가
        # --------------------------------------------------------
        normal_results = score_mask_all_zones(
            observed=normal,
            model=model,
            robust_sigma_floor=(
                loaded.robust_sigma_floor
            ),
            zones=zones,
            configs=configs,
            candidate_center_masks=(
                candidate_center_masks
            ),
        )
    
        for zone_name in ZONE_NAMES:
            result = normal_results[
                zone_name
            ]
    
            if result.best_candidate is None:
                raise RuntimeError(
                    f"{path}: no valid normal "
                    f"{zone_name} candidate"
                )
    
            if not np.isfinite(
                result.zone_score
            ):
                raise RuntimeError(
                    f"{path}: non-finite normal "
                    f"{zone_name} score"
                )
    
        # --------------------------------------------------------
        # 2. 각 synthetic defect를 같은 normal mask에 주입
        # --------------------------------------------------------
        for spec in defects:
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
                # 이 normal mask에서 해당 defect 위치에
                # observable metal이 없으면 이 pair는 사용하지 않음.
                continue
    
            target_zone = _find_target_zone(
                alpha=alpha,
                zones=zones,
            )
    
            # ----------------------------------------------------
            # 3. Defect mask 역시 5-zone 전체 scorer 실행
            # ----------------------------------------------------
            defect_results = score_mask_all_zones(
                observed=defective,
                model=model,
                robust_sigma_floor=(
                    loaded.robust_sigma_floor
                ),
                zones=zones,
                configs=configs,
                candidate_center_masks=(
                    candidate_center_masks
                ),
            )
    
            row: dict[str, Any] = {
                "source_name": path.name,
                "source_path": str(
                    path.resolve()
                ),
                "defect_id": spec.defect_id,
                "defect_shape": spec.shape,
                "target_zone": target_zone,
                "injected_removed_energy": (
                    injected_removed_energy
                ),
            }
    
            # ----------------------------------------------------
            # 4. Normal / Defect의 5개 zone 결과 전부 저장
            # ----------------------------------------------------
            for zone_name in ZONE_NAMES:
                normal_result = (
                    normal_results[
                        zone_name
                    ]
                )
    
                defect_result = (
                    defect_results[
                        zone_name
                    ]
                )
    
                if defect_result.best_candidate is None:
                    raise RuntimeError(
                        f"{path} / {spec.defect_id}: "
                        f"no valid defect "
                        f"{zone_name} candidate"
                    )
    
                if not np.isfinite(
                    defect_result.zone_score
                ):
                    raise RuntimeError(
                        f"{path} / {spec.defect_id}: "
                        f"non-finite defect "
                        f"{zone_name} score"
                    )
    
                row.update(
                    _prefixed_result_fields(
                        "normal",
                        zone_name,
                        normal_result,
                    )
                )
    
                row.update(
                    _prefixed_result_fields(
                        "defect",
                        zone_name,
                        defect_result,
                    )
                )
    
                row[
                    f"{zone_name}_paired_score_shift"
                ] = float(
                    defect_result.zone_score
                    - normal_result.zone_score
                )
    
            # target zone score를 별도 컬럼으로 저장
            row["normal_target_zone_score"] = float(
                normal_results[
                    target_zone
                ].zone_score
            )
    
            row["defect_target_zone_score"] = float(
                defect_results[
                    target_zone
                ].zone_score
            )
    
            row["target_zone_score_shift"] = float(
                defect_results[
                    target_zone
                ].zone_score
                - normal_results[
                    target_zone
                ].zone_score
            )
    
            pair_rows.append(row)

    csv_path = (
        args.output_dir
        / "multi_zone_injected_defect_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                pair_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(pair_rows)

    review_root = (
        args.output_dir
        / "reviews_top_scores"
    )

    if args.review_count > 0:
        for zone_name in ZONE_NAMES:
            selected = sorted(
                review_candidates[
                    zone_name
                ],
                key=lambda item: (
                    item.zone_score
                ),
                reverse=True,
            )[
                :args.review_count
            ]

            for rank, item in enumerate(
                selected,
                start=1,
            ):
                observed = load_soft_mask(
                    item.source_path
                )

                filename = (
                    f"{rank:03d}_"
                    f"score_{item.zone_score:08.3f}_"
                    f"{item.source_path.stem}.png"
                )

                _save_zone_review(
                    mask=observed,
                    zone=zones[zone_name],
                    candidate_center_mask=(
                        candidate_center_masks[
                            zone_name
                        ]
                    ),
                    item=item,
                    path=(
                        review_root
                        / zone_name
                        / filename
                    ),
                )

    zone_score_stats = {
        zone_name: _stats(
            float(
                row[
                    f"{zone_name}_zone_score"
                ]
            )
            for row in pair_rows
        )
        for zone_name in ZONE_NAMES
    }

    summary = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "purpose": (
            "Five-zone oracle-free "
            "local geometry scoring"
        ),
        "threshold_applied": False,
        "image_count": len(pair_rows),
        "geometry_model_json": str(
            loaded.model_json_path.resolve()
        ),
        "aligned_soft_dir": str(
            args.aligned_soft_dir.resolve()
        ),
        "defect_zones_dir": str(
            args.defect_zones_dir.resolve()
        ),
        "zone_order": list(
            ZONE_NAMES
        ),
        "zone_configs": {
            zone_name: {
                "score_feature": (
                    configs[
                        zone_name
                    ].score_feature
                ),
                "candidate_bounds": list(
                    CANDIDATE_CENTER_BOUNDS[
                        zone_name
                    ]
                ),
                "patch_radius": (
                    configs[
                        zone_name
                    ].patch_radius
                ),
                "maximum_reference_patches": (
                    configs[
                        zone_name
                    ].max_reference_patches
                ),
                "minimum_reference_patches": (
                    configs[
                        zone_name
                    ].minimum_reference_patches
                ),
                "minimum_reference_center_distance": (
                    configs[
                        zone_name
                    ].minimum_reference_center_distance
                ),
                "minimum_canonical_patch_mass": (
                    configs[
                        zone_name
                    ].minimum_canonical_patch_mass
                ),
            }
            for zone_name in ZONE_NAMES
        },
        "zone_score_stats": (
            zone_score_stats
        ),
        "artifacts": {
            "scores_csv": str(
                csv_path.resolve()
            ),
            "review_root": str(
                review_root.resolve()
            ),
        },
    }

    summary_path = (
        args.output_dir
        / "multi_zone_oracle_free_summary.json"
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
        "Multi-zone scoring complete: "
        f"{len(pair_rows)} masks"
    )

    print(
        f"Scores: {csv_path.resolve()}"
    )

    print(
        f"Summary: {summary_path.resolve()}"
    )

    print(
        f"Reviews: {review_root.resolve()}"
    )


if __name__ == "__main__":
    main()