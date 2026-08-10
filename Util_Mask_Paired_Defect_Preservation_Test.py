"""Test whether constrained alignment suppresses local missing-metal defects.

The input is a folder of trusted-normal soft masks plus the frozen reference
and anchor used by the production alignment experiment. For every normal mask:

1. Align the untouched normal mask to canonical coordinates.
2. Inject a known local missing-metal defect in canonical coordinates.
3. Inverse-warp the injected mask to the original pose.
4. Independently align the injected copy with the same reference and anchor.
5. Measure defect-signal preservation and transform drift.

No company image is copied outside the selected output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image, ImageDraw

from Normal_Mask_Alignment import (
    AlignmentConfig,
    AlignmentResult,
    Transform,
    align_mask,
    discover_masks,
    inverse_warp_soft,
    load_soft_mask,
    parse_rotations,
    validate_config,
    warp_soft,
)


@dataclass(frozen=True)
class DefectSpec:
    defect_id: str
    shape: str
    coordinates: dict[str, Any]
    strength: float = 1.0
    edge_sigma_px: float = 0.8
    evaluation_margin_px: int = 5


@dataclass(frozen=True)
class GateConfig:
    min_p05_operational_preservation: float = 0.90
    max_p95_shift_drift_px: float = 0.50
    max_p95_scale_edge_drift_px: float = 0.50
    max_rotation_change_count: int = 0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: float("nan") for key in ("min", "p05", "median", "p95", "max", "mean")}
    return {
        "min": float(min(values)),
        "p05": _percentile(values, 5),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": float(max(values)),
        "mean": float(np.mean(values)),
    }


def load_defect_config(path: Path) -> tuple[list[DefectSpec], GateConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_defects = payload.get("defects", [])
    if not raw_defects:
        raise ValueError("Defect config must contain at least one item in 'defects'")
    defects: list[DefectSpec] = []
    seen: set[str] = set()
    for raw in raw_defects:
        defect_id = str(raw["id"])
        if defect_id in seen:
            raise ValueError(f"Duplicate defect id: {defect_id}")
        seen.add(defect_id)
        shape = str(raw["shape"]).lower()
        if shape not in {"rectangle", "ellipse", "polygon"}:
            raise ValueError(f"Unsupported shape for {defect_id}: {shape}")
        strength = float(raw.get("strength", 1.0))
        sigma = float(raw.get("edge_sigma_px", 0.8))
        margin = int(raw.get("evaluation_margin_px", 5))
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"{defect_id}: strength must be in (0, 1]")
        if sigma < 0 or margin < 0:
            raise ValueError(f"{defect_id}: sigma and margin must be non-negative")
        defects.append(DefectSpec(defect_id, shape, dict(raw["coordinates"]), strength, sigma, margin))
    gate = GateConfig(**payload.get("candidate_gate", {}))
    return defects, gate


def make_defect_alpha(shape: tuple[int, int], spec: DefectSpec) -> np.ndarray:
    height, width = shape
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    c = spec.coordinates
    if spec.shape == "rectangle":
        x, y = float(c["x"]), float(c["y"])
        w, h = float(c["width"]), float(c["height"])
        if w <= 0 or h <= 0:
            raise ValueError(f"{spec.defect_id}: width/height must be positive")
        draw.rectangle((x, y, x + w, y + h), fill=255)
    elif spec.shape == "ellipse":
        cx, cy = float(c["center_x"]), float(c["center_y"])
        w, h = float(c["width"]), float(c["height"])
        if w <= 0 or h <= 0:
            raise ValueError(f"{spec.defect_id}: width/height must be positive")
        draw.ellipse((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), fill=255)
    else:
        points = [(float(x), float(y)) for x, y in c["points"]]
        if len(points) < 3:
            raise ValueError(f"{spec.defect_id}: polygon requires at least three points")
        draw.polygon(points, fill=255)
    alpha = np.asarray(canvas, dtype=np.float32) / 255.0
    if spec.edge_sigma_px > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), spec.edge_sigma_px)
    return np.clip(alpha * spec.strength, 0.0, 1.0)


def inject_missing_metal(normal: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.clip(normal * (1.0 - alpha), 0.0, 1.0)


def evaluation_zone(alpha: np.ndarray, margin_px: int) -> np.ndarray:
    zone = (alpha > 1e-3).astype(np.uint8)
    if margin_px:
        size = margin_px * 2 + 1
        zone = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return zone.astype(bool)


def _signal(reference_normal: np.ndarray, defective: np.ndarray, zone: np.ndarray) -> tuple[float, float, int]:
    residual = np.maximum(reference_normal - defective, 0.0)
    values = residual[zone]
    return float(values.sum()), float(values.max(initial=0.0)), int(np.count_nonzero(values >= 0.05))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-6 else float("nan")


def evaluate_pair(
    normal_source: np.ndarray,
    reference: np.ndarray,
    anchor: np.ndarray,
    spec: DefectSpec,
    config: AlignmentConfig,
    normal_result: AlignmentResult | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if normal_result is None:
        normal_result = align_mask(normal_source, reference, anchor, config)
    normal_canonical = normal_result.aligned_soft
    alpha = make_defect_alpha(reference.shape, spec)
    fixed_defective = inject_missing_metal(normal_canonical, alpha)
    removed_energy = float(np.maximum(normal_canonical - fixed_defective, 0.0).sum())
    if removed_energy <= 0.05:
        raise ValueError(
            f"{spec.defect_id}: configured defect does not overlap observable metal; "
            "check canonical coordinates"
        )

    defective_source = inverse_warp_soft(fixed_defective, normal_result.transform, normal_source.shape)
    defect_result = align_mask(defective_source, reference, anchor, config)
    reoptimized_defective = defect_result.aligned_soft

    # Interpolation-only comparator: both members use the transform selected by
    # the defective copy. Operational comparator keeps the frozen normal pose,
    # so any alignment-driven absorption remains visible.
    normal_at_defect_transform = warp_soft(normal_source, defect_result.transform, reference.shape)
    zone = evaluation_zone(alpha, spec.evaluation_margin_px)
    fixed_sum, fixed_max, fixed_area = _signal(normal_canonical, fixed_defective, zone)
    operational_sum, operational_max, operational_area = _signal(
        normal_canonical, reoptimized_defective, zone
    )
    intrinsic_sum, intrinsic_max, intrinsic_area = _signal(
        normal_at_defect_transform, reoptimized_defective, zone
    )

    t0, t1 = normal_result.transform, defect_result.transform
    shift_drift = float(np.hypot(t1.shift_x - t0.shift_x, t1.shift_y - t0.shift_y))
    height, width = reference.shape
    scale_edge_drift = float(
        max(abs(t1.scale_x - t0.scale_x) * width / 2.0, abs(t1.scale_y - t0.scale_y) * height / 2.0)
    )
    row: dict[str, Any] = {
        "defect_id": spec.defect_id,
        "shape": spec.shape,
        "injected_removed_energy": removed_energy,
        "fixed_signal_sum": fixed_sum,
        "operational_signal_sum": operational_sum,
        "intrinsic_signal_sum": intrinsic_sum,
        "operational_preservation_ratio": _ratio(operational_sum, fixed_sum),
        "intrinsic_preservation_ratio": _ratio(intrinsic_sum, fixed_sum),
        "fixed_max_depth": fixed_max,
        "operational_max_depth": operational_max,
        "intrinsic_max_depth": intrinsic_max,
        "fixed_area_ge_0_05": fixed_area,
        "operational_area_ge_0_05": operational_area,
        "intrinsic_area_ge_0_05": intrinsic_area,
        "normal_rotation": t0.rotation_degrees,
        "defect_rotation": t1.rotation_degrees,
        "rotation_changed": int(t0.rotation_degrees != t1.rotation_degrees),
        "normal_scale_x": t0.scale_x,
        "normal_scale_y": t0.scale_y,
        "defect_scale_x": t1.scale_x,
        "defect_scale_y": t1.scale_y,
        "normal_shift_x": t0.shift_x,
        "normal_shift_y": t0.shift_y,
        "defect_shift_x": t1.shift_x,
        "defect_shift_y": t1.shift_y,
        "shift_drift_px": shift_drift,
        "scale_edge_drift_px": scale_edge_drift,
        "normal_anchor_rmse": normal_result.anchor_rmse,
        "defect_anchor_rmse": defect_result.anchor_rmse,
        "normal_binary_iou": normal_result.binary_iou,
        "defect_binary_iou": defect_result.binary_iou,
        "normal_phase_response": normal_result.phase_response,
        "defect_phase_response": defect_result.phase_response,
    }
    artifacts = {
        "normal": normal_canonical,
        "fixed_defective": fixed_defective,
        "reoptimized_defective": reoptimized_defective,
        "fixed_residual": np.maximum(normal_canonical - fixed_defective, 0.0),
        "operational_residual": np.maximum(normal_canonical - reoptimized_defective, 0.0),
        "zone": zone.astype(np.float32),
    }
    return row, artifacts


def _gray_rgb(array: np.ndarray) -> np.ndarray:
    gray = np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def save_review(artifacts: dict[str, np.ndarray], row: dict[str, Any], path: Path) -> None:
    panels: list[tuple[str, np.ndarray]] = [
        ("Aligned normal", _gray_rgb(artifacts["normal"])),
        ("Fixed-transform defect", _gray_rgb(artifacts["fixed_defective"])),
        ("Re-optimized defect", _gray_rgb(artifacts["reoptimized_defective"])),
    ]
    for title, key in (("Fixed missing signal", "fixed_residual"), ("After re-alignment", "operational_residual")):
        residual = artifacts[key]
        heat = cv2.applyColorMap(np.round(np.clip(residual * 4.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        panels.append((title, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))
    overlay = _gray_rgb(artifacts["normal"]).astype(np.float32)
    zone = artifacts["zone"] > 0
    overlay[zone] = 0.45 * overlay[zone] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    panels.append(("Evaluation zone", overlay.astype(np.uint8)))

    h, w = artifacts["normal"].shape
    label_h, footer_h = 24, 44
    canvas = Image.new("RGB", (w * 3, (h + label_h) * 2 + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x, y = (index % 3) * w, (index // 3) * (h + label_h)
        canvas.paste(Image.fromarray(panel), (x, y + label_h))
        draw.text((x + 6, y + 5), title, fill="black")
    footer_y = (h + label_h) * 2 + 5
    draw.text(
        (6, footer_y),
        f"{row['source_name']} | {row['defect_id']} | preservation={row['operational_preservation_ratio']:.3f} | "
        f"shift drift={row['shift_drift_px']:.3f}px | scale-edge drift={row['scale_edge_drift_px']:.3f}px",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_summary(rows: list[dict[str, Any]], gate: GateConfig) -> dict[str, Any]:
    metric_names = (
        "operational_preservation_ratio",
        "intrinsic_preservation_ratio",
        "shift_drift_px",
        "scale_edge_drift_px",
    )
    overall = {name: summary_stats([float(row[name]) for row in rows]) for name in metric_names}
    by_defect: dict[str, Any] = {}
    for defect_id in sorted({str(row["defect_id"]) for row in rows}):
        subset = [row for row in rows if row["defect_id"] == defect_id]
        by_defect[defect_id] = {
            "count": len(subset),
            **{name: summary_stats([float(row[name]) for row in subset]) for name in metric_names},
        }
    failed_defects = [
        defect_id
        for defect_id, metrics in by_defect.items()
        if metrics["operational_preservation_ratio"]["p05"] < gate.min_p05_operational_preservation
    ]
    rotation_changes = sum(int(row["rotation_changed"]) for row in rows)
    checks = {
        "each_defect_p05_preservation": not failed_defects,
        "p95_shift_drift": overall["shift_drift_px"]["p95"] <= gate.max_p95_shift_drift_px,
        "p95_scale_edge_drift": overall["scale_edge_drift_px"]["p95"] <= gate.max_p95_scale_edge_drift_px,
        "rotation_change_count": rotation_changes <= gate.max_rotation_change_count,
    }
    return {
        "candidate_gate_pass": all(checks.values()),
        "candidate_gate": asdict(gate),
        "gate_checks": checks,
        "failed_defect_ids": failed_defects,
        "rotation_change_count": rotation_changes,
        "overall": overall,
        "by_defect": by_defect,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-soft-dir", type=Path, required=True)
    parser.add_argument("--reference-soft", type=Path, required=True)
    parser.add_argument("--anchor-mask", type=Path, required=True)
    parser.add_argument("--defect-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--review-count", type=int, default=40)
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--max-scale-deviation", type=float, default=0.04)
    parser.add_argument("--coarse-scale-step", type=float, default=0.02)
    parser.add_argument("--refine-scale-step", type=float, default=0.005)
    parser.add_argument("--max-shift-px", type=float, default=12.0)
    parser.add_argument("--search-size", type=int, default=64)
    parser.add_argument("--binary-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    paths = discover_masks(args.input_soft_dir, args.recursive)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        paths = paths[: args.max_images]
    reference = load_soft_mask(args.reference_soft)
    anchor = load_soft_mask(args.anchor_mask)
    if reference.shape != anchor.shape:
        raise ValueError("Reference and anchor must have identical shapes")
    defects, gate = load_defect_config(args.defect_config)
    config = AlignmentConfig(
        rotations=parse_rotations(args.rotations),
        max_scale_deviation=args.max_scale_deviation,
        coarse_scale_step=args.coarse_scale_step,
        refine_scale_step=args.refine_scale_step,
        max_shift_px=args.max_shift_px,
        search_size=args.search_size,
        binary_threshold=args.binary_threshold,
        reference_iterations=0,
        review_count=0,
    )
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    reviews: list[tuple[float, dict[str, Any], dict[str, np.ndarray]]] = []
    for image_index, path in enumerate(
        tqdm(paths, desc = "Processing masks", unit = "mask"),
        start = 1,
        ):
        normal = load_soft_mask(path)
        if normal.shape != reference.shape:
            raise ValueError(f"Shape mismatch: {path} has {normal.shape}, expected {reference.shape}")
        normal_result = align_mask(normal, reference, anchor, config)
        for spec in defects:
            row, artifacts = evaluate_pair(
                normal, reference, anchor, spec, config, normal_result=normal_result
            )
            row["source_name"] = path.name
            row["source_path"] = str(path.resolve())
            rows.append(row)
            reviews.append((float(row["operational_preservation_ratio"]), row, artifacts))
        print(f"[{image_index}/{len(paths)}] {path.name}")

    csv_path = args.output_dir / "paired_defect_preservation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for index, (_, row, artifacts) in enumerate(sorted(reviews, key=lambda item: item[0])[: args.review_count], start=1):
        safe_defect = str(row["defect_id"]).replace("/", "_")
        save_review(artifacts, row, args.output_dir / "reviews_worst" / f"{index:03d}_{Path(row['source_name']).stem}_{safe_defect}.png")

    analysis = build_summary(rows, gate)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Paired test of local missing-metal preservation under constrained alignment",
        "input_mask_count": len(paths),
        "defect_count_per_mask": len(defects),
        "pair_count": len(rows),
        "reference_soft": str(args.reference_soft.resolve()),
        "anchor_mask": str(args.anchor_mask.resolve()),
        "defect_config": str(args.defect_config.resolve()),
        "alignment_config": asdict(config),
        "measurement_definition": {
            "operational": "positive(aligned_normal_with_T0 - independently_realigned_defect_with_T1) inside defect zone",
            "intrinsic": "positive(normal_with_T1 - defect_with_T1) inside defect zone",
            "interpretation": "operational preservation includes transform absorption; intrinsic preservation isolates resampling loss",
        },
        **analysis,
        "artifacts": {
            "results_csv": str(csv_path.resolve()),
            "worst_review_dir": str((args.output_dir / "reviews_worst").resolve()),
        },
    }
    summary_path = args.output_dir / "paired_defect_preservation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Candidate gate pass: {summary['candidate_gate_pass']}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
