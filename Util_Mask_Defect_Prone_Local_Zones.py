from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MODEL_MEDIAN_PATH = Path(
    "/절대경로/normal_median.npy"
)

MASTER_MASK_PATH = Path(
    "/절대경로/defect_prone_mask.npy"
)

OUTPUT_DIR = Path(
    "/절대경로/defect_prone_zones"
)


# ============================================================
# 직접 좌표 입력
# ============================================================

ZONE_POLYGONS: dict[str, list[list[tuple[int, int]]]] = {

    "tip1": [
        # [(x1, y1), (x2, y2), ...]
    ],

    "tip2": [
    ],

    "charge_notch": [
    ],

    "charge_vertical": [
    ],

    "discharge_boundary": [
    ],

    "charge_boundary": [
    ],
}


ZONE_RECTANGLES: dict[str, list[tuple[int, int, int, int]]] = {

    "tip1": [
        # (x, y, width, height)
    ],

    "tip2": [
    ],

    "charge_notch": [
    ],

    "charge_vertical": [
    ],

    "discharge_boundary": [
    ],

    "charge_boundary": [
    ],
}


def build_zone(
    shape: tuple[int, int],
    polygons: list[list[tuple[int, int]]],
    rectangles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)

    for points in polygons:
        if len(points) < 3:
            raise ValueError("Polygon must contain at least 3 points")

        polygon = np.asarray(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)

    for x, y, width, height in rectangles:
        if width <= 0 or height <= 0:
            raise ValueError("Rectangle width/height must be positive")

        mask[
            y:y + height,
            x:x + width,
        ] = 1

    return mask.astype(bool)


def save_zone(
    zone_name: str,
    zone: np.ndarray,
    median: np.ndarray,
) -> None:
    zone_dir = OUTPUT_DIR / zone_name
    zone_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        zone_dir / f"{zone_name}.npy",
        zone,
        allow_pickle=False,
    )

    Image.fromarray(
        zone.astype(np.uint8) * 255,
        mode="L",
    ).save(
        zone_dir / f"{zone_name}.png"
    )

    median_u8 = np.round(
        np.clip(median, 0.0, 1.0) * 255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(
        median_u8,
        cv2.COLOR_GRAY2RGB,
    )

    selected = zone.astype(bool)

    overlay[selected, 0] = 255
    overlay[selected, 1] = (
        overlay[selected, 1].astype(np.float32) * 0.35
    ).astype(np.uint8)
    overlay[selected, 2] = (
        overlay[selected, 2].astype(np.float32) * 0.35
    ).astype(np.uint8)

    Image.fromarray(
        overlay,
        mode="RGB",
    ).save(
        zone_dir / f"{zone_name}_overlay.png"
    )


def main() -> None:
    median = np.load(
        MODEL_MEDIAN_PATH,
        allow_pickle=False,
    ).astype(np.float32)

    master = np.load(
        MASTER_MASK_PATH,
        allow_pickle=False,
    ).astype(bool)

    if median.shape != master.shape:
        raise ValueError(
            f"Shape mismatch: median={median.shape}, "
            f"master={master.shape}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    zone_names = sorted(
        set(ZONE_POLYGONS) | set(ZONE_RECTANGLES)
    )

    for zone_name in zone_names:
        zone = build_zone(
            median.shape,
            ZONE_POLYGONS.get(zone_name, []),
            ZONE_RECTANGLES.get(zone_name, []),
        )

        # local zone은 반드시 master defect-prone 영역 안에만 존재
        zone &= master

        if not zone.any():
            raise ValueError(
                f"{zone_name}: zone is empty after "
                "intersection with master mask"
            )

        save_zone(
            zone_name,
            zone,
            median,
        )

        print(
            f"{zone_name}: {int(zone.sum())} pixels"
        )


if __name__ == "__main__":
    main()