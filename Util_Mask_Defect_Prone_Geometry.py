from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 1. USER SETTINGS
# ============================================================

MODEL_MEDIAN_PATH = Path(
    "/절대경로/normal_geometry_output/model/normal_median.npy"
)

OUTPUT_DIR = Path(
    "/절대경로/defect_prone_geometry"
)


# ------------------------------------------------------------
# Defect-prone 영역을 직접 정의
#
# 좌표계:
#   x → 오른쪽
#   y → 아래
#
# Polygon은 [(x1, y1), (x2, y2), ...] 형식
# Rectangle은 (x, y, width, height) 형식
#
# 필요 없는 쪽은 빈 리스트 []로 두면 됨.
# ------------------------------------------------------------

DEFECT_PRONE_POLYGONS: list[list[tuple[int, int]]] = [
    # 예시:
    # [(160, 65), (175, 65), (175, 85), (160, 85)],
    # [(180, 60), (200, 60), (200, 90), (180, 90)],
]


DEFECT_PRONE_RECTANGLES: list[tuple[int, int, int, int]] = [
    # 예시:
    # (120, 120, 30, 50),
]


# Overlay에서 defect-prone 영역을 얼마나 강하게 표시할지
OVERLAY_ALPHA = 0.55


# ============================================================
# 2. LOAD
# ============================================================

def load_model_median(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Model median does not exist: {path}")

    median = np.load(path).astype(np.float32)

    if median.ndim != 2:
        raise ValueError(
            f"Expected 2-D model median, got shape={median.shape}"
        )

    if not np.isfinite(median).all():
        raise ValueError("Model median contains non-finite values")

    return np.clip(median, 0.0, 1.0)


# ============================================================
# 3. BUILD DEFECT-PRONE MASK
# ============================================================

def build_defect_prone_mask(
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Polygon
    # --------------------------------------------------------
    for polygon_index, points in enumerate(
        DEFECT_PRONE_POLYGONS,
        start=1,
    ):
        if len(points) < 3:
            raise ValueError(
                f"Polygon {polygon_index} must contain "
                "at least 3 points"
            )

        polygon = np.asarray(
            points,
            dtype=np.int32,
        )

        xs = polygon[:, 0]
        ys = polygon[:, 1]

        if (
            np.any(xs < 0)
            or np.any(xs >= width)
            or np.any(ys < 0)
            or np.any(ys >= height)
        ):
            raise ValueError(
                f"Polygon {polygon_index} contains "
                "coordinates outside image bounds"
            )

        cv2.fillPoly(
            mask,
            [polygon],
            color=1,
        )

    # --------------------------------------------------------
    # Rectangle
    # --------------------------------------------------------
    for rectangle_index, (
        x,
        y,
        rect_width,
        rect_height,
    ) in enumerate(
        DEFECT_PRONE_RECTANGLES,
        start=1,
    ):
        if rect_width <= 0 or rect_height <= 0:
            raise ValueError(
                f"Rectangle {rectangle_index} has "
                "non-positive width/height"
            )

        x2 = x + rect_width
        y2 = y + rect_height

        if (
            x < 0
            or y < 0
            or x2 > width
            or y2 > height
        ):
            raise ValueError(
                f"Rectangle {rectangle_index} is "
                "outside image bounds"
            )

        mask[
            y:y2,
            x:x2,
        ] = 1

    if not mask.any():
        raise ValueError(
            "Defect-prone mask is empty. "
            "Define at least one polygon or rectangle."
        )

    return mask


# ============================================================
# 4. SAVE BINARY MASK
# ============================================================

def save_mask_npy(
    mask: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        mask.astype(bool),
    )


def save_mask_png(
    mask: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = (
        mask.astype(np.uint8) * 255
    )

    Image.fromarray(
        image,
        mode="L",
    ).save(path)


# ============================================================
# 5. OVERLAY
# ============================================================

def create_overlay(
    model_median: np.ndarray,
    defect_prone_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "Overlay alpha must be between 0 and 1"
        )

    median_u8 = np.round(
        np.clip(
            model_median,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)

    base_rgb = cv2.cvtColor(
        median_u8,
        cv2.COLOR_GRAY2RGB,
    ).astype(np.float32)

    overlay_rgb = base_rgb.copy()

    selected = defect_prone_mask.astype(bool)

    # Defect-prone 영역을 red overlay로 표시
    overlay_rgb[
        selected
    ] = (
        (1.0 - alpha)
        * overlay_rgb[selected]
        + alpha
        * np.array(
            [255.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    return np.clip(
        overlay_rgb,
        0,
        255,
    ).astype(np.uint8)


def save_overlay(
    overlay: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        overlay,
        mode="RGB",
    ).save(path)


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:
    model_median = load_model_median(
        MODEL_MEDIAN_PATH
    )

    defect_prone_mask = build_defect_prone_mask(
        model_median.shape
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    npy_path = (
        OUTPUT_DIR
        / "defect_prone_mask.npy"
    )

    png_path = (
        OUTPUT_DIR
        / "defect_prone_mask.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / "defect_prone_mask_overlay.png"
    )

    save_mask_npy(
        defect_prone_mask,
        npy_path,
    )

    save_mask_png(
        defect_prone_mask,
        png_path,
    )

    overlay = create_overlay(
        model_median,
        defect_prone_mask,
        OVERLAY_ALPHA,
    )

    save_overlay(
        overlay,
        overlay_path,
    )

    pixel_count = int(
        np.count_nonzero(defect_prone_mask)
    )

    total_pixels = int(
        defect_prone_mask.size
    )

    fraction = (
        pixel_count
        / total_pixels
    )

    print(
        f"Model median shape: "
        f"{model_median.shape}"
    )

    print(
        f"Defect-prone pixels: "
        f"{pixel_count}"
    )

    print(
        f"Defect-prone fraction: "
        f"{fraction:.4%}"
    )

    print(
        f"Saved NPY: "
        f"{npy_path.resolve()}"
    )

    print(
        f"Saved PNG: "
        f"{png_path.resolve()}"
    )

    print(
        f"Saved overlay: "
        f"{overlay_path.resolve()}"
    )


if __name__ == "__main__":
    main()