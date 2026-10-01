"""Legacy Main_Training.py의 첫 Multi-ROI baseline.

실행: python Main_Training_MultiROI.py
ROI 크기/좌표/개수/순서는 옆의 multi_roi_common.py 상단 CONFIG에서 수정.
"""
import os
import random
import datetime
import json
import warnings

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from multi_roi_common import (
    CONFIG as ROI_CONFIG, build_feature, build_multi_roi_model,
    validate_config, view_order, set_training_stage,
    transplant_legacy_weights, save_contract,
)


# ==================== Training CONFIG ====================
# ROI 설정은 ROI_CONFIG = multi_roi_common.CONFIG 한 곳에서만 관리합니다.
# 예: INTERNAL_ROIS에 {"name":"extra", "x":100, "y":100,
#                     "width":32, "height":32} 추가 -> local view 1개 추가.
CONFIG = {
    "DATA_ROOT": "/data_home/user/2025/username/Python/TRAIN",
    "WORK_DIR": "/data_home/user/2025/username/Python/EMG_MODEL_MULTI_ROI",
    "BATCH": 2,  # 원본 16. 8 views이면 이미지 2장이 encoder view 16개에 해당.
    "VAL_SPLIT": 0.2,
    "SEED": 42,
    "EPOCHS_STAGE1": 10,
    "EPOCHS_STAGE2": 10,
    "LR_STAGE1": 1e-3,
    "LR_STAGE2": 1e-4,
    "MIXED_PRECISION": True,  # Legacy 유지. CPU만 쓰면 False가 편리합니다.
    "BACKBONE_WEIGHTS": "imagenet",  # Legacy 기본값. 다운로드 없이 학습하려면 None.
    # 선택적 Legacy 이식: 전체 .keras 모델 경로. None이면 Legacy 파일 불필요.
    # 예: "/data_home/user/2025/username/Python/EMG_MODEL/Stage2.keras"
    "LEGACY_MODEL_PATH": None,
    # Legacy의 '5채널 생성 후' flip/brightness/contrast 순서를 유지합니다.
    # 고정 좌표 zone이 비대칭이면 flip으로 defect가 다른 zone으로 이동할 수 있음.
    # 그 경우 False로 하고 공정 비교를 위해 Base도 같은 augmentation으로 학습하세요.
    # view마다 따로 augment하지 않고 global 5채널에 한 번 적용한 뒤 crop합니다.
    "AUG_FLIP_LEFT_RIGHT": True,
    "AUG_BRIGHTNESS_DELTA": 0.05,
    "AUG_CONTRAST": (0.9, 1.1),
}

cls_ok, cls_esd = "OK", "ESD"  # 반드시 Legacy label 유지: ESD=0, OK=1


def list_labeled_files(root):
    def paths(label):
        folder = os.path.join(root, label)
        return sorted(os.path.join(folder, file) for file in os.listdir(folder)
                      if file.lower().endswith((".jpg", ".jpeg", ".png")))
    return [(p, 1) for p in paths(cls_ok)] + [(p, 0) for p in paths(cls_esd)]


def stratified_split(pairs, val_ratio=0.2, seed=42):
    # 원본 정렬 -> 전체 shuffle -> class별 split -> 각 split shuffle 순서 유지.
    by_label = {0: [], 1: []}
    for path, label in pairs:
        by_label[label].append((path, label))
    train, val = [], []
    for label, bucket in by_label.items():
        nv = int(round(len(bucket) * val_ratio))
        if not 0 < nv < len(bucket):
            raise ValueError(f"label={label}: train/val 양쪽에 샘플이 필요합니다.")
        val.extend(bucket[:nv])
        train.extend(bucket[nv:])
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    return train, val


def augment_feature(x, y):
    if CONFIG["AUG_FLIP_LEFT_RIGHT"]:
        x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=CONFIG["AUG_BRIGHTNESS_DELTA"])
    x = tf.image.random_contrast(x, *CONFIG["AUG_CONTRAST"])
    return x, y


def make_ds(pairs, batch=None, shuffle=True):
    paths = [p for p, _ in pairs]
    labels = [label for _, label in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(pairs), seed=CONFIG["SEED"], reshuffle_each_iteration=True)
    ds = ds.map(build_feature, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.map(augment_feature, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch or CONFIG["BATCH"]).prefetch(tf.data.AUTOTUNE)


class ROIModelCheckpoint(ModelCheckpoint):
    """표준 .keras checkpoint에 검증용 ROI 설정을 함께 저장."""
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if os.path.isfile(self.filepath):
            save_contract(self.filepath)


def callbacks_for(path):
    # 원본 monitor/patience/lr schedule 유지.
    return [
        ROIModelCheckpoint(path, monitor="val_accuracy", save_best_only=True,
                           mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    ]


def compile_model(model, lr):
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])


def main():
    validate_config(ROI_CONFIG)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if CONFIG["MIXED_PRECISION"] else "float32")
    seed = CONFIG["SEED"]
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if CONFIG["AUG_FLIP_LEFT_RIGHT"]:
        warnings.warn("Legacy flip이 켜져 있습니다. fixed ROI가 좌우 비대칭이면 CONFIG 설명을 확인하세요.")
    for key in ("EPOCHS_STAGE1", "EPOCHS_STAGE2", "BATCH"):
        if CONFIG[key] < 1:
            raise ValueError(f"{key}는 1 이상이어야 합니다.")

    all_pairs = list_labeled_files(CONFIG["DATA_ROOT"])
    random.Random(seed).shuffle(all_pairs)
    train_list, val_list = stratified_split(all_pairs, CONFIG["VAL_SPLIT"], seed)
    train_ds, val_ds = make_ds(train_list), make_ds(val_list, shuffle=False)
    print(f"Train: {len(train_list)}, Val: {len(val_list)}; ESD=0, OK=1")
    print("View order:", view_order())
    counts = np.bincount([label for _, label in train_list], minlength=2)
    class_weight = {i: float(counts.sum()) / (2 * counts[i]) for i in range(2)}
    print("Class weight:", class_weight)

    # 원본 이미지 크기/전처리 오류는 긴 학습 전에 발견.
    next(iter(train_ds.take(1)))
    next(iter(val_ds.take(1)))
    legacy_path = CONFIG["LEGACY_MODEL_PATH"]
    model, encoder, base = build_multi_roi_model(
        backbone_weights=None if legacy_path else CONFIG["BACKBONE_WEIGHTS"])
    if legacy_path:
        transplant_legacy_weights(legacy_path, encoder, base)

    # 같은 폴더에서 재실행해도 이전 모델을 덮어쓰지 않도록 run 하위 폴더 사용.
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(CONFIG["WORK_DIR"], run_id)
    os.makedirs(run_dir, exist_ok=False)
    pd.DataFrame([(p, label, split) for split, pairs in (("train", train_list), ("val", val_list))
                  for p, label in pairs], columns=["path", "label", "split"]).to_csv(
                      os.path.join(run_dir, "split.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as file:
        json.dump({"training": CONFIG, "roi": ROI_CONFIG, "tensorflow": tf.__version__},
                  file, ensure_ascii=False, indent=2)

    # Legacy 파일 없이도 ImageNet 초기값으로 두 단계 학습 가능.
    # ImageNet도 사용하지 않는 완전 scratch이면 random backbone을 고정하지 않음.
    scratch = not legacy_path and CONFIG["BACKBONE_WEIGHTS"] is None
    set_training_stage(base, "scratch" if scratch else 1)
    compile_model(model, CONFIG["LR_STAGE1"])
    stage1_path = os.path.join(run_dir, "MultiROI_Stage1.keras")
    history1 = model.fit(train_ds, epochs=CONFIG["EPOCHS_STAGE1"], validation_data=val_ds,
                         class_weight=class_weight, callbacks=callbacks_for(stage1_path))
    # Keras 버전에 따른 EarlyStopping 마지막 epoch 복원 차이를 피함.
    model.load_weights(stage1_path)

    set_training_stage(base, "scratch" if scratch else 2)
    compile_model(model, CONFIG["LR_STAGE2"])
    stage2_path = os.path.join(run_dir, "MultiROI_Stage2.keras")
    history2 = model.fit(train_ds, epochs=CONFIG["EPOCHS_STAGE2"], validation_data=val_ds,
                         class_weight=class_weight, callbacks=callbacks_for(stage2_path))
    model.load_weights(stage2_path)

    histories = []
    for stage, history in enumerate((history1, history2), start=1):
        frame = pd.DataFrame(history.history)
        frame.insert(0, "epoch_in_stage", range(1, len(frame) + 1))
        frame.insert(0, "stage", stage)
        histories.append(frame)
    pd.concat(histories, ignore_index=True).to_csv(
        os.path.join(run_dir, "training_log.csv"), index=False, encoding="utf-8-sig")
    final_path = os.path.join(run_dir, "MultiROI_Final.keras")
    model.save(final_path)  # stage 2의 best val_accuracy weight
    save_contract(final_path)
    print(f"Training Complete. Final model: {final_path}")
    print("Inference CONFIG의 MODEL_PATH에 위 경로를 설정하세요. .roi.json도 함께 보관하세요.")


if __name__ == "__main__":
    main()
