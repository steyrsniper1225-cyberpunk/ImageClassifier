"""
다중 채널(5→3) 이진분류 모델(.keras) 추론 스크립트
- 학습 파이프라인과 동일한 전처리로 (256,256,5) 텐서 생성
- 파일명 오름차순으로 추론 후 CSV [Filename, Result] 저장

필수: TF 2.x, numpy, pandas, opencv-python
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import cv2
from PIL import Image, ImageOps

# -----------------------------
# 전역 파라미터 (학습과 동일)
# -----------------------------
IMG_SIZE = (256, 256)
ROI_X, ROI_Y = 570, 670
ROI_W, ROI_H = 200, 200
PAD = 28
VALID_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png"}

# -----------------------------
# Custom Layer 정의 (모델 로드를 위해 필수)
# -----------------------------
class PreprocessingLayer(layers.Layer):
    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        if backbone_name == "VGG16":
            self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
        elif backbone_name == "ResNet50V2":
            self.preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
        elif backbone_name == "EfficientNetV2S":
            self.preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
        else:
            # 추론 시에는 학습된 모델의 이름을 정확히 모르므로, 기본값을 설정하거나 오류 대신 경고를 출력할 수 있습니다.
            # 하지만 보통은 어떤 모델인지 알고 사용하므로, 여기서는 학습과 동일하게 처리합니다.
            raise ValueError("Unsupported backbone name")

    def call(self, inputs):
        if self.backbone_name in ["VGG16", "ResNet50V2"]:
            scaled_inputs = inputs * 255.0
            return self.preprocess_fn(scaled_inputs)
        else:
            return self.preprocess_fn(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"backbone_name": self.backbone_name})
        return config

# -----------------------------
# 전처리 함수들 (학습과 동일)
# -----------------------------
def decode_image_tf(path_str: str) -> tf.Tensor:
    with Image.open(path_str) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img_np = np.array(img)
        
    img_tensor = tf.convert_to_tensor(img_np)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    return img_tensor # (H, W, 3) float32 [0, 1]

def crop_roi_tf(img: tf.Tensor) -> tf.Tensor:
    """ROI(+PAD) 크롭, float32 [0,1]"""
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    x0 = tf.clip_by_value(ROI_X - PAD, 0, w)
    y0 = tf.clip_by_value(ROI_Y - PAD, 0, h)
    x1 = tf.clip_by_value(ROI_X + ROI_W + PAD, 0, w)
    y1 = tf.clip_by_value(ROI_Y + ROI_H + PAD, 0, h)
    roi = img[y0:y1, x0:x1]
    return roi

def apply_sobel_tf(image_tensor: tf.Tensor) -> tf.Tensor:
    """0~1 RGB 텐서 -> 0~1 Sobel magnitude 텐서"""
    gray = tf.image.rgb_to_grayscale(image_tensor)
    gray_4d = tf.expand_dims(gray, axis = 0)
    sob = tf.image.sobel_edges(gray_4d)
    sob = tf.squeeze(sob, axis = 0)
    gx, gy = sob[..., 0], sob[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy)
    mag_min = tf.reduce_min(mag)
    mag_max = tf.reduce_max(mag)
    mag = (mag - mag_min) / (mag_max - mag_min + 1e-6)
    return mag

def apply_canny_tf(image_tensor_np: np.ndarray) -> np.ndarray:
    """0~1 Numpy 배열 -> 0~1 Canny edge Numpy 배열"""
    img_uint8 = (image_tensor_np * 255).astype(np.uint8)
    gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    canny_edge = cv2.Canny(gray_uint8, 100, 200)
    canny_edge = canny_edge.astype(np.float32) / 255.0
    return np.expand_dims(canny_edge, axis = -1)

def create_input_tensor(img_path: str) -> tf.Tensor:
    """단일 이미지 경로 -> (256,256,5) float32 텐서"""
    x01 = decode_image_tf(img_path)
    x01 = crop_roi_tf(x01)
    sobel_ch = apply_sobel_tf(x01)
    canny_ch = tf.py_function(
        func = apply_canny_tf,
        inp = [x01.numpy()], # py_function은 eager tensor(numpy)를 입력으로 받음
        Tout = tf.float32
    )
    canny_ch.set_shape([IMG_SIZE[0], IMG_SIZE[1], 1])
    
    # RGB, Sobel, Canny 채널 결합
    x5 = tf.concat([x01, sobel_ch, canny_ch], axis = -1)
    return x5

def build_batch_tensor(paths):
    """파일 경로 리스트 -> (N,256,256,5) float32 텐서"""
    tensors = [create_input_tensor(str(p)) for p in paths]
    batch = tf.stack(tensors, axis = 0)
    return batch

def iter_batches(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size]

# -----------------------------
# 메인
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description = "이미지 분류 모델 추론 스크립트")
    parser.add_argument("--images", default = "/username/Python/images", help="입력 이미지 폴더 경로")
    parser.add_argument("--model", required = True, help="학습된 .keras 모델 경로")
    parser.add_argument("--out_csv", default = "pred_results.csv", help = "출력 CSV 경로")
    parser.add_argument("--batch", type = int, default = 16, help = "배치 크기")
    parser.add_argument("--thresh", type = float, default = 0.8, help = "OK 판정 임계값 (sigmoid 확률)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    assert images_dir.is_dir(), f"이미지 폴더가 존재하지 않습니다: {images_dir}"

    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])
    if not files:
        raise SystemExit(f"[ERROR] 유효한 이미지 파일이 없습니다: {images_dir}")

    # 모델 로드 (Custom Layer 등록)
    custom_objects = {"PreprocessingLayer": PreprocessingLayer}
    model = load_model(args.model, custom_objects = custom_objects, compile = False)
    print(f"[INFO] 모델 로드 완료: {args.model}")

    # 입력 채널 확인
    try:
        in_ch = int(model.input_shape[-1])
        print(f"[INFO] 모델 입력 채널: {in_ch}")
    except Exception as e:
        raise SystemExit(f"[ERROR] 모델 입력 형태를 확인할 수 없습니다: {e}")

    # 예측
    basenames = [p.name for p in files]
    preds = []

    print(f"[INFO] 총 {len(files)}개 이미지에 대해 추론을 시작합니다...")
    for chunk in iter_batches(files, args.batch):
        batch_x = build_batch_tensor(chunk)
        if batch_x.shape[-1] != in_ch:
             raise SystemExit(f"[ERROR] 전처리된 텐서의 채널({batch_x.shape[-1]})이 모델 입력 채널({in_ch})과 다릅니다.")
        
        prob_ok = model.predict(batch_x, verbose = 0).ravel()
        preds.extend(prob_ok.tolist())

    preds = np.array(preds, dtype =  np.float32)
    labels = np.where(preds >= args.thresh, "OK", "ESD")

    # CSV 저장
    df_out = pd.DataFrame(
        {"Filename": basenames,
         "Result": labels,
         "Probability_OK": preds}
    )
    df_out.to_csv(args.out_csv, index =  False, encoding = "utf-8-sig")

    print(f"\n[DONE] Saved CSV → {args.out_csv}")
    print(f"Total images: {len(basenames)} | OK: {(labels=='OK').sum()} | ESD: {(labels=='ESD').sum()}")

if __name__ == "__main__":
    main()
