"""
다채널(5ch) 이진분류 모델(.keras/.h5) 추론 스크립트
- (추가) 자동 ROI 탐색 (Template Matching, 4-Angle) 및 256x256 크롭
- (수정) (256,256,3) RGB + Sobel(1) + Canny(1) = 5채널 텐서 생성
- 학습 파이프라인과 동일하게 [0, 1] float32 텐서를 모델에 전달
- 파일명 오름차순으로 추론 후 CSV [Filename, Result] 저장

필수: TF 2.x, numpy, pandas, opencv-python, pillow
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
ADD_SOBEL = True
ADD_CANNY = True

# 확장자: 필요시 png, bmp 추가
VALID_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"}

# -----------------------------
# (추가) 자동 ROI 탐색 파라미터
# -----------------------------
# 템플릿 크기 (학습된 ROI 크기와 동일해야 함)
TPL_H, TPL_W = 256, 256
TPL_H_HALF, TPL_W_HALF = TPL_H // 2, TPL_W // 2

# 탐색 영역 힌트 (원본 1500x1500 기준)
HINT_X_MIN, HINT_X_MAX = 20, 1224
HINT_Y_MIN, HINT_Y_MAX = 20, 1224
MAX_SHIFT = 20  # 최대 20픽셀 Shift

# -----------------------------
# (추가) 역직렬화를 위한 Custom Layer 정의
# (Training_Script와 동일한 정의)
# -----------------------------
class PreprocessingLayer(layers.Layer):
    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        if backbone_name == 'VGG16':
            self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
        elif backbone_name == 'ResNet50V2':
            self.preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
        elif backbone_name == 'EfficientNetV2S':
            self.preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
        else:
            raise ValueError("Unsupported backbone name")

    def call(self, inputs):
        # (중요) 모델은 [0,1] 입력을 받아 [0,255]로 스케일링 후 Backbone 전처리 수행
        scaled_inputs = inputs * 255.0
        return self.preprocess_fn(scaled_inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"backbone_name": self.backbone_name})
        return config

# -----------------------------
# (추가) 자동 ROI 탐색 헬퍼 함수
# -----------------------------
def find_best_rotated_template_match(img_input_gray, tpl_base_gray):
    """4개 각도로 템플릿 매칭을 수행하고 최적값을 반환"""
    tpl_0 = tpl_base_gray
    tpl_90 = cv2.rotate(tpl_0, cv2.ROTATE_90_CLOCKWISE)
    tpl_180 = cv2.rotate(tpl_0, cv2.ROTATE_180)
    tpl_270 = cv2.rotate(tpl_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    templates = [(tpl_0, 0), (tpl_90, 90), (tpl_180, 180), (tpl_270, 270)]
    best_score = -1.0
    best_top_left = None
    best_angle = None

    for tpl, angle in templates:
        if tpl.shape[0] > img_input_gray.shape[0] or tpl.shape[1] > img_input_gray.shape[1]:
            continue
            
        res = cv2.matchTemplate(img_input_gray, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_top_left = max_loc
            best_angle = angle
            
    return best_score, best_top_left, best_angle

def auto_crop_roi_pil(pil_img_transposed, tpl_base_gray):
    """
    EXIF 처리된 PIL 이미지를 입력받아, 템플릿 매칭 후
    (256, 256) 크기의 0도로 회전된 RGB PIL 이미지를 반환합니다.
    """
    try:
        # 1. PIL(RGB) -> OpenCV(BGR) -> Grayscale (매칭용)
        img_input_color = cv2.cvtColor(np.array(pil_img_transposed), cv2.COLOR_RGB2BGR)
        img_input_gray = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)
        
        # 2. 탐색 영역(Search Region) 정의
        h_img, w_img = img_input_gray.shape
        x_min = max(0, HINT_X_MIN - MAX_SHIFT)
        x_max = min(w_img, HINT_X_MAX + TPL_W + MAX_SHIFT)
        y_min = max(0, HINT_Y_MIN - MAX_SHIFT)
        y_max = min(h_img, HINT_Y_MAX + TPL_H + MAX_SHIFT)
        
        search_region_gray = img_input_gray[y_min:y_max, x_min:x_max]

        # 3. 템플릿 매칭 실행 (탐색 영역 기준)
        best_score, best_top_left_relative, best_angle = find_best_rotated_template_match(
            search_region_gray, tpl_base_gray
        )

        if best_top_left_relative is None:
            print(f"  [WARN] 템플릿 매칭 실패 (탐색 영역이 템플릿보다 작을 수 있음)")
            return None
            
        # 4. 상대 좌표 -> 원본 이미지의 '절대 좌표'로 변환
        best_top_left = (
            best_top_left_relative[0] + x_min,
            best_top_left_relative[1] + y_min
        )
        
        # 5. 원본 PIL 이미지(RGB)에서 Crop (Box: left, upper, right, lower)
        tl_x, tl_y = best_top_left
        br_x, br_y = tl_x + TPL_W, tl_y + TPL_H
        box = (tl_x, tl_y, br_x, br_y)
        cropped_pil_img = pil_img_transposed.crop(box)

        # 6. 매칭된 각도에 따라 0도로 회전
        if best_angle == 0:
            final_cropped_pil_img = cropped_pil_img
        elif best_angle == 90: # 90도(CW) 매칭 -> -90도(CCW) 회전
            final_cropped_pil_img = cropped_pil_img.rotate(90, expand=True)
        elif best_angle == 180:
            final_cropped_pil_img = cropped_pil_img.rotate(180, expand=True)
        elif best_angle == 270: # 270도(CCW) 매칭 -> +90도(CW) 회전
            final_cropped_pil_img = cropped_pil_img.rotate(270, expand=True)
        
        # (256, 256, 3) RGB PIL 이미지 반환
        return final_cropped_pil_img

    except Exception as e:
        print(f"  [ERROR] auto_crop_roi_pil 중 오류: {e}")
        return None

# -----------------------------
# (수정) 전처리 함수들
# -----------------------------

def apply_sobel_tf(image_tensor):
    """(256, 256, 3) [0,1] TF Tensor -> (256, 256, 1) [0,1] Sobel TF Tensor"""
    # Training_Script와 동일한 로직
    gray = tf.image.rgb_to_grayscale(image_tensor)
    gray_4d = tf.expand_dims(gray, axis=0)
    sob = tf.image.sobel_edges(gray_4d)
    sob = tf.squeeze(sob, axis=0)
    gx, gy = sob[..., 0], sob[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy)
    mag_min = tf.reduce_min(mag)
    mag_max = tf.reduce_max(mag)
    mag = (mag - mag_min) / (mag_max - mag_min + 1e-6)
    return mag

def apply_canny_np(image_tensor_np_01):
    """(256, 256, 3) [0,1] Numpy Array -> (256, 256, 1) [0,1] Canny Numpy Array"""
    # Training_Script의 py_function 로직을 NumPy/CV2로 구현
    img_uint8 = (image_tensor_np_01 * 255).astype(np.uint8)
    gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    canny_edge = cv2.Canny(gray_uint8, 100, 200)
    canny_edge_01 = canny_edge.astype(np.float32) / 255.0
    return np.expand_dims(canny_edge_01, axis=-1)

def process_single_image_to_tensor(path_str: str, tpl_base_gray):
    """
    단일 이미지 경로 -> 자동 ROI 크롭 -> (256, 256, 5) float32 [0,1] 텐서 생성
    (실패 시 None 반환)
    """
    try:
        # 1. PIL로 이미지 열기 및 EXIF 처리
        pil_img = Image.open(path_str)
        pil_img_transposed = ImageOps.exif_transpose(pil_img)

        # 2. 자동 ROI 탐색, 크롭, 회전 (결과: 256x256 RGB PIL 이미지)
        cropped_pil_img = auto_crop_roi_pil(pil_img_transposed, tpl_base_gray)
        
        if cropped_pil_img is None:
            print(f"[WARN] ROI 탐색 실패: {path_str}")
            return None, None

        # 3. (256, 256, 3) RGB uint8 -> float32 [0, 1] Numpy Array
        x_rgb_np = np.array(cropped_pil_img)
        x01_np = x_rgb_np.astype(np.float32) / 255.0

        # (검증) (2)번 항목: 모델에 float32 [0,1] 텐서를 전달 준비
        x01_tf = tf.convert_to_tensor(x01_np)
        feats = [x01_tf]

        # 4. Sobel 채널 (TF 연산)
        if ADD_SOBEL:
            sobel_ch = apply_sobel_tf(x01_tf)
            feats.append(sobel_ch)

        # 5. Canny 채널 (NumPy/CV2 연산)
        if ADD_CANNY:
            canny_ch_np = apply_canny_np(x01_np)
            feats.append(tf.convert_to_tensor(canny_ch_np))
        
        # 6. 채널 결합 (256, 256, 5)
        x5 = tf.concat(feats, axis=-1)
        
        return x5, Path(path_str).name

    except Exception as e:
        print(f"[ERROR] 파일 처리 중 예외 발생 {path_str}: {e}")
        return None, None

def iter_batches(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size]

def build_batch_tensor_and_names(paths_chunk, tpl_base_gray):
    """
    파일 경로 리스트 -> (N, 256, 256, 5) 텐서 및 성공한 파일명 리스트 반환
    """
    tensors = []
    basenames = []
    
    for p in paths_chunk:
        # process_single_image_to_tensor는 (텐서, 파일명) 또는 (None, None)을 반환
        x5_tensor, name = process_single_image_to_tensor(str(p), tpl_base_gray)
        
        if x5_tensor is not None:
            tensors.append(x5_tensor)
            basenames.append(name)
        else:
            print(f"  -> 스킵: {p.name}")

    if not tensors:
        return None, []

    batch_tensor = tf.stack(tensors, axis=0) # (N, 256, 256, 5)
    return batch_tensor, basenames

# -----------------------------
# 메인
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="/username/Python/images_inference",
                        help="입력 이미지 폴더 경로")
    parser.add_argument("--template", required=True,
                        help="자동 ROI 탐색용 (256, 256) 템플릿 이미지 경로 (.bmp, .jpg 등)")
    parser.add_argument("--model", default="/username/Python/ImageClassifier_Model_VGG16_final.keras",
                        help="학습 모델(.keras/.h5) 경로")
    parser.add_argument("--out_csv", default="/username/Python/pred_results_auto_roi.csv",
                        help="출력 CSV 경로")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="OK 판정 임계값 (sigmoid 확률)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    template_path = Path(args.template)
    assert images_dir.is_dir(), f"이미지 폴더가 존재하지 않습니다: {images_dir}"
    assert template_path.is_file(), f"템플릿 파일이 존재하지 않습니다: {template_path}"

    # 1. 템플릿 로드 (스크립트 실행 시 1회)
    try:
        tpl_base_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if tpl_base_gray is None:
            raise ValueError("템플릿 이미지를 cv2로 로드할 수 없습니다.")
        if tpl_base_gray.shape[0] != TPL_H or tpl_base_gray.shape[1] != TPL_W:
            print(f"[WARN] 템플릿 크기가 ({TPL_H}, {TPL_W})가 아닙니다. 현재: {tpl_base_gray.shape}")
        print(f"템플릿 로드 완료: {template_path}")
    except Exception as e:
        raise SystemExit(f"[ERROR] 템플릿 로드 실패: {e}")

    # 2. 파일명 오름차순 수집
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix in VALID_EXTS]
    files.sort(key=lambda p: p.name)
    if not files:
        raise SystemExit(f"[ERROR] 유효한 이미지 파일이 없습니다: {images_dir}")

    # 3. 모델 로드 (CustomLayer 포함)
    print(f"모델 로드 중: {args.model}")
    model = load_model(
        args.model,
        custom_objects={"PreprocessingLayer": PreprocessingLayer},
        compile=False
    )

    # (검증) (2)번 항목: 모델 입력은 [0,1]을 받도록 설계되었습니다.
    # PreprocessingLayer가 내부적으로 [0,255] 스케일링을 처리합니다.

    # 4. 입력 채널 확인 (안전장치)
    try:
        in_ch = int(model.input_shape[-1])
    except Exception:
        in_ch = None
        
    expected_ch = 3 + (1 if ADD_SOBEL else 0) + (1 if ADD_CANNY else 0)
    
    if in_ch is not None and in_ch != expected_ch:
        raise SystemExit(f"[ERROR] 모델 입력 채널({in_ch})이 설정({expected_ch})과 다릅니다.")
    print(f"모델 입력 채널 확인: {in_ch}")

    # 5. 예측
    all_results = [] # (Filename, Probability) 튜플 저장

    print(f"\n--- 추론 시작 (총 {len(files)}개 파일, 배치 크기 {args.batch}) ---")
    for chunk in iter_batches(files, args.batch):
        print(f"  ... 배치 처리 중 ({len(chunk)}개)")
        # (수정) 자동 크롭 및 전처리, 배치 생성
        batch_x, batch_names = build_batch_tensor_and_names(chunk, tpl_base_gray)
        
        if batch_x is None: # 배치 전체가 실패한 경우
            print("  -> 배치 스킵 (모든 파일 처리 실패)")
            continue

        # (N, 256, 256, 5) float32 [0,1] 텐서로 추론
        prob_ok = model.predict(batch_x, verbose=0).ravel() # (N,)
        
        all_results.extend(zip(batch_names, prob_ok.tolist()))

    if not all_results:
        raise SystemExit("[ERROR] 모든 이미지 처리/추론에 실패했습니다.")

    # 6. CSV 저장
    df_out = pd.DataFrame(all_results, columns=["Filename", "Probability"])
    df_out["Result"] = np.where(df_out["Probability"] >= args.thresh, "OK", "ESD")
    
    # 원본 파일명 순서가 아닌, 처리에 성공한 파일명 순서대로 저장
    df_out.sort_values(by="Filename", inplace=True)
    
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Saved CSV → {args.out_csv}")
    ok_count = (df_out["Result"] == 'OK').sum()
    esd_count = (df_out["Result"] == 'ESD').sum()
    print(f"Total processed images: {len(df_out)} | OK: {ok_count} | ESD: {esd_count}")
    if len(files) != len(df_out):
        print(f"[INFO] 원본 {len(files)}개 파일 중 {len(files) - len(df_out)}개는 ROI 탐색/처리에 실패하여 제외되었습니다.")

if __name__ == "__main__":
    main()
