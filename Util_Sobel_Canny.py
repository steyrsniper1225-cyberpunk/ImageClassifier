import os
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------
# 제공된 전처리 함수
# -----------------------------------------------------------------

def apply_sobel(image_tensor):
    """0~1 범위의 RGB Tensor -> 0~1 범위의 Sobel magnitude Tensor"""
    gray = tf.image.rgb_to_grayscale(image_tensor) # Sobel은 Grayscale image에 적용, Tensor(256, 256, 1) float32 [0, 1]
    gray_4d = tf.expand_dims(gray, axis = 0)
    sob = tf.image.sobel_edges(gray_4d)
    sob = tf.squeeze(sob, axis = 0)
    gx, gy = sob[..., 0], sob[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy)
    mag_min = tf.reduce_min(mag)
    mag_max = tf.reduce_max(mag)
    mag = (mag - mag_min) / (mag_max - mag_min + 1e-6)
    return mag
    # Tensor(256, 256, 1) float32 [0, 1]

def apply_canny(image_tensor):
    """0~1 범위의 RGB Tensor -> 0~1 범위의 Canny edge Numpy Array"""
    # 입력 'image_tensor'는 [0,1] 범위의 float32 RGB 텐서입니다.
    image_tensor = image_tensor.numpy() # NumPy(256, 256, 3) float32 [0, 1]
    img_uint8 = image_tensor.astype(uint8) # NumPy(256, 256, 3) uint8 [0, 255]
    gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) # NumPy(256, 256, 3) uint8 [0, 255]
    canny_edge = cv2.Canny(gray_uint8, 100, 200) # NumPy(256, 256, 1) uint8 [0, 255]
    canny_edge = canny_edge.astype(np.float32) / 255.0 # NumPy(256, 256, 1) float32 [0, 1]
    return np.expand_dims(canny_edge, axis = -1)
    # NumPy(256, 256, 1) float32 [0, 1]

# -----------------------------------------------------------------
# 메인 실행 스크립트
# -----------------------------------------------------------------

def process_and_save_images(base_dir, save_dir):
    """
    지정된 폴더의 모든 이미지를 읽어 Sobel, Canny 처리를 한 후 
    결과를 .jpg 파일로 저장합니다.
    """
    base_path = Path(base_dir)
    save_path = Path(save_dir)

    # 1. 저장 폴더 생성
    save_path.mkdir(parents=True, exist_ok=True)

    # 2. 처리할 이미지 확장자 (훈련 스크립트 참고)
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # 3. 이미지 파일 목록 가져오기
    image_files = [p for p in base_path.iterdir() if p.is_file() and p.suffix in valid_exts]

    if not image_files:
        print(f"처리할 유효한 이미지가 폴더에 없습니다: {base_path}")
        return

    print(f"총 {len(image_files)}개의 이미지를 찾았습니다. 처리를 시작합니다...")

    for img_file in image_files:
        try:
            # --- 이미지 로드 ---
            img_bytes = tf.io.read_file(str(img_file))
            # decode_image가 JPG, PNG 등을 자동 감지 (channels=3로 RGB 보장)
            img_uint8 = tf.io.decode_image(img_bytes, channels=3)
            # 함수 입력에 맞게 [0, 1] float32 텐서로 변환
            img_tensor_01 = tf.image.convert_image_dtype(img_uint8, tf.float32) # Tensor(256, 256, 3) float32 [0, 1]

            # --- 1. Sobel 적용 및 변환 ---
            sobel_result_01 = apply_sobel(img_tensor_01) # Tensor(256, 256, 1) float32 [0, 1]
            # 저장을 위해 [0, 255] uint8 텐서로 변환
            sobel_img_uint8 = tf.image.convert_image_dtype(sobel_result_01, tf.uint8) # Tensor(256, 256, 1) uint8 [0, 255]

            # --- 2. Canny 적용 및 변환 ---
            canny_result_01 = apply_canny(img_tensor_01) # Numpy(256, 256, 1) float32 [0, 1]
            # 저장을 위해 [0, 255] uint8 텐서로 변환
            canny_img_uint8 = (canny_result_01 * 255.0).astype(np.uint8) # NumPy(256, 256, 1) uint8 [0, 255]
            canny_img_uint8 = tf.convert_to_tensor(canny_img_uint8) # Tensor(256, 256, 1) uint8 [0, 255]

            # --- 3. 파일 저장 ---
            file_stem = img_file.stem  # 파일명 (확장자 제외)

            # Sobel 결과 저장
            sobel_filename = save_path / f"{file_stem}_sobel.jpg"
            sobel_jpeg_bytes = tf.io.encode_jpeg(sobel_img_uint8, quality=95)
            tf.io.write_file(str(sobel_filename), sobel_jpeg_bytes)

            # Canny 결과 저장
            canny_filename = save_path / f"{file_stem}_canny.jpg"
            canny_jpeg_bytes = tf.io.encode_jpeg(canny_img_uint8, quality=95)
            tf.io.write_file(str(canny_filename), canny_jpeg_bytes)

            # print(f"처리 완료: {img_file.name}")

        except Exception as e:
            print(f"오류 발생 ({img_file.name}): {e}")

    print(f"\n모든 작업 완료. 결과 저장 위치: {save_path}")

# --- 경로 설정 및 실행 ---
# Windows 경로는 백슬래시(\) 문제를 피하기 위해 raw string (r"...") 사용을 권장합니다.
base_dir = r"C:\Users\LGPC\Desktop\ROI_Algo"
save_dir = r"C:\Users\LGPC\Desktop\ROI_Algo\imagesave"

process_and_save_images(base_dir, save_dir)
