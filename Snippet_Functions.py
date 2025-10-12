#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ========== by Gemini ==========

import tensorflow as tf

def apply_sobel_and_threshold_tf(image_batch):
    """
    TensorFlow의 Sobel 함수를 이용해 엣지를 계산하고 임계값을 적용합니다.

    Args:
        image_batch (tf.Tensor): (batch, H, W, 1) 형태의 Grayscale 이미지 텐서.
                                 값의 범위는 0~1 사이 float32 타입이어야 함.

    Returns:
        tf.Tensor: 임계값이 적용된 (batch, H, W, 1) 형태의 이진 엣지 맵.
    """
    # 1. tf.image.sobel_edges() 호출
    # 출력 형태: (batch, H, W, 1, 2). 마지막 차원의 0은 dy, 1은 dx.
    sobel = tf.image.sobel_edges(image_batch)

    # 2. 기울기 크기(Magnitude) 계산
    dy = sobel[..., 0]  # y 방향 기울기
    dx = sobel[..., 1]  # x 방향 기울기
    magnitude = tf.sqrt(tf.square(dy) + tf.square(dx))

    # 3. Magnitude를 0~1 범위로 정규화
    # 각 이미지별로 min-max 정규화를 수행하여 일관된 임계값 적용을 보장
    batch_size = tf.shape(magnitude)[0]
    # (batch, H*W) 형태로 flatten
    mag_flat = tf.reshape(magnitude, [batch_size, -1])
    min_vals = tf.reduce_min(mag_flat, axis=1, keepdims=True)
    max_vals = tf.reduce_max(mag_flat, axis=1, keepdims=True)
    # (batch, 1) -> (batch, 1, 1, 1) 형태로 브로드캐스팅 준비
    min_vals = tf.reshape(min_vals, [batch_size, 1, 1, 1])
    max_vals = tf.reshape(max_vals, [batch_size, 1, 1, 1])
    
    mag_normalized = (magnitude - min_vals) / (max_vals - min_vals + 1e-6)

    # 4. Threshold 적용 (0.5)
    # mag_normalized > 0.5 연산의 결과는 boolean (True/False) 타입임
    binary_edge = mag_normalized > 0.5

    # 5. Boolean 타입을 float32 타입으로 변환 (True->1.0, False->0.0)
    # 딥러닝 모델의 입력 채널로 사용하기 위함
    binary_edge_float = tf.cast(binary_edge, tf.float32)

    return binary_edge_float

# --- 함수 사용 예시 ---
# 1. 임의의 이미지 배치 생성 (4개의 64x64 이미지)
sample_images = tf.random.uniform(shape=[4, 64, 64, 1], minval=0.0, maxval=1.0)

# 2. 함수 호출
sobel_result_with_threshold = apply_sobel_and_threshold_tf(sample_images)

# 3. 결과 텐서의 형태 확인
print("Result tensor shape:", sobel_result_with_threshold.shape)
# 예상 출력: Result tensor shape: (4, 64, 64, 1)

import tensorflow as tf

def apply_sobel_and_threshold_tf(image_batch):
    """
    TensorFlow의 Sobel 함수를 이용해 엣지를 계산하고 임계값을 적용합니다.

    Args:
        image_batch (tf.Tensor): (batch, H, W, 1) 형태의 Grayscale 이미지 텐서.
                                 값의 범위는 0~1 사이 float32 타입이어야 함.

    Returns:
        tf.Tensor: 임계값이 적용된 (batch, H, W, 1) 형태의 이진 엣지 맵.
    """
    # 1. tf.image.sobel_edges() 호출
    # 출력 형태: (batch, H, W, 1, 2). 마지막 차원의 0은 dy, 1은 dx.
    sobel = tf.image.sobel_edges(image_batch)

    # 2. 기울기 크기(Magnitude) 계산
    dy = sobel[..., 0]  # y 방향 기울기
    dx = sobel[..., 1]  # x 방향 기울기
    magnitude = tf.sqrt(tf.square(dy) + tf.square(dx))

    # 3. Magnitude를 0~1 범위로 정규화
    # 각 이미지별로 min-max 정규화를 수행하여 일관된 임계값 적용을 보장
    batch_size = tf.shape(magnitude)[0]
    # (batch, H*W) 형태로 flatten
    mag_flat = tf.reshape(magnitude, [batch_size, -1])
    min_vals = tf.reduce_min(mag_flat, axis=1, keepdims=True)
    max_vals = tf.reduce_max(mag_flat, axis=1, keepdims=True)
    # (batch, 1) -> (batch, 1, 1, 1) 형태로 브로드캐스팅 준비
    min_vals = tf.reshape(min_vals, [batch_size, 1, 1, 1])
    max_vals = tf.reshape(max_vals, [batch_size, 1, 1, 1])
    
    mag_normalized = (magnitude - min_vals) / (max_vals - min_vals + 1e-6)

    # 4. Threshold 적용 (0.5)
    # mag_normalized > 0.5 연산의 결과는 boolean (True/False) 타입임
    binary_edge = mag_normalized > 0.5

    # 5. Boolean 타입을 float32 타입으로 변환 (True->1.0, False->0.0)
    # 딥러닝 모델의 입력 채널로 사용하기 위함
    binary_edge_float = tf.cast(binary_edge, tf.float32)

    return binary_edge_float

# --- 함수 사용 예시 ---
# 1. 임의의 이미지 배치 생성 (4개의 64x64 이미지)
sample_images = tf.random.uniform(shape=[4, 64, 64, 1], minval=0.0, maxval=1.0)

# 2. 함수 호출
sobel_result_with_threshold = apply_sobel_and_threshold_tf(sample_images)

# 3. 결과 텐서의 형태 확인
print("Result tensor shape:", sobel_result_with_threshold.shape)
# 예상 출력: Result tensor shape: (4, 64, 64, 1)


# ========== by ChatGPT ==========
import tensorflow as tf

def sobel_tf(img, threshold=0.5):
    """
    img: Tensor [H,W,1] or [B,H,W,1], float32, [0,1] 범위
    threshold: 0~1 기준 이진화 임계
    """
    # 차원 보정
    if img.ndim == 3:  # [H,W,1]
        img = tf.expand_dims(img, axis=0)  # [1,H,W,1]

    # Sobel edge 계산
    sobel = tf.image.sobel_edges(img)  # [B,H,W,1,2]
    gx = sobel[..., 0]  # dy
    gy = sobel[..., 1]  # dx
    mag = tf.sqrt(tf.square(gx) + tf.square(gy))  # magnitude

    # 정규화
    mag_min = tf.reduce_min(mag, axis=[1,2,3], keepdims=True)
    mag_max = tf.reduce_max(mag, axis=[1,2,3], keepdims=True)
    mag_norm = (mag - mag_min) / (mag_max - mag_min + 1e-8)

    # Threshold 적용
    edge_bin = tf.cast(mag_norm > threshold, tf.float32)

    # [H,W] 단일 이미지 반환
    return tf.squeeze(edge_bin)  # shape: [H,W]
