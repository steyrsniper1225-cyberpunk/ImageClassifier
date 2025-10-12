#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os , random, datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50V2, EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ========== Path & Para. ==========
data_root = "/data_home/user/2025/username/Python/imagedata"
cls_ok = "OK"
cls_esd = "ESD"
classes = [cls_esd, cls_ok] # Label : esd(0), ok(1)
class_mode = "binary"

img_size = (256, 256)
BATCH = 16
VAL_SPLIT = 0.2
SEED = 42
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10

# ========== Set ROI ==========
ROI_X, ROI_Y = 570, 670
ROI_W, ROI_H = 200, 200
PAD = 28

ADD_SOBEL = True
ADD_CANNY = True

AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# 원본 : (1500, 1500, 3)
# ROI crop : (200, 200, 3) // 좌측 상단 (0, 0) 기준으로 (570, 670)을 crop의 좌상단으로 잡음
# padding(+28) : (256, 256, 3)
# 최종 채널 구성 : RGB(3) + Sobel(1) + Canny(1) = 5 Channels
# Batch : (16, 256, 256, 5)

# ========== File List & Train/Val Split ==========
def list_labeled_files(root):
    p_ok = os.path.join(root, cls_ok)
    p_esd = os.path.join(root, cls_esd)
    ok_files = [os.path.join(p_ok, f) for f in os.listdir(p_ok) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    esd_files = [os.path.join(p_esd, f) for f in os.listdir(p_esd) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    ok_files.sort(); esd_files.sort()
    return [(p, 1) for p in ok_files] + [(p, 0) for p in esd_files]
    # [(filepath, label), ... ] tuple들을 모은 list
    # label -> OK : 1, ESD : 0으로 부여

all_pairs = list_labeled_files(data_root) # data_root를 읽고 tuple List 생성
random.Random(SEED).shuffle(all_pairs) # tuple List의 원소들을 shuffle

def stratified_split(pairs, val_ratio = 0.2):
    by_label = {0: [], 1: []}
    for p, l in pairs: by_label[l].append((p, l)) # lower case of Alphabet "L"
    train, val = [], []
    for l, bucket in by_label.items():
        n = len(bucket); nv = int(round(n * val_ratio))
        val.extend(bucket[:nv]); train.extend(bucket[nv:])
    random.Random(SEED).shuffle(train); random.Random(SEED).shuffle(val)
    return train, val
    # train, val 둘 다 (filepath, label) tuple을 나눠 갖는 list

train_list, val_list = stratified_split(all_pairs, VAL_SPLIT)

print(f"Train : {len(train_list)} Val : {len(val_list)}")
print("Class_Indices : ", {cls_esd : 0, cls_ok : 1})

# ========== Preprocessing (TF Calculation) ==========
def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3) # JPG -> (1500, 1500, 3) tensor(uint8)
    img = tf.image.convert_image_dtype(img, tf.float32) # (1500, 1500, 3) (float32)
    return img
    # (1500, 1500, 3) tensor (float32)

def crop_roi(img):
    h, w = tf.shape(img)[0], tf.shape(img)[1] # h : 1500, w : 1500
    x0 = tf.clip_by_value(ROI_X - PAD, 0, w) # (570 - 28, 0, 1500)
    y0 = tf.clip_by_value(ROI_Y - PAD, 0, h) # (670 - 28, 0, 1500)
    x1 = tf.clip_by_value(ROI_X + ROI_W + PAD, 0, w) # (570 + 200 + 28, 0, 1500)
    y1 = tf.clip_by_value(ROI_Y + ROI_H + PAD, 0, h) # (670 + 200 + 28, 0, 1500)
    roi = img[y0:y1, x0:x1] # img[642:898, 542:798] -> (256, 256, 3) cropped
    return roi
    # (256, 256, 3) tensor (float32) pixel : 0~255

def apply_sobel(image_tensor):
    """0~1 범위의 RGB Tensor -> 0~1 범위의 Sobel magnitude Tensor"""
    gray = tf.image.rgb_to_grayscale(image_tensor) # Sobel은 Grayscale image에 적용
    gray_4d = tf.expand_dims(gray, axis = 0)
    sob = tf.image.sobel_edges(gray_4d) # (1, 256, 256, 1, 2)
    sob = tf.squeeze(sob, axis = 0) # (256, 256, 1, 2)
    gx, gy = sob[..., 0], sob[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy) # (256, 256, 1)
    mag_min = tf.reduce_min(mag)
    mag_max = tf.reduce_max(mag)
    mag = (mag - mag_min) / (mag_max - mag_min + 1e-6)
    return mag

def apply_canny(image_tensor_np):
    image_tensor_np = image_tensor_np.numpy()
    img_uint8 = tf.cast(image_tensor_np * 255, dtype = tf.uint8).numpy()
    gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    canny_edge = cv2.Canny(gray_uint8, 100, 200)
    canny_edge = canny_edge.astype(np.float32) / 255.0
    return np.expand_dims(canny_edge, axis = -1)

def build_feature(path, label):
    img = decode_image(path) # 이미지(JPG)를 읽고 (1500, 1500, 3) tensor로 변환 (uint8)
    x01 = crop_roi(img) # ROI 설정에 따라 tensor를 잘라냄 (256, 256, 3) (float32)

    feats = [x01]

    if ADD_SOBEL:
        sobel_ch = apply_sobel(x01)
        feats.append(sobel_ch)

    if ADD_CANNY:
        canny_ch = tf.py_function(
            func = apply_canny,
            inp = [x01],
            Tout = tf.float32)
        canny_ch.set_shape((img_size[0], img_size[1], 1))
        feats.append(canny_ch)

    # RGB(3) + Sobel(1) + Canny(1) = 5 Channels
    feat = tf.concat(feats, axis = -1)

    return feat, tf.cast(label, tf.float32)
    # feat : (256, 256, 5) (float32) 0 ~ 1 사이의 값들
    # label : () float32 0 or 1의 값들

def make_ds(pairs, batch = BATCH, shuffle = True):
    paths = [p for p,_ in pairs] # (filepath, label)에서 filepath(...JPG)
    labels = [l for _,l in pairs] # (filepath, label)에서 label(0 or 1)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # 두 list를 from_tensor_slices가 tensor로 묶음. datatype : tf.data.Dataset
    # list를 tensor로 묶은 객체

    if shuffle:
        ds = ds.shuffle(len(pairs), seed = SEED, reshuffle_each_iteration = True)

    ds = ds.map(build_feature, num_parallel_calls = AUTOTUNE)
    # [ feat : (256, 256, 4)(tensor), label : ()(scalar, 0.0 or 1.0의 값임) ]

    def augmentation(x, y):
        # x : (256, 256, 5) Tensor 전체에 Augmentation 적용
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta = 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        k = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
        x = tf.image.rot90(x, k = k)
        return x, y

    if shuffle:
        ds = ds.map(augmentation, num_parallel_calls = AUTOTUNE)

    ds = ds.batch(batch).prefetch(AUTOTUNE)
    # x(feat) : (20, 256, 256, 5) -> train은 shuffle=True라서 aug도 진행
    # y(label) : (20, )              val은 shuffle=False라서 aug를 skip

    return ds

def history_to_df(history):
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df

train_ds = make_ds(train_list, shuffle = True)
val_ds = make_ds(val_list, shuffle = False)

# ========== Class weight ==========
# Label : ESD(0), OK(1)

counts = np.bincount([l for _, l in train_list], minlength = 2)
N = counts.sum(); K = 2

class_weight = {i: float(N) / (K * counts[i]) for i in range(K)}

print("Class_Weight : ", class_weight)


# ========== Custom Layer 정의(Lambda 대체) ==========
class PreprocessingLayer(layers.Layer):
    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        # 각 Backbone에 맞는 전처리 함수를 미리 할당
        if backbone_name == 'VGG16':
            self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input
        elif backbone_name == 'ResNet50V2':
            self.preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
        elif backbone_name == 'EfficientNetV2S':
            # EfficientNetV2S는 [0, 1] 스케일을 그대로 사용하므로 함수가 다름
            self.preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
        else:
            raise ValueError("Unsupported backbone name")

    def call(self, inputs):
        # EfficientNetV2S를 제외하고는 [0, 255] 스케일로 변환 후 전처리
        if self.backbone_name in ['VGG16', 'ResNet50V2']:
            scaled_inputs = inputs * 255.0
            return self.preprocess_fn(scaled_inputs)
        else: # EfficientNetV2S
            return self.preprocess_fn(inputs)

    # get_config는 모델 저장/로드를 위해 필수
    def get_config(self):
        config = super().get_config()
        config.update({"backbone_name": self.backbone_name})
        return config

# ========== Channel Mapper (N -> 3), Backbone ==========
sample_x, _ = next(iter(train_ds.take(1))) # train_ds의 원소는 (20, 256, 256, 5)
in_ch = int(sample_x.shape[-1]) # input_channel : 5
print("Input channels : ", in_ch)

def build_model(backbone_name="VGG16"):
    inp = layers.Input(shape=(img_size[0], img_size[1], in_ch), name="multi_channel_input")

    # Channel Mapper: N channels -> 3 channels
    x = layers.Conv2D(16, 1, padding="same", activation="relu", name="ch_mapper_16")(inp)
    x = layers.Conv2D(3, 1, padding="same", activation=None, name="ch_mapper_3")(x) # (256, 256, 3), [0, 1] 범위
    preproc_layer = PreprocessingLayer(backbone_name, name = f"{backbone_name.lower()}_preproc")

    # Backbone별 전용 전처리 레이어를 모델 내부에 적용
    # 이로써 모델 외부 데이터는 항상 [0, 1]로 유지 가능
    if backbone_name == "VGG16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    elif backbone_name == "ResNet50V2":
        base = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    elif backbone_name == "EfficientNetV2S": # EfficientNetV2는 [0, 255] 스케일링이 필요 없음
        base = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
        
    x = preproc_layer(x)
    y = base(x)

    # Custom Head
    h = layers.GlobalAveragePooling2D()(y)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="sigmoid")(h)

    model = models.Model(inputs=inp, outputs=out)
    return model, base

# 사용할 Backbone 선택: VGG16, ResNet50V2, EfficientNetV2S
BACKBONE = "VGG16"
model, base_model = build_model(BACKBONE)
# model.summary() -> Summary 확인 필요 시 주석 해제
# Input : (256, 256, 5)
# Channel Mapper : 5 -> 3
# Backbone : (256, 256, 3) -> feature maps
# Custom Head : 0 ~ 1 classification


# ========== Stage 1: Fine-tuning Head ==========
# Backbone은 고정하고 Channel Mapper와 Head만 학습
base_model.trainable = False

model.compile(
    optimizer=Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

work_dir = data_root
ckpt_path = os.path.join(work_dir, f"Model_{BACKBONE}_best.keras")

callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
]

print("\n--- Stage 1: Training Head ---")
history1 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=callbacks
)
