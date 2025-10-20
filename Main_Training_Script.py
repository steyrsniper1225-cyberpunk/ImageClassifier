import os , random, datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
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

ADD_SOBEL = True
ADD_CANNY = True

AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

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
def decode_and_orient(path):
    # tf.py_function은 Tensor를 전달하므로, numpy()로 일반 문자열로 변환
    path_str = path.numpy().decode("utf-8")
    with Image.open(path_str) as img:
        img = ImageOps.exif_transpose(img) # EXIF 정보를 읽고 이미지를 정방향으로 회전
        img_np = np.array(img) # 다시 Tensorflow에서 처리할 수 있게끔 NumPy 배열로 변환
    return img_np
    # np.array(height, width, 3) uint8 [0~255]

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

def apply_canny(image_tensor_np):
    img_uint8 = (image_tensor_np * 255).astype(np.uint8)
    gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    canny_edge = cv2.Canny(gray_uint8, 100, 200)
    canny_edge = canny_edge.astype(np.float32) / 255.0
    return np.expand_dims(canny_edge, axis = -1)
    # Tensor(256, 256, 1) float32 [0, 1]

def build_feature(path, label):
    img_tensor = tf.py_function(decode_and_orient, [path], tf.uint8) # Tensor(H, W, 3) uint8 [0~255]
    img_tensor.set_shape([None, None, 3]) # shape 재설정
    
    # uint8 [0~255]를 정규화된 float32 [0, 1]로 변환
    img = tf.image.convert_image_dtype(img_tensor, tf.float32) # Tensor(H, W, 3) float32 [0, 1]
    x01 = tf.image.resize(img, img_size, method = "bilinear") # resize
    x01.set_shape((img_size[0], img_size[1], 3)) # (256, 256, 3) reminder

    feats = [x01] # Tensor(256, 256, 3) float32 [0, 1]

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
    # Tensor(256, 256, 5) float32 [0, 1]

    return feat, tf.cast(label, tf.float32)
    # feat : Tensor(256, 256, 5) float32 [0, 1]
    # label : Scalar() float32 [0 or 1]

def make_ds(pairs, batch = BATCH, shuffle = True):
    paths = [p for p,_ in pairs] # (filepath, label)에서 filepath(...JPG)
    labels = [l for _,l in pairs] # (filepath, label)에서 label(0 or 1)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # 두 list를 from_tensor_slices가 tensor로 묶음. datatype : tf.data.Dataset
    # list를 tensor로 묶은 객체

    if shuffle:
        ds = ds.shuffle(len(pairs), seed = SEED, reshuffle_each_iteration = True)

    ds = ds.map(build_feature, num_parallel_calls = AUTOTUNE)
    # (feat : Tensor(256, 256, 5) float32 [0, 1], label : Scalar() float32 [0 or 1])

    def augmentation(x, y):
        # x : Tensor(256, 256, 5) float32 [0, 1] 전체에 Augmentation 적용. 단, train_ds만 적용함.
        x = tf.image.random_brightness(x, max_delta = 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        k = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
        x = tf.image.rot90(x, k = k)
        return x, y
        # (feat : Tensor(256, 256, 5) float32 [0, 1], label : Scalar() float32 [0 or 1])

    if shuffle:
        ds = ds.map(augmentation, num_parallel_calls = AUTOTUNE)
        # (feat : Tensor(256, 256, 5) float32 [0, 1], label : Scalar() float32 [0 or 1])

    ds = ds.batch(batch).prefetch(AUTOTUNE)
    # x(feat) : Tensor(16, 256, 256, 5) float32 [0, 1] -> train은 shuffle=True라서 aug도 진행
    # y(label) : Scalar(20, ) float32 [0 or 1]              val은 shuffle=False라서 aug를 skip

    return ds
    # (feat : Tensor(256, 256, 5) float32 [0, 1], label : Scalar() float32 [0 or 1])

def history_to_df(history):
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df

train_ds = make_ds(train_list, shuffle = True)
# (feat : Tensor(256, 256, 5) float32 [0, 1], label : Scalar() float32 [0 or 1])
val_ds = make_ds(val_list, shuffle = False)
# (feat : Tensor(256, 256, 5) float32 [0, 1], label : Scalar() float32 [0 or 1])

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
            self.preprocess_fn = tf.keras.applications.efficientnet_v2.preprocess_input
        else:
            raise ValueError("Unsupported backbone name")

    def call(self, inputs):
        scaled_inputs = inputs * 255.0
        return self.preprocess_fn(scaled_inputs)

    # get_config는 모델 저장/로드를 위해 필수
    def get_config(self):
        config = super().get_config()
        config.update({"backbone_name": self.backbone_name})
        return config

# ========== Channel Mapper (N -> 3), Backbone ==========
sample_x, _ = next(iter(train_ds.take(1)))
in_ch = int(sample_x.shape[-1])
print("Input channels : ", in_ch)

def build_model(backbone_name="VGG16"):
    inp = layers.Input(shape=(img_size[0], img_size[1], in_ch), name="multi_channel_input")

    # Channel Mapper: N channels -> 3 channels
    x = layers.Conv2D(16, 1, padding="same", activation="relu", name="ch_mapper_16")(inp)
    x = layers.Conv2D(3, 1, padding="same", activation=None, name="ch_mapper_3")(x)
    preproc_layer = PreprocessingLayer(backbone_name, name = f"{backbone_name.lower()}_preproc")

    # Backbone별 전용 전처리 레이어를 모델 내부에 적용
    if backbone_name == "VGG16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    elif backbone_name == "ResNet50V2":
        base = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    elif backbone_name == "EfficientNetV2S":
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

# ========== Stage 2: Fine-tuning Backbone ==========
print("\n--- Stage 2: Fine-tuning Backbone ---")

# Backbone 모델의 동결 해제
base_model.trainable = True

# (참고) ResNet/EfficientNet 등 BatchNormalization 레이어가 있는 모델은
# BN 레이어는 계속 동결하는 것이 안정적일 수 있습니다. (VGG16은 해당 없음)
if BACKBONE != "VGG16":
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

# 미세 조정을 위해 더 낮은 학습률로 모델을 다시 컴파일
model.compile(
    optimizer=Adam(1e-5), # Stage 1 (1e-3) 보다 훨씬 낮은 학습률 사용
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Stage 2용 콜백 (필요시 EarlyStopping patience 등을 조절할 수 있음)
callbacks_stage2 = [
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-7) # min_lr을 더 낮출 수 있음
]

# Stage 2 학습 실행
history2 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2, # 총 Epoch 횟수
    initial_epoch=EPOCHS_STAGE1,       # Stage 1이 끝난 시점부터 시작
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=callbacks_stage2
)


'''
# ========== Save History and Final Model ==========

# Stage 1과 Stage 2의 history DataFrame 결합
df_hist1 = history_to_df(history1)
df_hist2 = history_to_df(history2)
df_hist = pd.concat([df_hist1, df_hist2], ignore_index=True)

# 학습 로그(CSV) 저장
save_dir = data_root
os.makedirs(save_dir, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(save_dir, f"training_log_{BACKBONE}_{ts}.csv")

df_hist.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nSaved Training Log: {csv_path}")
df_hist.head()

# 최종 모델 저장
final_path = os.path.join(work_dir, f"Model_{BACKBONE}_final.keras")
model.save(final_path)
print(f"Final Model Saved: {final_path}")
'''
