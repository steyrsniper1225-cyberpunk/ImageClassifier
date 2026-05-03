import os , random, datetime
import pandas as pd
import numpy as np

os.environ["TF_CUDNN_USE_AUTOTUNE"] = '0'
os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit(False)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ========== Path & Para. ==========
data_root = "/data_home/user/2025/username/Python/TRAIN"
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
    # str -> C:\Users\LGPC\Desktop\ROI_Algo\OK
    # str -> C:\Users\LGPC\Desktop\ROI_Algo\ESD
    
    ok = [os.path.join(p_ok, f) for f in os.listdir(p_ok) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    esd = [os.path.join(p_esd, f) for f in os.listdir(p_esd) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    # list -> OK/ESD 폴더 안의 모든 이미지에 대해 절대경로로 list를 저장
    
    ok.sort()
    esd.sort()
    # 오름차순으로 정렬
    
    return [(p, 1) for p in ok] + [(p, 0) for p in esd]
    # list -> tuple(path, label)로 이루어진 list[tuple(), tuple() ... tuple()]를 저장
    # 전체 데이터에 대해 tuple(esd 이미지 젇대경로, 1), tuple(ok 이미지 절대경로, 0)

all_pairs = list_labeled_files(data_root) # data_root를 읽고 List(tuple, tuple ... tuple) 생성
random.Random(SEED).shuffle(all_pairs) # all_pairs(list)에 대해 shuffle 적용
'''
shuffle 완료된 all_pairs ->>
[('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (65)_cropped.jpg', 1),
 ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (18)_cropped.jpg', 0),
 ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (31)_cropped.jpg', 0),
 ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (48)_cropped.jpg', 1),
 ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (67)_cropped.jpg', 0),
 ...
 ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (15)_cropped.jpg', 1),
 ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (35)_cropped.jpg', 1)]
'''

def stratified_split(pairs, val_ratio = 0.2):
    by_label = {0: [], 1: []}
    # dict -> {0: [], 1: []} 비어 있는 dictionary를 by_label에 저장
    
    for p, l in pairs:
        by_label[l].append((p, l)) # lower case of Alphabet "L"
    # (list)all_pairs의 tuple(path, label)들을 모두 순회. label이 0이면 key : 0에 tuple()을 .append()
    # dict -> "key : 0"에는 (ESD이미지 절대 경로, 0),
    # dict -> "key : 1"에는 (OK이미지 절대 경로, 1) 형태로 by_label에 저장
    # all_pairs -> by_label : dict 형태로 tuple들을 정리하는데 0은 esd, 1은 ok로 sort
    
    train, val = [], []
    # list -> train, val 각각 빈 list 생성
    
    for l, bucket in by_label.items():
        n = len(bucket)
        nv = int(round(n * val_ratio))
        val.extend(bucket[:nv])
        train.extend(bucket[nv:])
    '''
    (dict)by_label ->>
    {0: [('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (18)_cropped.jpg', 0),
    ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (31)_cropped.jpg', 0),
    ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (67)_cropped.jpg', 0),
    ...
    
    1: [('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (65)_cropped.jpg', 1),
    ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (48)_cropped.jpg', 1),
    ('C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (2)_cropped.jpg', 1),
    ...
    
    '''
    # dict_items -> by_label.items()은 dict인 by_label의 모든 데이터를 나열함
    # int -> label은 0(esd tuple들에 대한 key), 또는 1(ok tuple들에 대한 key)
    # list -> bucket은 tuple(esd/ok 이미지 경로, 0 or 1)들의 list
    # int -> n은 bucket에 들어있는 tuple의 수 (이 예에서는 80)
    # int -> nv는 n에 split_ratio를 곱한 값 (이 예에서는 16)
    # list -> val에는 tuple(esd 이미지 경로, 0)와 tuple(ok 이미지 경로, 1)을 0.2(val_ratio) 비율만큼 저장
    # list -> train에는 tuple(esd 이미지 경로, 0)와 tuple(ok 이미지 경로, 1)을 0.8 비율만큼 저장
        
    random.Random(SEED).shuffle(train)
    random.Random(SEED).shuffle(val)
    # train과 val의 데이터(tuple)들을 shuffle
    
    return train, val
    # train, val 둘 다 tuple(ok/esd 이미지 경로, 0 or 1 label)을 val_ratio로 정의된 비율만큼 나눠 갖는 list

train_list, val_list = stratified_split(all_pairs, VAL_SPLIT)

print(f"Train : {len(train_list)} Val : {len(val_list)}")
print("Class_Indices : ", {cls_esd : 0, cls_ok : 1})

# ========== Preprocessing (TF Calculation) ==========
def decode_image(path):
    img = tf.io.read_file(path) # Tensor(scalar)
    img = tf.io.decode_jpeg(img, channels = 3) # Tensor(256, 256, 3) uint8 [0,255]
    '''
    ([[[255, 254, 224],
        [255, 254, 224],
        [255, 254, 224],
        ...,
    '''
    img = tf.image.convert_image_dtype(img, tf.float32) # Tensor(256, 256, 3) float32 [0,1]
    '''
    ([[[1.        , 0.9960785 , 0.87843144],
        [1.        , 0.9960785 , 0.87843144],
        [1.        , 0.9960785 , 0.87843144],
        ...,
    '''
    return img # Tensor(256, 256, 3) float32 [0,1]

def per_image_std(x):
    return tf.image.per_image_standardization(x) # Tensor(256, 256, 3) float32 [0,1]
    '''
    ([[[ 0.9758827,  0.9662763,  0.6780788],
        [ 0.9758827,  0.9662763,  0.6780788],
        [ 0.9758827,  0.9662763,  0.6780788],
        ...,
    '''
    
def normalize01(x):
    mn = tf.reduce_min(x) # Tensor(scalar) float32, value : -1.4545833
    mx = tf.reduce_max(x) # Tensor(scalar) float32, value : 0.9758827
    return (x - mn) / (mx - mn + 1e-6) # Tensor(256, 256, 3) float32 [0,1]
    '''
    ([[[0.9999996 , 0.99604714, 0.87747014],
        [0.9999996 , 0.99604714, 0.87747014],
        [0.9999996 , 0.99604714, 0.87747014],
        ...,
    '''

def sobel_mag(x01):
    x4 = tf.expand_dims(x01, axis = 0) # Tensor(1, 256, 256, 3) float32 [0,1]
    '''
    ([[[[0.9999996 , 0.99604714, 0.87747014],
         [0.9999996 , 0.99604714, 0.87747014],
         [0.9999996 , 0.99604714, 0.87747014],
         ...,
    '''
    
    sob = tf.image.sobel_edges(x4) # Tensor(1, 256, 256, 3, 2) float32 [0,1]
    '''
    ([[[[[-5.9604645e-08,  0.0000000e+00],
          [-1.1920929e-07,  0.0000000e+00],
          [-1.1920929e-07,  0.0000000e+00]],

         [[-5.9604645e-08,  0.0000000e+00],
          [-1.1920929e-07,  0.0000000e+00],
          [-1.1920929e-07,  0.0000000e+00]],

         [[-5.9604645e-08,  0.0000000e+00],
          [-1.1920929e-07,  0.0000000e+00],
          [-1.1920929e-07,  0.0000000e+00]],

         ...,
    '''
    
    sob = tf.squeeze(sob, axis = 0) # Tensor(256, 256, 3, 2) float32 [0,1]
    '''
    ([[[[-5.9604645e-08,  0.0000000e+00],
         [-1.1920929e-07,  0.0000000e+00],
         [-1.1920929e-07,  0.0000000e+00]],

        [[-5.9604645e-08,  0.0000000e+00],
         [-1.1920929e-07,  0.0000000e+00],
         [-1.1920929e-07,  0.0000000e+00]],

        [[-5.9604645e-08,  0.0000000e+00],
         [-1.1920929e-07,  0.0000000e+00],
         [-1.1920929e-07,  0.0000000e+00]],

        ...,
    '''
    
    gx = sob[..., 0] # Tensor(256, 256, 3) float32 [0,1]
    gy = sob[..., 1] # Tensor(256, 256, 3) float32 [0,1]
    '''
    gx = ([[[-5.9604645e-08, -1.1920929e-07, -1.1920929e-07],
        [-5.9604645e-08, -1.1920929e-07, -1.1920929e-07],
        [-5.9604645e-08, -1.1920929e-07, -1.1920929e-07],
        ...,
    '''
    '''
    gy = ([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        ...,
    '''
    
    mag = tf.sqrt(gx * gx + gy * gy) # Tensor(256, 256, 3) float32 [0,1]
    '''
    ([[[5.9604645e-08, 1.1920929e-07, 1.1920929e-07],
        [5.9604645e-08, 1.1920929e-07, 1.1920929e-07],
        [5.9604645e-08, 1.1920929e-07, 1.1920929e-07],
        ...,
    '''
    
    mag = tf.reduce_mean(mag, axis = -1, keepdims = True) # Tensor(256, 256, 1) float32 [0,1]
    '''
    ([[[9.934107e-08],
        [9.934107e-08],
        [9.934107e-08],
        ...,
    '''
    return mag # Tensor(256, 256, 1) float32 [0,1]

def darkness(x01):
    gray = tf.image.rgb_to_grayscale(x01) # Tensor(256, 256, 1) float32 [0,1]
    return 1.0 - gray
    '''
    ([[[0.01638883],
        [0.01638883],
        [0.01638883],
        ...,
    '''

def build_feature(path, label):
    img = decode_image(path) # Tensor(256, 256, 3) float32 [0,1]
    x = per_image_std(img) # Tensor(256, 256, 3) float32 [0,1]
    
    x01 = normalize01(x) # Tensor(256, 256, 3) float32 [0,1]

    feats = [x01] # list -> 각각의 원소들이 Tensor(256, 256, 3) float32 [0,1]인 list

    if feats:
        feats.append(sobel_mag(x01)) # Tensor(256, 256, 1) float32 [0,1]

    if feats:
        feats.append(darkness(x01)) # Tensor(256, 256, 1) float32 [0,1]

    feat = tf.concat(feats, axis = -1) # Tensor(256, 256, 5) float32 [0,1]
    '''
    ([[[9.9999958e-01, 9.9604714e-01, 8.7747014e-01, 9.9341072e-08,
         1.6388834e-02],
        [9.9999958e-01, 9.9604714e-01, 8.7747014e-01, 9.9341072e-08,
         1.6388834e-02],
        [9.9999958e-01, 9.9604714e-01, 8.7747014e-01, 9.9341072e-08,
         1.6388834e-02],
        ...,
    '''

    return feat, tf.cast(label, tf.float32)
    # Tensor(256, 256, 5) float32 [0,1]
    # Tensor(scalar) float32, value : 1.0

def make_ds(pairs, batch = BATCH, shuffle = True):
    paths = [p for p,_ in pairs]
    labels = [l for _,l in pairs]
    # list -> paths  : (list)all_pairs의 tuple(이미지 경로, 0 or 1)을 분해하여 (str)이미지 경로들만을 취한 list를 생성
    '''
    ['C:\\Users\\LGPC\\Desktop\\ROI_Algo\\ESD\\ESD (32)_cropped.jpg',
     'C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (36)_cropped.jpg',
     'C:\\Users\\LGPC\\Desktop\\ROI_Algo\\OK\\OK (14)_cropped.jpg',
     '''
    # list -> labels : (list)all_pairs의 tuple(이미지 경로, 0 or 1)을 분해하여 (int)분류 라벨인 0 or 1만을 취한 list를 생성
    '''
    [0,
     1,
     1,
     '''    
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # TensorSliceDataset -> ds : (list)paths, (list)labels를 묶은 개체
    # <_TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))>

    if shuffle:
        ds = ds.shuffle(len(pairs), seed = SEED, reshuffle_each_iteration = True)
        # train_list는 shuffle, val_lisd는 shuffle하지 않음

    ds = ds.map(build_feature, num_parallel_calls = AUTOTUNE)

    def aug(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta = 0.05)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        return x, y

    if shuffle:
        ds = ds.map(aug, num_parallel_calls = AUTOTUNE)

    ds = ds.batch(batch).prefetch(AUTOTUNE)

    return ds

def history_to_df(history):
    d = history.history
    df = pd.DataFrame(d)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df

# train_ds, val_ds는 데이터가 아니고 이미지 읽기로부터 Batch 생성까지의 처리를 어떻게 할지 기술한 "설계도"와 같은 tf.data.Dataset 개체
# model.fit()이 train_ds에 Batch를 요청하여 1개씩 받아서 학습 진행 (=1개 Step)
# train_ds에 대해 Step이 완료(=모든 Batch 사용)되면 val_ds에 Batch를 요청하여 성능 확인
# Epoch 1회 완료하면 model.fit()이 train_ds에 Batch를 재요청
# train_ds는 이미지 읽기로부터 Batch 생성을 새롭게 진행
# 따라서 이미지의 순서는 Epoch 1때와는 다르게 shuffle되며 Augmentation도 새롭게 적용됨(단, val_ds는 shuffle, augmentatino 미 진행)
# 정해진 Epochs만큼 학습을 반복

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


# ========== Callbacks ==========
work_dir = "/data_home/user/2025/username/Python/EMG_MODEL"
ckpt_stage1 = os.path.join(work_dir, f"Stage1.keras")
ckpt_stage2 = os.path.join(work_dir, f"Stage2.keras")

callbacks1 = [
    ModelCheckpoint(ckpt_stage1, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
]

callbacks2 = [
    ModelCheckpoint(ckpt_stage2, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
]

# ========== Channel Mapper (N -> 3), Backbone ==========
sample_x, _ = next(iter(train_ds.take(1)))
'''
([[[[ 1.0044246 ,  0.9937564 ,  0.8754952 , -0.04516168,
          -0.05333477],
         [ 1.0044246 ,  0.9937564 ,  0.8754952 , -0.04516168,
          -0.05333477],
         [ 1.0044246 ,  0.9937564 ,  0.8754952 , -0.04516168,
          -0.05333477],
         ...,
'''
in_ch = int(sample_x.shape[-1])
print("Input channels : ", in_ch)

inp = layers.Input(shape = (img_size[0], img_size[1], in_ch), name = "multi_input")

x = layers.Conv2D(16, 1, padding = "same", activation = "relu", name = "ch_mapper_16")(inp)
x = layers.Conv2D(3, 1, padding = "same", activation = None, name = "ch_mapper_3")(x)

base = ResNet50V2(weights = "imagenet", include_top = False, input_shape = (img_size[0], img_size[1], 3))
y = base(x)

    # Custom Head
h = layers.GlobalAveragePooling2D()(y)
h = layers.Dropout(0.3)(h)
h = layers.Dense(256, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(1e-4))(h)
h = layers.Dropout(0.3)(h)
out = layers.Dense(1, "sigmoid", dtype = "float32")(h)

model = models.Model(inputs = inp, outputs = out)

# ========== Stage 1: Fine-tuning Head ==========
# Backbone은 고정하고 Channel Mapper와 Head만 학습
for layer in base.layers:
    layer.trainable = False

model.get_layer("ch_mapper_16").trainable = True
model.get_layer("ch_mapper_3").trainable = True

model.compile(
    optimizer = Adam(1e-3),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

history1 = model.fit(
    train_ds,
    epochs = EPOCHS_STAGE1,
    validation_data = val_ds,
    class_weight = class_weight,
    callbacks = callbacks1
)

print("Training Complete")

# ========== Stage 2: Fine-tuning Backbone ==========
def unfreeze_top_resnet(base_model, prefix = "conv5_"):
    for layer in base_model.layers:
        if layer.name.startswith(prefix):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = False
            
unfreeze_top_resnet(base)

model.get_layer("ch_mapper_16").trainable = True
model.get_layer("ch_mapper_3").trainable = True

# 미세 조정을 위해 더 낮은 학습률로 모델을 다시 컴파일
model.compile(
    optimizer = Adam(1e-4),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

history2 = model.fit(
    train_ds,
    epochs = EPOCHS_STAGE2,
    validation_data = val_ds,
    class_weight = class_weight,
    callbacks = callbacks2
)

print("Training Complete")

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
