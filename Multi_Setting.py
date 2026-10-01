"""Training / Inference 공용 CONFIG, Legacy 5채널 전처리, Multi-ROI 모델.

이 파일을 두 Main_*_MultiROI.py와 같은 폴더에 두세요.
입력: float32 [B, 256, 256, 5]. 출력: float32 [B, 1], P(OK).
ROI crop/resize는 모델 안에서 실행되며 .keras에 그대로 저장됩니다.
"""
import json
import os
import re
from numbers import Integral

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2


# ==================== 공용 CONFIG: ROI는 여기만 수정 ====================
CONFIG = {
    "IMAGE_SIZE": (256, 256),  # (height, width). 이 baseline에서는 고정.
    # 아래 좌표는 실행 가능한 예시입니다. 실제 defect zone 좌표로 교체하세요.
    # 좌표 원점: 정렬 완료된 256x256 global ROI의 왼쪽 위 (0, 0).
    # x = 가로/열, y = 세로/행. crop = image[y:y+height, x:x+width].
    # 끝 좌표는 미포함. x+width <= 256, y+height <= 256 이어야 합니다.
    # 순서 = [global] + 아래 리스트 순서 = feature concat 순서.
    # 개수 변경: 항목 추가/삭제. 크기 변경: width/height 수정.
    # 예: {"name": "fine_extra", "x": 100, "y": 160, "width": 32, "height": 32}
    # name은 고유한 영문/숫자/밑줄. 좌표/크기/순서를 바꾸면 재학습하세요.
    "INTERNAL_ROIS": [
        {"name": "mid_1",  "x": 32,  "y": 32,  "width": 64, "height": 64},
        {"name": "mid_2",  "x": 160, "y": 32,  "width": 64, "height": 64},
        {"name": "mid_3",  "x": 32,  "y": 160, "width": 64, "height": 64},
        {"name": "mid_4",  "x": 160, "y": 160, "width": 64, "height": 64},
        {"name": "fine_1", "x": 48,  "y": 48,  "width": 32, "height": 32},
        {"name": "fine_2", "x": 176, "y": 48,  "width": 32, "height": 32},
        {"name": "fine_3", "x": 112, "y": 176, "width": 32, "height": 32},
    ],
    "RESIZE_INTERPOLATION": "bilinear",  # 5채널을 함께 256x256으로 확대
    "MLP_UNITS": 128,  # 2048 * view수 -> 128 -> 1. projection/attention 없음.
    "DROPOUT": 0.3,
    "L2": 1e-4,
}

PREPROCESS_VERSION = "legacy_std_minmax_rgb_sobel_inverse_gray_v1"


def validate_config(config=CONFIG):
    if tuple(config["IMAGE_SIZE"]) != (256, 256):
        raise ValueError("이 baseline의 IMAGE_SIZE는 (256, 256)이어야 합니다.")
    names = {"global"}
    if not config["INTERNAL_ROIS"]:
        raise ValueError("Multi-ROI baseline에는 internal ROI가 1개 이상 필요합니다.")
    for roi in config["INTERNAL_ROIS"]:
        name = roi["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"잘못된 ROI name: {name!r}")
        if name in names:
            raise ValueError(f"중복 ROI name: {name}")
        names.add(name)
        for key in ("x", "y", "width", "height"):
            if isinstance(roi[key], bool) or not isinstance(roi[key], Integral):
                raise ValueError(f"{name}.{key}는 정수여야 합니다.")
        x, y, w, h = (roi[k] for k in ("x", "y", "width", "height"))
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 256 or y + h > 256:
            raise ValueError(f"ROI 범위 오류: {roi}")


def view_order(config=CONFIG):
    return ["global"] + [roi["name"] for roi in config["INTERNAL_ROIS"]]


# Legacy 수식/연산 순서를 유지. Sobel은 RGB 채널별 magnitude의 평균이며
# 재정규화하지 않습니다(Sobel 범위를 [0,1]로 clip하지 않음).
def per_image_std(x):
    return tf.image.per_image_standardization(x)


def normalize01(x):
    mn = tf.reduce_min(x)
    mx = tf.reduce_max(x)
    return (x - mn) / (mx - mn + 1e-6)


def sobel_mag(x01):
    sob = tf.squeeze(tf.image.sobel_edges(tf.expand_dims(x01, 0)), 0)
    gx, gy = sob[..., 0], sob[..., 1]
    return tf.reduce_mean(tf.sqrt(gx * gx + gy * gy), axis=-1, keepdims=True)


def darkness(x01):
    return 1.0 - tf.image.rgb_to_grayscale(x01)


def build_legacy_feature(rgb):
    """정렬된 uint8 RGB 또는 float32 RGB [0,1] -> [256,256,5].

    Training 파일은 이미 template crop/회전 정렬된 이미지여야 합니다.
    잘못된 크기를 조용히 resize하지 않고 실패시켜 좌표 오류를 방지합니다.
    """
    rgb = tf.ensure_shape(tf.convert_to_tensor(rgb), (256, 256, 3))
    rgb01 = tf.image.convert_image_dtype(rgb, tf.float32)
    x01 = normalize01(per_image_std(rgb01))
    return tf.ensure_shape(tf.concat([x01, sobel_mag(x01), darkness(x01)], -1),
                           (256, 256, 5))


def decode_image(path):
    # JPEG는 Legacy decoder 그대로. 기존 파일 목록에 포함되던 PNG도 지원.
    encoded = tf.io.read_file(path)
    image = tf.cond(tf.image.is_jpeg(encoded),
                    lambda: tf.io.decode_jpeg(encoded, channels=3),
                    lambda: tf.io.decode_png(encoded, channels=3))
    return tf.ensure_shape(image, (256, 256, 3))


def build_feature(path, label):
    return build_legacy_feature(decode_image(path)), tf.cast(label, tf.float32)


def build_multi_roi_model(config=CONFIG, backbone_weights="imagenet"):
    """8개 view도 encoder 인스턴스/weight는 단 1개. custom layer 불필요.

    Legacy는 preprocess_input을 import만 했고 실제 호출하지 않았습니다.
    여기서도 mapper 뒤 ResNet preprocess_input/추가 정규화를 넣지 않습니다.
    backbone_weights=None이면 외부 weight 다운로드 없이 독립 학습 가능.
    """
    validate_config(config)
    enc_input = layers.Input((256, 256, 5), dtype="float32", name="encoder_input")
    x = layers.Conv2D(16, 1, padding="same", activation="relu",
                      name="ch_mapper_16")(enc_input)
    x = layers.Conv2D(3, 1, padding="same", activation=None,
                      name="ch_mapper_3")(x)
    base = ResNet50V2(weights=backbone_weights, include_top=False,
                      input_shape=(256, 256, 3))
    # Legacy stage 1/2 모두 BN 고정. 공유 view 호출에서도 통계를 바꾸지 않음.
    x = base(x, training=False)
    feature = layers.GlobalAveragePooling2D(name="gap")(x)
    encoder = models.Model(enc_input, feature, name="shared_encoder")

    inp = layers.Input((256, 256, 5), dtype="float32", name="multi_input")
    # global도 float32로 통과. crop/resize는 5채널 생성 후, mapper 전에 수행.
    global_view = layers.Activation("linear", dtype="float32", name="view_global")(inp)
    views = [global_view]
    for roi in config["INTERNAL_ROIS"]:
        x, y, w, h = (roi[k] for k in ("x", "y", "width", "height"))
        crop = layers.Cropping2D(((y, 256-y-h), (x, 256-x-w)),
                                 data_format="channels_last", dtype="float32",
                                 name=f"crop_{roi['name']}")(inp)
        views.append(layers.Resizing(256, 256,
                                     interpolation=config["RESIZE_INTERPOLATION"],
                                     dtype="float32", name=f"view_{roi['name']}")(crop))
    # 호출은 여러 번이지만 모든 view가 정확히 같은 mapper + ResNet + GAP 공유.
    features = [layers.Activation("linear", name=f"feature_{name}")(encoder(view))
                for name, view in zip(view_order(config), views)]
    h = layers.Concatenate(name="fixed_order_concat")(features)
    h = layers.Dropout(config["DROPOUT"], name="fusion_dropout_1")(h)
    h = layers.Dense(config["MLP_UNITS"], activation="relu", name="fusion_dense",
                     kernel_regularizer=tf.keras.regularizers.l2(config["L2"]))(h)
    h = layers.Dropout(config["DROPOUT"], name="fusion_dropout_2")(h)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="ok_probability")(h)
    return models.Model(inp, out, name="multi_roi_resnet50v2"), encoder, base


def set_training_stage(base, stage):
    """stage 1: backbone 고정 / stage 2: conv5_* 중 BN 이외만 해제.

    weights=None일 때 stage='scratch'로 전체 convolution을 처음부터 학습.
    trainable 설정 변경 뒤에는 반드시 model.compile()을 다시 호출합니다.
    """
    if stage not in (1, 2, "scratch"):
        raise ValueError(f"Unknown training stage: {stage}")
    base.trainable = True  # 상위 model=False 상태가 자식 unfreeze를 막지 않도록
    for layer in base.layers:
        layer.trainable = (stage == "scratch" or
                           (stage == 2 and layer.name.startswith("conv5_"))) and not isinstance(
                               layer, layers.BatchNormalization)


def transplant_legacy_weights(legacy_path, encoder, base):
    """선택: 원본 Main_Training.py의 전체 .keras 모델에서 encoder만 이식.

    사용: build_multi_roi_model(backbone_weights=None) 후 이 함수를 호출.
    Legacy 마지막 Dense/head는 복사하지 않으며 새 MLP는 새로 학습합니다.
    .weights.h5만 있다면 원본 architecture를 먼저 만들어 load_weights한 뒤
    전체 .keras로 저장하세요. by_name/skip_mismatch로 누락을 숨기지 않습니다.
    """
    legacy = models.load_model(legacy_path, compile=False)
    candidates = [layer for layer in legacy.layers if isinstance(layer, models.Model)
                  and any(child.name == "conv5_block3_out" for child in layer.layers)]
    if len(candidates) != 1:
        raise ValueError("Legacy에서 유일한 ResNet50V2 backbone을 찾지 못했습니다.")
    pairs = [(legacy.get_layer(name), encoder.get_layer(name))
             for name in ("ch_mapper_16", "ch_mapper_3")]
    pairs.append((candidates[0], base))
    # 먼저 전부 검증하고 나서 복사하여 부분 이식을 방지.
    for source, target in pairs:
        if ([tuple(w.shape) for w in source.weights] !=
                [tuple(w.shape) for w in target.weights]):
            raise ValueError(f"Weight shape 불일치: {source.name} -> {target.name}")
    for source, target in pairs:
        target.set_weights(source.get_weights())
    print("Legacy mapper 5->16->3 + ResNet50V2 이식 완료. Fusion head는 새 weight.")


def model_contract(config=CONFIG):
    validate_config(config)
    return json.loads(json.dumps({
        "format_version": 1, "preprocessing": PREPROCESS_VERSION,
        "input_shape": [None, 256, 256, 5], "output_shape": [None, 1],
        "class_indices": {"ESD": 0, "OK": 1}, "output_semantics": "P(OK)",
        "view_order": view_order(config), "config": config,
    }))


def save_contract(model_path, config=CONFIG):
    path = os.fspath(model_path) + ".roi.json"
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(model_contract(config), file, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def load_multi_roi_model(model_path, config=CONFIG):
    """좌표/순서/전처리 불일치는 추론/파일 이동 전에 실패시킵니다.

    .keras와 함께 생성된 .keras.roi.json을 항상 같이 복사하세요.
    코드 CONFIG만 수정해 기존 모델의 ROI를 바꿀 수는 없습니다.
    """
    with open(os.fspath(model_path) + ".roi.json", encoding="utf-8") as file:
        saved = json.load(file)
    if saved != model_contract(config):
        raise ValueError("저장된 모델과 공용 CONFIG/전처리가 다릅니다. 학습 당시 설정을 사용하세요.")
    model = models.load_model(model_path, compile=False)
    if (model.name != "multi_roi_resnet50v2" or
            tuple(model.input_shape) != (None, 256, 256, 5) or
            tuple(model.output_shape) != (None, 1)):
        raise ValueError("Multi-ROI model input/output/architecture 불일치")
    model.get_layer("shared_encoder").get_layer("ch_mapper_16")
    for roi in config["INTERNAL_ROIS"]:
        x, y, w, h = (roi[k] for k in ("x", "y", "width", "height"))
        crop = model.get_layer(f"crop_{roi['name']}")
        if tuple(map(tuple, crop.cropping)) != ((y, 256-y-h), (x, 256-x-w)):
            raise ValueError(f"저장된 crop 좌표 불일치: {roi['name']}")
        resize = model.get_layer(f"view_{roi['name']}")
        if (resize.height, resize.width, resize.interpolation) != (
                256, 256, config["RESIZE_INTERPOLATION"]):
            raise ValueError(f"저장된 resize 불일치: {roi['name']}")
    # 실제 serialized concat 입력 순서도 확인(Keras 2/3 양쪽 지원).
    def source_name(tensor):
        history = tensor._keras_history
        source = getattr(history, "operation", None)
        if source is None:
            source = history[0]
        return source.name
    actual = [source_name(t) for t in model.get_layer("fixed_order_concat").input]
    if actual != [f"feature_{name}" for name in view_order(config)]:
        raise ValueError("저장된 모델의 concat 순서 불일치")
    return model
