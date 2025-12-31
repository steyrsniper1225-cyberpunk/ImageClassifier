import os
import shutil
import argparse
import pandas as pd
import warnings
import base64 
import io
import numpy as np
import cv2
from PIL import Image, ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type = str)
args = parser.parse_args()
BASE_PATH = args.base_path
IMAGE_FOLDER = os.path.join(BASE_PATH, "EMG_IMAGE")

CURRENT_FACTORY = "A"
MODEL_PATH = None
IMG_SIZE = None
CROP_PARAMS = None
RULE_PARAMS = None

THRESHOLD = 0.4 # Model Threshold
GLOBAL_TEMPLATE_CV2_GRAY = None
TPL_H, TPL_W = 0, 0

TEMPLATE_A_BASE64 = "none"
TEMPLATE_B_BASE64 = "none"

CONFIG = {
    "A": {
        "MODEL_NAME": "Model_A.keras",
        "IMG_SIZE": (256, 256),
        "TEMPLATE_BASE64": TEMPLATE_A_BASE64,
        "CROP_PARAMS": {"HINT_X_MIN": 20, "HINT_X_MAX": 1224, "HINT_Y_MIN": 20, "HINT_Y_MAX": 1224, "MAX_SHIFT": 20},
        "RULE_PARAMS": {"ROW_START": 65, "ROW_END": 70, "CHANNEL": "Green", "RESIDUAL_THRESH": -10.0}
    },
    "B": {
        "MODEL_NAME": "Model_B.keras",
        "IMG_SIZE": (256, 256),
        "TEMPLATE_BASE64": TEMPLATE_B_BASE64,
        "CROP_PARAMS": {"HINT_X_MIN": 20, "HINT_X_MAX": 1224, "HINT_Y_MIN": 20, "HINT_Y_MAX": 1224, "MAX_SHIFT": 20},
        "RULE_PARAMS": {"ROW_START": 65, "ROW_END": 70, "CHANNEL": "Green", "RESIDUAL_THRESH": -10.0}
    }
}

def per_image_std(x):
    return tf.image.per_image_standardization(x)

def normalize01(x):
    mn = tf.reduce_min(x)
    mx = tf.reduce_max(x)
    return (x - mn) / (mx - mn + 1e-6)

def sobel_mag(x01):
    x4 = tf.expand_dims(x01, axis=0)
    sob = tf.image.sobel_edges(x4)
    sob = tf.squeeze(sob, axis=0)
    gx = sob[..., 0]; gy = sob[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy)
    mag = tf.reduce_mean(mag, axis=-1, keepdims=True)
    return mag

def darkness(x01):
    gray = tf.image.rgb_to_grayscale(x01)
    return 1.0 - gray

def calculate_residual_score(img_pil, rule_config):
    try:
        img_arr = np.array(img_pil)
        ch_map = {"Red": 0, "Green": 1, "Blue": 2}
        ch_idx = ch_map.get(rule_config["CHANNEL"], 1)
        
        r_start = rule_config["ROW_START"]
        r_end = rule_config["ROW_END"]
        
        roi = img_arr[r_start:r_end, :, ch_idx]
        profile = np.mean(roi, axis=0)
        
        x = np.arange(len(profile))
        slope, intercept = np.polyfit(x, profile, 1)
        fitted_line = slope * x + intercept
        
        residuals = profile - fitted_line
        min_residual = np.min(residuals)
        return min_residual

    except Exception as e:
        tqdm.write(f"  [Warning] 잔차 계산 중 오류: {e}")
        return 0.0

def load_template_for_cv2(b64_string, target_size):
    try:
        binary_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(binary_data, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_color is None:
            raise ValueError("CV2가 이미지를 디코딩하지 못했습니다.")
            
        target_size_wh = (target_size[1], target_size[0]) 
        img_resized = cv2.resize(img_color, target_size_wh, interpolation=cv2.INTER_LANCZOS4)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        print(f"템플릿 로드 성공 (CV2 Grayscale, 크기: {img_gray.shape})")
        return img_gray
        
    except Exception as e:
        print(f"Base64 템플릿 로드 실패 (CV2용): {e}")
        return None

def find_best_rotated_template_match(img_input_gray, tpl_base_gray):
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

def preprocess_for_inference(path, tpl_base_gray, tpl_h, tpl_w, crop_config, rule_config):
    try:
        # === 1. Image Load & Crop ===
        # 1-1. PIL로 이미지 열기 (EXIF 자동 회전 적용)
        pil_img = Image.open(path)
        pil_img_transposed = ImageOps.exif_transpose(pil_img)
        
        # 1-2. PIL(RGB) -> OpenCV(BGR) Numpy 배열로 변환
        img_input_color = cv2.cvtColor(np.array(pil_img_transposed), cv2.COLOR_RGB2BGR)
        img_input_gray = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)

        # 1-3. 탐색 영역(Search Region) 정의 (Config 사용)
        h_img, w_img = img_input_gray.shape
        x_min = max(0, crop_config["HINT_X_MIN"] - crop_config["MAX_SHIFT"])
        x_max = min(w_img, crop_config["HINT_X_MAX"] + tpl_w + crop_config["MAX_SHIFT"])
        y_min = max(0, crop_config["HINT_Y_MIN"] - crop_config["MAX_SHIFT"])
        y_max = min(h_img, crop_config["HINT_Y_MAX"] + tpl_h + crop_config["MAX_SHIFT"])
        
        # 1-4. 탐색 영역으로 이미지 자르기
        search_region_gray = img_input_gray[y_min:y_max, x_min:x_max]
        
        # 1-5. 템플릿 매칭 실행
        best_score, best_top_left_relative, best_angle = find_best_rotated_template_match(
            search_region_gray,
            tpl_base_gray
        )

        if best_top_left_relative is None:
            tqdm.write(f"  [Warning] {os.path.basename(path)} 매칭 실패.")
            return None, None # Tuple 반환
        
        # 1-6. '절대 좌표'로 변환
        best_top_left = (
            best_top_left_relative[0] + x_min,
            best_top_left_relative[1] + y_min
        )
        
        # 1-7. (중요) 원본 PIL 이미지(pil_img_transposed)에서 Crop
        tl_x, tl_y = best_top_left
        br_x, br_y = tl_x + tpl_w, tl_y + tpl_h
        box = (tl_x, tl_y, br_x, br_y) 
        
        cropped_pil_img = pil_img_transposed.crop(box)

        # 1-8. 최적 각도에 따라 회전
        if best_angle == 90:
            final_cropped_pil_img = cropped_pil_img.rotate(90, expand=True) 
        elif best_angle == 180:
            final_cropped_pil_img = cropped_pil_img.rotate(180, expand=True)
        elif best_angle == 270:
            final_cropped_pil_img = cropped_pil_img.rotate(270, expand=True)
        else:
            final_cropped_pil_img = cropped_pil_img
        
        # === (★추가) Rule-based Score 계산 ===
        # 전처리 전에 원본(256x256 RGB) 상태에서 계산
        rule_score = calculate_residual_score(final_cropped_pil_img, rule_config)

        # === 2. Tensor Conversion ===
        img_array = np.array(final_cropped_pil_img)
        img_tensor = tf.convert_to_tensor(img_array)
        tensor_01_rgb = tf.image.convert_image_dtype(img_tensor, tf.float32)
        
        img_std = per_image_std(tensor_01_rgb)
        x01 = normalize01(img_std) 
        edge = sobel_mag(x01)
        dark = darkness(x01)
        
        final_tensor = tf.concat([x01, edge, dark], axis=-1)
        
        return final_tensor, rule_score

    except Exception as e:
        tqdm.write(f"  [Error] {os.path.basename(path)} 처리 중 오류: {e}")
        return None, None

def parse_filename(filename):
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('___')

        if len(parts) < 3:
            raise ValueError("___ 없음")
            
        part1 = parts[0]
        part2 = parts[1]
        part3 = parts[2]
        
        p1_split = part1.split("_")
        p2_split = part2.split("_")
        p3_split = pary3.split("_")
        
        if len(p1_split) < 5:
            raise ValueError(f"part1 : underbar _ is under 5, Current : {len(p1_split)}")
        
        if len(p2_split) < 2:
            raise ValueError(f"part2 : underbar _ is under 2, Current : {len(p2_split)}")
            
        if len(p3_split) < 1:
            raise ValueError(f"part3 : underbar _ is under 1, Current : {len(p3_split)}")
            
        metadata = {
            "Lot ID": p1_split[0],
            "Glass_ID: p1_split[1],
            "Panel_ID": p1_split[2],
            "MACHINE_ID": p1_split[3],
            "PROCESS_CODE": int(p1_split[4]),
            "DEF_PNT_X": float(p2_split[0]),
            "DEF_PNT_Y": float(p2_split[1]),
            "SEQ": int(p3_split[0])
        }
        
        return metadata

# --- 5. 메인 추론 스크립트 ---

def main():
    global MODEL_PATH, IMG_SIZE, CROP_PARAMS, RULE_PARAMS
    global GLOBAL_TEMPLATE_CV2_GRAY, TPL_H, TPL_W
    
    # 1. 설정 로드
    try:
        config = CONFIG[CURRENT_FACTORY]
        MODEL_PATH = os.path.join(BASE_PATH, config["MODEL_NAME"])
        IMG_SIZE = config["IMG_SIZE"]
        CROP_PARAMS = config["CROP_PARAMS"]
        RULE_PARAMS = config["RULE_PARAMS"] # (★추가)
        
        TPL_H, TPL_W = IMG_SIZE[0], IMG_SIZE[1]
        
        print(f"--- [ {CURRENT_FACTORY} ] 설정 로드 ---")
        print(f"Rule-based: Row {RULE_PARAMS['ROW_START']}~{RULE_PARAMS['ROW_END']}, "
              f"Ch {RULE_PARAMS['CHANNEL']}, Thresh {RULE_PARAMS['RESIDUAL_THRESH']}")

        GLOBAL_TEMPLATE_CV2_GRAY = load_template_for_cv2(config["TEMPLATE_BASE64"], IMG_SIZE)
        if GLOBAL_TEMPLATE_CV2_GRAY is None:
            return
    
    except Exception as e:
        print(f"설정 로드 오류: {e}")
        return
    
    # 2. 모델 로드
    print(f"모델 로딩: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 3. 이미지 파일 검색
    try:
        image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    except:
        return

    if not image_files:
        return

    print(f"총 {len(image_files)}개 처리 시작...")

    results_list = []
    processed_gls_ids = set()

    # 4. 추론 루프
    for filename in tqdm(image_files, desc="추론"):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            metadata = parse_filename(filename)
            if metadata is None: continue
            
            gls_id = metadata["GLS_ID"]
            OK_FOLDER = os.path.join(BASE_PATH, f"{gls_id}_OK")
            ESD_FOLDER = os.path.join(BASE_PATH, f"{gls_id}_ESD")
            
            if gls_id not in processed_gls_ids:
                os.makedirs(OK_FOLDER, exist_ok=True)
                os.makedirs(ESD_FOLDER, exist_ok=True)
                processed_gls_ids.add(gls_id)

            # (★수정) 전처리 호출 (결과: Tensor, Score)
            tensor, rule_score = preprocess_for_inference(
                file_path, GLOBAL_TEMPLATE_CV2_GRAY, TPL_H, TPL_W, CROP_PARAMS, RULE_PARAMS
            )
            
            if tensor is None: continue

            # (★수정) 모델 추론
            tensor_batch = tf.expand_dims(tensor, axis=0)
            pred = model.predict(tensor_batch, verbose=0)[0][0]
            pred = float(pred) # 0.0 ~ 1.0 (높을수록 OK라고 가정)

            # (★수정) 하이브리드 판정 로직
            # Logic: Rule Score가 임계값보다 낮으면(오목하면) 무조건 불량 처리 (누출 방지)
            #        그렇지 않으면 모델의 판단을 따름
            
            is_rule_bad = rule_score < RULE_PARAMS["RESIDUAL_THRESH"]
            is_model_ok = pred >= THRESHOLD
            
            if is_rule_bad:
                result_label = "ESD" # Rule-based 강제 불량 (누출 방지)
                decision_note = "Rule_Defect"
            else:
                if is_model_ok:
                    result_label = "OK"
                    decision_note = "Model_OK"
                else:
                    # 모델이 불량이라 했지만 Rule은 정상인 경우 -> 일단 모델 의견 존중
                    result_label = "ESD" 
                    decision_note = "Model_Defect"

            # 결과 저장
            row = {
                "FILENAME": filename,
                "RESULT": result_label,
                "MODEL_PROB": round(pred, 4),       # 모델 점수
                "RULE_SCORE": round(rule_score, 4), # 잔차 점수
                "NOTE": decision_note
            }
            row.update(metadata)
            results_list.append(row)

            # 파일 이동
            dest_folder = OK_FOLDER if result_label == "OK" else ESD_FOLDER
            shutil.move(file_path, os.path.join(dest_folder, filename))

        except Exception as e:
            tqdm.write(f"  [Error] {filename}: {e}")

    # 5. 결과 저장 (Excel)
    if results_list:
        df = pd.DataFrame(results_list)
        try:
            for gls_id, g_df in df.groupby("GLS_ID"):
                g_df.to_excel(os.path.join(BASE_PATH, f"{gls_id}.xlsx"), index=False)
            print("Excel 저장 완료.")
        except Exception as e:
            print(f"Excel 저장 실패: {e}")

if __name__ == "__main__":
    main()
