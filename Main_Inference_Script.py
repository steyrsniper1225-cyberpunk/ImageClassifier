import os
import shutil
import pandas as pd
import warnings
import base64 
import io
import numpy as np # (추가) CV2/PIL 처리를 위해 import
import cv2         # (추가) 템플릿 매칭/CV2 처리를 위해 import
from PIL import Image, ImageOps # (추가) EXIF 처리 및 Crop을 위해 import

# 1. TensorFlow 로그 레벨 설정 (가장 중요)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# --- 1. 설정 (★수정: Crop 파라미터 추가) ---
BASE_PATH = r"D:\\"
IMAGE_FOLDER = os.path.join(BASE_PATH, "EMG_IMAGE")
THRESHOLD = 0.4

# (1단계에서 생성한 Base64 문자열을 여기에 붙여넣으세요)
TEMPLATE_A_BASE64 = "none"
TEMPLATE_B_BASE64 = "none"
# (예시 문자열입니다. 실제 변환된 문자열로 대체해야 합니다.)

CONFIG = {
    "A": {
        "MODEL_NAME": "Test_Model_A.keras",
        "IMG_SIZE": (256, 256), # (tpl_h, tpl_w)와 동일
        "TEMPLATE_BASE64": TEMPLATE_A_BASE64,
        "CROP_PARAMS": {
            "HINT_X_MIN": 20,
            "HINT_X_MAX": 1224,
            "HINT_Y_MIN": 20,
            "HINT_Y_MAX": 1224,
            "MAX_SHIFT": 20
        }
    },
    "B": {
        "MODEL_NAME": "Test_Model_B.keras",
        "IMG_SIZE": (320, 320), # B공장은 템플릿/모델 크기가 다를 경우
        "TEMPLATE_BASE64": TEMPLATE_B_BASE64,
         "CROP_PARAMS": {
            "HINT_X_MIN": 30,
            "HINT_X_MAX": 1300,
            "HINT_Y_MIN": 30,
            "HINT_Y_MAX": 1300,
            "MAX_SHIFT": 25
        }
    }
}

# --- (★수정: 여기만 "A" 또는 "B"로 변경하여 사용) ---
CURRENT_FACTORY = "A"
# ---------------------------------------------

# --- 2. 전역 설정 변수 (main 함수에서 채워짐) ---
MODEL_PATH = None
IMG_SIZE = None
CROP_PARAMS = None
GLOBAL_TEMPLATE_CV2_GRAY = None # (★수정) CV2용 그레이스케일 템플릿
TPL_H, TPL_W = 0, 0
TPL_H_HALF, TPL_W_HALF = 0, 0

# --- 3. 학습 시 사용한 TensorFlow 전처리 함수 ---
# (Crop 이후에 텐서를 대상으로 실행될 함수들)

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

# --- 4. 헬퍼 함수 (★수정: Crop 로직 전체 내장) ---

def load_template_for_cv2(b64_string, target_size):
    """(★신규) Base64 문자열을 CV2용 Grayscale Numpy 배열로 로드합니다."""
    try:
        # Base64 문자열을 바이너리 데이터로 디코딩
        binary_data = base64.b64decode(b64_string)
        
        # 바이너리를 Numpy 배열로 변환
        nparr = np.frombuffer(binary_data, np.uint8)
        
        # Numpy 배열을 CV2 이미지로 디코딩 (컬러)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError("CV2가 이미지를 디코딩하지 못했습니다.")
            
        # (중요) PIL/CV2의 resize는 (width, height) 순서
        # TF의 resize는 (height, width) 순서이므로, (W, H)로 전달
        target_size_wh = (target_size[1], target_size[0]) 
        
        # 리사이즈 (OpenCV)
        img_resized = cv2.resize(img_color, target_size_wh, interpolation=cv2.INTER_LANCZOS4)
        
        # 그레이스케일로 변환
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        print(f"템플릿 로드 성공 (CV2 Grayscale, 크기: {img_gray.shape})")
        return img_gray
        
    except Exception as e:
        print(f"Base64 템플릿 로드 실패 (CV2용): {e}")
        return None

def find_best_rotated_template_match(img_input_gray, tpl_base_gray):
    """(★신규) 제공된 Crop 스크립트의 템플릿 매칭 함수 (그대로 복사)"""
    tpl_0 = tpl_base_gray
    tpl_90 = cv2.rotate(tpl_0, cv2.ROTATE_90_CLOCKWISE)
    tpl_180 = cv2.rotate(tpl_0, cv2.ROTATE_180)
    tpl_270 = cv2.rotate(tpl_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    templates = [(tpl_0, 0), (tpl_90, 90), (tpl_180, 180), (tpl_270, 270)]
    best_score = -1.0
    best_top_left = None
    best_angle = None

    # (로그 출력이 많으므로 tqdm.write로 변경)
    # tqdm.write("  템플릿 매칭 시작 (4개 각도 비교 중)...")

    for tpl, angle in templates:
        if tpl.shape[0] > img_input_gray.shape[0] or tpl.shape[1] > img_input_gray.shape[1]:
            # tqdm.write(f"    ... {angle}도: 템플릿이 탐색영역보다 커서 스킵")
            continue
            
        res = cv2.matchTemplate(img_input_gray, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # tqdm.write(f"    ... {angle}도 매칭 점수: {max_val:.4f}")
        if max_val > best_score:
            best_score = max_val
            best_top_left = max_loc
            best_angle = angle
            
    return best_score, best_top_left, best_angle

def preprocess_for_inference(path, tpl_base_gray, tpl_h, tpl_w, tpl_h_half, tpl_w_half, crop_config):
    """
    (★수정)
    1. PIL/CV2로 원본(1500x1500) 이미지 로드 및 Crop
    2. 결과(256x256)를 TF 텐서로 변환
    3. 후속 TF 전처리 (std, norm, sobel, dark) 적용
    """
    
    # === 1단계: PIL/CV2를 사용한 이미지 로드 및 Crop ===
    try:
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
            tqdm.write(f"  [Warning] {os.path.basename(path)} 매칭 실패. 건너뜁니다.")
            return None
            
        # 1-6. '절대 좌표'로 변환
        best_top_left = (
            best_top_left_relative[0] + x_min,
            best_top_left_relative[1] + y_min
        )
        
        # 1-7. (중요) 원본 PIL 이미지(pil_img_transposed)에서 Crop
        tl_x, tl_y = best_top_left
        br_x, br_y = tl_x + tpl_w, tl_y + tpl_h
        box = (tl_x, tl_y, br_x, br_y) # PIL.crop()용 (left, upper, right, lower)
        
        cropped_pil_img = pil_img_transposed.crop(box)

        # 1-8. 최적 각도에 따라 회전
        if best_angle == 0:
            final_cropped_pil_img = cropped_pil_img
        elif best_angle == 90:
            final_cropped_pil_img = cropped_pil_img.rotate(90, expand=True) 
        elif best_angle == 180:
            final_cropped_pil_img = cropped_pil_img.rotate(180, expand=True)
        else: # 270
            final_cropped_pil_img = cropped_pil_img.rotate(270, expand=True)
        
        # 'final_cropped_pil_img'가 (256, 256, 3) 크기의 PIL 이미지가 됨

    except Exception as e:
        tqdm.write(f"  [Error] {os.path.basename(path)} Crop 처리 중 오류: {e}")
        return None

    # === 2단계: TensorFlow 전처리 파이프라인 적용 ===
    try:
        # 2-1. PIL Image -> Numpy Array -> tf.Tensor로 변환
        img_array = np.array(final_cropped_pil_img)
        img_tensor = tf.convert_to_tensor(img_array)
        
        # 2-2. [0,1] float32로 변환
        tensor_01_rgb = tf.image.convert_image_dtype(img_tensor, tf.float32)
        
        # 2-3. (중요) 사용자 요청 파이프라인 적용
        # crop_image() [완료] -> per_image_std() -> normalize01()
        img_std = per_image_std(tensor_01_rgb)
        x01 = normalize01(img_std) # (256, 256, 3)
        
        # 2-4. sobel_mag() add -> darkness() add
        edge = sobel_mag(x01)
        # edge_norm = normalize01(edge) # (256, 256, 1)
        
        dark = darkness(x01) # (256, 256, 1)
        
        # 2-5. 채널 병합
        channels = [x01, edge, dark]
        final_tensor = tf.concat(channels, axis=-1)
        
        # 5채널 확인
        if final_tensor.shape[-1] != 5:
             raise ValueError(f"최종 채널이 5가 아님 (현재: {final_tensor.shape[-1]})")
             
        return final_tensor

    except Exception as e:
        tqdm.write(f"  [Error] {os.path.basename(path)} 텐서 변환/전처리 중 오류: {e}")
        return None


def parse_filename(filename):
    """파일명 파싱 (이전과 동일)"""
    # (내용 동일)
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('___')
        if len(parts) < 2: raise ValueError("___ 구분자 없음")
        part1, part2, part3 = parts[0], parts[1], parts[2]
        p1_split = part1.split('_'); p2_split = part2.split('_'); p3_split = part3.split('_')
        if len(p1_split) < 5: raise ValueError("Part 1 _ 부족")
        if len(p2_split) < 1: raise ValueError("Part 2 _ 부족")
        metadata = {
            "LOT_ID": p1_split[0], "GLS_ID": p1_split[1], "PNL_ID": p1_split[2],
            "EQUIPMENT_ID": p1_split[3], "PROCESS_CODE": int(p1_split[4]),
            "DEF_PNT_X": float(p2_split[0]), "DEF_PNT_Y": float(p2_split[1]),
            "SEQ": int(p3_split[0])
        }
        return metadata
    except Exception as e:
        # (parse_filename은 preprocess 이전에 호출되므로 tqdm.write 사용)
        tqdm.write(f"  [Warning] 파일명 파싱 실패: {filename}. 오류: {e}")
        return None

# --- 5. 메인 추론 스크립트 (★수정: 설정 로드) ---

def main():
    
    # (★수정) 전역 변수 설정
    global MODEL_PATH, IMG_SIZE, CROP_PARAMS
    global GLOBAL_TEMPLATE_CV2_GRAY, TPL_H, TPL_W, TPL_H_HALF, TPL_W_HALF
    
    try:
        config = CONFIG[CURRENT_FACTORY]
        MODEL_PATH = os.path.join(BASE_PATH, config["MODEL_NAME"])
        IMG_SIZE = config["IMG_SIZE"]
        CROP_PARAMS = config["CROP_PARAMS"]
        
        # 템플릿 크기 전역 변수 설정
        TPL_H, TPL_W = IMG_SIZE[0], IMG_SIZE[1]
        TPL_H_HALF, TPL_W_HALF = TPL_H // 2, TPL_W // 2
        
        print(f"--- [ {CURRENT_FACTORY} ] 공장 설정 로드 ---")
        print(f"모델: {MODEL_PATH}")
        print(f"이미지/템플릿 크기: {IMG_SIZE}")
        
        # (★수정) CV2용 그레이스케일 템플릿 로드
        GLOBAL_TEMPLATE_CV2_GRAY = load_template_for_cv2(config["TEMPLATE_BASE64"], IMG_SIZE)
        if GLOBAL_TEMPLATE_CV2_GRAY is None:
            print("CV2 템플릿 로드 실패. 스크립트를 종료합니다.")
            return
            
    except KeyError:
        print(f"'{CURRENT_FACTORY}'에 대한 설정을 CONFIG에서 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"설정 로드 중 오류: {e}")
        return
    
    print("-" * 30)
    
    print(f"모델 로딩 중: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("스크립트를 종료합니다.")
        return
    print("모델 로딩 완료.")

    # (이하 이미지 파일 검색 로직은 동일)
    try:
        image_files = sorted([
            f for f in os.listdir(IMAGE_FOLDER) 
            if f.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png'))
        ])
    except FileNotFoundError:
        print(f"이미지 폴더를 찾을 수 없습니다: {IMAGE_FOLDER}")
        return
    if not image_files:
        print(f"이미지 폴더에 파일이 없습니다: {IMAGE_FOLDER}")
        return

    print(f"총 {len(image_files)}개의 이미지 파일 처리 시작...")

    results_list = []
    processed_gls_ids = set() 

    for filename in tqdm(image_files, desc="추론 진행"):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        if not os.path.isfile(file_path):
            continue

        try:
            # 1. 파일명 파싱 (Crop 전에 수행)
            metadata = parse_filename(filename)
            if metadata is None:
                continue # parse_filename 내부에서 이미 경고 출력
            
            current_gls_id = metadata["GLS_ID"]

            # 2. 동적 폴더 경로 설정
            OK_FOLDER_DYNAMIC = os.path.join(BASE_PATH, f"{current_gls_id}_OK")
            ESD_FOLDER_DYNAMIC = os.path.join(BASE_PATH, f"{current_gls_id}_ESD")
            
            if current_gls_id not in processed_gls_ids:
                os.makedirs(OK_FOLDER_DYNAMIC, exist_ok=True)
                os.makedirs(ESD_FOLDER_DYNAMIC, exist_ok=True)
                processed_gls_ids.add(current_gls_id)
                tqdm.write(f"  (결과 폴더 확인/생성: {OK_FOLDER_DYNAMIC}, {ESD_FOLDER_DYNAMIC})")

            # 3. (★수정) 통합 전처리 (Crop + TF 변환 + TF 전처리)
            tensor = preprocess_for_inference(
                file_path, 
                GLOBAL_TEMPLATE_CV2_GRAY, 
                TPL_H, TPL_W, 
                TPL_H_HALF, TPL_W_HALF, 
                CROP_PARAMS
            )
            
            # Crop/전처리 실패 시
            if tensor is None:
                continue # preprocess_for_inference 내부에서 이미 경고 출력

            # 4. 모델 추론
            tensor_batch = tf.expand_dims(tensor, axis=0)
            pred = model.predict(tensor_batch, verbose=0)[0][0]
            pred = float(pred)

            # 5. 결과 분류
            result_label = "OK" if pred >= THRESHOLD else "ESD"

            # 6. 결과 저장 (배포판 PRED 제외)
            row = {
                "FILENAME": filename,
                "RESULT": result_label
            }
            row.update(metadata)
            results_list.append(row)

            # 7. 파일 이동
            if result_label == "OK":
                dest_path = os.path.join(OK_FOLDER_DYNAMIC, filename)
            else:
                dest_path = os.path.join(ESD_FOLDER_DYNAMIC, filename)
            
            shutil.move(file_path, dest_path)

        except Exception as e:
            tqdm.write(f"  [Error] 파일 {filename} 처리 중 메인 루프 오류: {e}")

    print("추론 및 파일 이동 완료.")

    # --- 6. DataFrame 생성 및 Excel 저장 (이전과 동일) ---
    if not results_list:
        print("처리된 이미지가 없어 Excel 파일을 생성하지 않습니다.")
        return

    result_df = pd.DataFrame(results_list)
    
    try:
        grouped = result_df.groupby("GLS_ID")
        if not grouped.groups:
            print("결과 데이터에 GLS_ID가 없습니다. Excel 저장 실패.")
            return

        print(f"\n총 {len(grouped)}개의 GLS_ID에 대해 Excel 파일 저장 시작...")
        
        for gls_id, group_df in grouped:
            output_excel_path = os.path.join(BASE_PATH, f"{gls_id}.xlsx")
            group_df.to_excel(output_excel_path, index=False, engine='openpyxl')
            print(f"  -> 결과가 성공적으로 저장되었습니다: {output_excel_path}")
            
    except Exception as e:
        print(f"Excel 파일 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
