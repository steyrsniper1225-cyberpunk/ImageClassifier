import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm # [수정] tqdm 라이브러리 import

# --- 0. 경로 및 기본 파라미터 설정 ---
base_dir = r"C:\Users\LGPC\Desktop\ROI_Algo"
template_image_path = os.path.join(base_dir, "Origin", "Template_03_blur.bmp")

cropped_dir_path = os.path.join(base_dir, "cropped_images")
os.makedirs(cropped_dir_path, exist_ok=True)
print(f"Crop된 이미지 저장 폴더: {cropped_dir_path}")

tpl_h, tpl_w = 256, 256
tpl_h_half, tpl_w_half = tpl_h // 2, tpl_w // 2

def find_best_rotated_template_match(img_input_gray, tpl_base_gray):
    # (이 함수는 수정할 필요 없음)
    tpl_0 = tpl_base_gray
    tpl_90 = cv2.rotate(tpl_0, cv2.ROTATE_90_CLOCKWISE)
    tpl_180 = cv2.rotate(tpl_0, cv2.ROTATE_180)
    tpl_270 = cv2.rotate(tpl_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    templates = [(tpl_0, 0), (tpl_90, 90), (tpl_180, 180), (tpl_270, 270)]
    best_score = -1.0
    best_top_left = None
    best_angle = None

    # [수정] tqdm 진행률 표시줄이 깨지지 않도록 내부 print문 주석 처리
    # print(" 템플릿 매칭 시작 (4개 각도 비교 중)...")

    for tpl, angle in templates:
        # (중요) tpl(256x256)이 img_input_gray(탐색영역)보다 크면 에러 발생
        if tpl.shape[0] > img_input_gray.shape[0] or tpl.shape[1] > img_input_gray.shape[1]:
            # [수정] tqdm 진행률 표시줄이 깨지지 않도록 내부 print문 주석 처리
            # print(f" ... {angle}도: 템플릿(w:{tpl.shape[1]})이 탐색영역(w:{img_input_gray.shape[1]})보다 커서 스킵")
            continue
            
        res = cv2.matchTemplate(img_input_gray, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # [수정] tqdm 진행률 표시줄이 깨지지 않도록 내부 print문 주석 처리
        # print(f" ... {angle}도 매칭 점수: {max_val:.4f} at {max_loc}")
        if max_val > best_score:
            best_score = max_val
            best_top_left = max_loc
            best_angle = angle
            
    return best_score, best_top_left, best_angle


# --- 1. 템플릿 이미지 로드 (스크립트 실행 시 한 번만) ---
try:
    tpl_base_gray = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    if tpl_base_gray is None:
        raise FileNotFoundError(f"템플릿 이미지를 로드할 수 없습니다. 경로 확인:\n{template_image_path}")
    if tpl_base_gray.shape[0] != tpl_h or tpl_base_gray.shape[1] != tpl_w:
        print(f"경고: 템플릿 크기가 (256, 256)이 아닙니다. 현재 크기: {tpl_base_gray.shape}")
    print(f"템플릿 이미지 로드 완료: {template_image_path}")
except Exception as e:
    print(f"템플릿 로드 중 치명적 오류 발생: {e}")
    exit()

results_list = []
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

try:
    all_files = os.listdir(base_dir)
except FileNotFoundError:
    print(f"오류: base_dir를 찾을 수 없습니다: {base_dir}")
    exit()

print(f"\n--- {base_dir} 경로에서 이미지 스캔 시작 ---")

# --- [수정] 2-1. tqdm을 사용하기 위해 실제 처리할 이미지 파일 목록을 먼저 생성 ---
image_files_to_process = []
for filename in all_files:
    input_image_path = os.path.join(base_dir, filename)
    if os.path.isfile(input_image_path) and filename.lower().endswith(valid_exts):
        image_files_to_process.append(filename)

print(f"총 {len(image_files_to_process)}개의 유효한 이미지 파일을 찾았습니다.")


# --- [수정] 2-2. 필터링된 이미지 목록을 tqdm으로 감싸서 처리 ---
for filename in tqdm(image_files_to_process, desc="이미지 Crop 진행 중"):
    input_image_path = os.path.join(base_dir, filename)

    # [수정] 파일 유효성 검사 코드는 이미 위에서 처리했으므로 루프 내에서 제거됨
    
    # [수정] tqdm 진행률 표시줄이 깨지지 않도록 상세 로그 주석 처리
    # print(f"\n=======================================================")
    # print(f"파일 처리 중: {filename}")
    # print(f"=======================================================")

    # ... (for 루프 시작) ...

    # 3. 입력 이미지 로드 (PIL로 EXIF 처리 후 OpenCV로 변환)
    try:
        # 3-1. PIL로 이미지 열기
        pil_img = Image.open(input_image_path)
        
        # 3-2. (요청사항) EXIF 데이터를 기준으로 이미지 자동 회전
        pil_img_transposed = ImageOps.exif_transpose(pil_img)
        
        # 3-3. PIL(RGB) 이미지를 OpenCV(BGR) numpy 배열로 변환
        # (Crop용 컬러 이미지)
        img_input_color = cv2.cvtColor(np.array(pil_img_transposed), cv2.COLOR_RGB2BGR)
        
        # 3-4. 매칭용 그레이스케일 이미지 생성
        img_input_gray = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)

    except Exception as e:
        # 오류 로그는 tqdm 진행률 표시줄을 깨뜨릴 수 있으나,
        # 어떤 파일에서 오류가 났는지 확인해야 하므로 유지합니다.
        # tqdm.write()를 사용하면 더 깔끔하게 오류를 출력할 수 있습니다.
        tqdm.write(f" [오류] {filename} 처리 중 오류: {e}. 건너뜁니다.")
        continue

    # 4. (추가) 탐색 영역(Search Region) 정의
    # 힌트와 Shift, 템플릿 크기를 모두 고려하여 탐색 영역을 계산합니다.
    HINT_X_MIN, HINT_X_MAX = 20, 1224
    HINT_Y_MIN, HINT_Y_MAX = 20, 1224
    MAX_SHIFT = 20 # 최대 20픽셀 Shift
    
    # X축 (가로): 힌트 최소값(20) - 최대 Shift(20) = 0
    x_min = HINT_X_MIN - MAX_SHIFT
    # X축 (가로): 힌트 최대값(1224) + 템플릿 너비(256) + 최대 Shift(20) = 1500
    x_max = HINT_X_MAX + tpl_w + MAX_SHIFT
    
    # Y축 (세로): 힌트 최소값(20) - 최대 Shift(20) = 0
    y_min = HINT_Y_MIN - MAX_SHIFT
    # Y축 (세로): 힌트 최대값(1224) + 템플릿 높이(256) + 최대 Shift(20) = 1500
    y_max = HINT_Y_MAX + tpl_h + MAX_SHIFT
    
    # (안전장치) 이미지 크기(1500x1500)를 벗어나지 않도록 좌표 보정
    h_img, w_img = img_input_gray.shape
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(h_img, y_max)
    x_max = min(w_img, x_max)

    # [수정] tqdm 진행률 표시줄이 깨지지 않도록 상세 로그 주석 처리
    # print(f"  전체 이미지 크기: (h:{h_img}, w:{w_img})")
    # print(f"  탐색 영역 (y, x): ({y_min}:{y_max}), ({x_min}:{x_max})")
    
    # 4-1. (수정) 탐색 영역으로 그레이스케일 이미지 자르기
    search_region_gray = img_input_gray[y_min:y_max, x_min:x_max]
    
    # 4-2. (수정) 잘라낸 이미지(search_region_gray)로 템플릿 매칭 실행
    best_score, best_top_left_relative, best_angle = find_best_rotated_template_match(
        search_region_gray, 
        tpl_base_gray
    )

    if best_top_left_relative is None:
        tqdm.write(f" [경고] {filename}에서 매칭에 실패했습니다. (탐색 영역이 템플릿보다 작을 수 있음)")
        continue
        
    # 4-3. (추가) 반환된 '상대 좌표'를 원본 이미지의 '절대 좌표'로 변환
    # (상대 x + 탐색영역 시작 x), (상대 y + 탐색영역 시작 y)
    best_top_left = (
        best_top_left_relative[0] + x_min,
        best_top_left_relative[1] + y_min
    )
    
    # --- ▲▲▲ 여기까지 수정/추가 ▲▲▲ ---


    # 5. 최종 ROI 좌표 계산 (이 부분은 기존 코드와 동일)
    ROI_X_center = best_top_left[0] + tpl_w_half
    ROI_Y_center = best_top_left[1] + tpl_h_half

    # [수정] tqdm 진행률 표시줄이 깨지지 않도록 상세 로그 주석 처리
    # print("-" * 40)
    # print(f"  [{filename}] 최종 매칭 결과:")
    # print(f"    - 최적 각도: {best_angle} 도")
    # print(f"    - 최고 점수: {best_score:.4f}")
    # (수정) 상대 좌표가 아닌 변환된 '절대 좌표'를 출력
    # print(f"    - 좌상단 (x, y): {best_top_left}")
    # print(f"\n    [계산된 ROI 중심 (ROI_X, ROI_Y)]")
    # print(f"    - ROI_X: {ROI_X_center}")
    # print(f"    - ROI_Y: {ROI_Y_center}")
    # print("-" * 40)

    # (요청사항 1) DataFrame을 위한 결과 저장
    results_list.append({
        "Filename": filename,
        "Best_Angle": best_angle,
        "ROI_TopLeft_X": best_top_left[0],
        "ROI_TopLeft_Y": best_top_left[1],
        "ROI_Center_X": ROI_X_center,
        "ROI_Center_Y": ROI_Y_center,
        "Score": best_score
    })

    # (요청사항 2) Crop 및 회전 후 저장
    try:
        # 2-1. 산출된 ROI(좌상단 좌표)를 기준으로 Crop
        tl_x, tl_y = best_top_left
        br_x, br_y = tl_x + tpl_w, tl_y + tpl_h
        
        # (중요) BGR numpy 배열(img_input_color) 대신
        # 원본 PIL 이미지(pil_img_transposed)에서 Crop합니다.
        # PIL.Image.crop()은 (left, upper, right, lower) 튜플을 받습니다.
        box = (tl_x, tl_y, br_x, br_y)
        
        # pil_img_transposed는 3-2 단계에서 정의된 RGB PIL 이미지입니다.
        cropped_pil_img = pil_img_transposed.crop(box) 

        # 2-2. 회전각 조건에 따라 최종 이미지 결정 (PIL.Image.rotate 사용)
        if best_angle == 0:
            final_cropped_pil_img = cropped_pil_img
        elif best_angle == 90:
            # 90도(CW)로 매칭됨 -> 0도로 돌리기 위해 -90도(CCW) 회전
            # PIL.Image.rotate(90)은 CCW이므로 -90 또는 270을 사용
            final_cropped_pil_img = cropped_pil_img.rotate(90, expand=True) 
        elif best_angle == 180:
            final_cropped_pil_img = cropped_pil_img.rotate(180, expand=True)
        elif best_angle == 270:
            # 270도(CCW)로 매칭됨 -> 0도로 돌리기 위해 +90도(CW) 회전
            final_cropped_pil_img = cropped_pil_img.rotate(270, expand=True)

        # 2-3. 새 파일명으로 저장
        save_filename = os.path.splitext(filename)[0] + "_cropped.jpg"
        save_path = os.path.join(cropped_dir_path, save_filename)
        
        # (중요) cv2.imwrite 대신 PIL의 save 메서드 사용
        # 이렇게 저장하면 파일이 RGB 채널 순서를 유지합니다.
        final_cropped_pil_img.save(save_path, "JPEG")

        # [수정] tqdm 진행률 표시줄이 깨지지 않도록 상세 로그 주석 처리
        # print(f"  Crop된 이미지 저장 완료: {save_path}")

    except Exception as e:
        tqdm.write(f" [오류] {filename} Crop 또는 저장 중 문제 발생: {e}")


# --- 3. (요청사항 1) 최종 결과를 Excel 파일로 저장 ---
if results_list:
    try:
        print("\n--- Excel 파일 저장 시작 ---")
        df_results = pd.DataFrame(results_list)
        excel_save_path = os.path.join(base_dir, "roi_match_results.xlsx")
        df_results.to_excel(excel_save_path, index=False, sheet_name="ROI_Results")
        print(f"Excel 결과 저장 완료: {excel_save_path}")
    except Exception as e:
        print(f"[오류] Excel 저장 실패: {e}")
else:
    print("\n처리할 이미지가 없어 Excel 파일을 생성하지 않았습니다.")

print("\n--- 모든 이미지 파일 처리가 완료되었습니다. ---")
