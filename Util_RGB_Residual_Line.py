import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 1. 설정
base_path = "/data_home/user/2025/username/Python"
save_excel_path = os.path.join(base_path, "residual_analysis_narrow.xlsx")

rule_config = {
    "CHANNEL": "Green",
    "ROW_START": 100,
    "ROW_END": 200
}

# 2. 잔차 계산 함수 (이전과 동일)
def calculate_residual_array(img_pil, rule_config):
    try:
        img_arr = np.array(img_pil)
        ch_map = {"Red": 0, "Green": 1, "Blue": 2}
        ch_idx = ch_map.get(rule_config["CHANNEL"], 1)
        
        r_start = rule_config["ROW_START"]
        r_end = rule_config["ROW_END"]
        
        if r_start >= img_arr.shape[0] or r_end > img_arr.shape[0]:
            return None
            
        roi = img_arr[r_start:r_end, :, ch_idx]
        profile = np.mean(roi, axis=0)
        
        x = np.arange(len(profile))
        slope, intercept = np.polyfit(x, profile, 1)
        fitted_line = slope * x + intercept
        
        residuals = profile - fitted_line
        return residuals

    except Exception:
        return None

# 3. 메인 로직 (Narrow Form 변환 적용)
def main():
    image_files = glob.glob(os.path.join(base_path, "*.jpg"))
    
    if not image_files:
        print(f"경로에 .jpg 파일이 없습니다: {base_path}")
        return

    data_buffer = []  # DataFrame들을 담을 리스트

    print(f"총 {len(image_files)}개의 이미지를 처리합니다...")
    
    for file_path in tqdm(image_files):
        filename = os.path.basename(file_path)
        
        try:
            with Image.open(file_path) as img:
                residuals = calculate_residual_array(img, rule_config)
                
                if residuals is not None:
                    # 해당 이미지의 데이터를 DataFrame으로 생성
                    # X_Index는 0부터 길이만큼 생성
                    temp_df = pd.DataFrame({
                        "FileName": filename,
                        "X_Index": np.arange(len(residuals)),
                        "Value": residuals
                    })
                    data_buffer.append(temp_df)
                    
        except Exception as e:
            print(f"이미지 로드 실패 ({filename}): {e}")

    # 4. Excel 저장
    if data_buffer:
        # 모든 데이터를 세로로 병합 (Concatenate)
        final_df = pd.concat(data_buffer, ignore_index=True)
        
        # 컬럼 순서 명시적 지정
        final_df = final_df[["FileName", "X_Index", "Value"]]
        
        # 저장
        final_df.to_excel(save_excel_path, index=False)
        print(f"\n[완료] 결과가 저장되었습니다: {save_excel_path}")
        print(f"데이터 형태: {final_df.shape}")
        print("컬럼 구성: FileName, X_Index, Value")
    else:
        print("\n[알림] 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()