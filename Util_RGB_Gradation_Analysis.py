import os
import glob
import cv2
import numpy as np
import pandas as pd

def extract_window_rgb_profile(image_dir_path, save_csv_path=None):
    """
    지정된 디렉토리의 .jpg 이미지를 읽어 특정 영역(Window)의 RGB 값을 추출합니다.
    Target Window: Y[60:74], X[0:255] (15 Rows)
    Output: DataFrame [FileName, RowID, Channel, Value]
    """
    
    data_frames = []
    
    # 1. 이미지 파일 리스트 검색
    image_files = sorted(glob.glob(os.path.join(image_dir_path, "*.jpg")))
    print(f"Found {len(image_files)} images in {image_dir_path}")

    for file_path in image_files:
        try:
            file_name = os.path.basename(file_path)
            
            # 2. 이미지 로드 (OpenCV 사용)
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            # BGR -> RGB 변환 (OpenCV는 기본적으로 BGR 로드)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. 특정 영역(Window) Crop
            # Y: 60~74 (15 rows) -> Slice: 60:75 (End index is exclusive)
            # X: 0~255 -> Slice: 0:256
            roi = img_rgb[60:75, 0:256] # Shape: (15, 256, 3)
            
            rows, cols, _ = roi.shape
            
            # 4. 데이터 구조화 (Vectorization for Speed)
            # Calc Abs Values
            r_vals = roi[:, :, 0].flatten()
            g_vals = roi[:, :, 1].flatten()
            b_vals = roi[:, :, 2].flatten()

            # Calc Norm Values (Row-wise Normalization)
            def normalize_rows(channel_data):
                mat = channel_data.astype(float)
                mins = mat.min(axis=1, keepdims=True)
                maxs = mat.max(axis=1, keepdims=True)
                rng = maxs - mins
                rng[rng == 0] = 1.0 # Avoid division by zero
                return ((mat - mins) / rng).flatten()

            r_norm = normalize_rows(roi[:, :, 0])
            g_norm = normalize_rows(roi[:, :, 1])
            b_norm = normalize_rows(roi[:, :, 2])
            
            # 메타데이터 생성
            # RowID: 60,60,..,60 (256개), 61,61,..,61 (256개) ... 
            row_ids = np.repeat(np.arange(60, 60 + rows), cols)
            # X_Index: 0,1,..,255, 0,1,..,255 ...
            col_ids = np.tile(np.arange(cols), rows)
            
            # 채널별 DataFrame 생성
            # 채널별 DataFrame 생성
            # X축 방향 순서(Index)는 DataFrame의 행 순서로 내재됨
            df_r = pd.DataFrame({'FileName': file_name, 'RowID': row_ids, 'X_Index': col_ids, 'Channel': 'R', 'Value_Abs': r_vals, 'Value_Norm': r_norm})
            df_g = pd.DataFrame({'FileName': file_name, 'RowID': row_ids, 'X_Index': col_ids, 'Channel': 'G', 'Value_Abs': g_vals, 'Value_Norm': g_norm})
            df_b = pd.DataFrame({'FileName': file_name, 'RowID': row_ids, 'X_Index': col_ids, 'Channel': 'B', 'Value_Abs': b_vals, 'Value_Norm': b_norm})
            
            # 병합 (R -> G -> B 순서로 적재)
            df_img = pd.concat([df_r, df_g, df_b], axis=0, ignore_index=True)
            
            # 최적화: FileName과 Channel을 category 타입으로 변환하여 메모리 절약 가능 (선택 사항)
            # df_img['FileName'] = df_img['FileName'].astype('category')
            # df_img['Channel'] = df_img['Channel'].astype('category')

            data_frames.append(df_img)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    # 5. 전체 데이터 병합 및 반환
    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)
        
        # 원하는 컬럼 순서 보장
        final_df = final_df[['FileName', 'RowID', 'X_Index', 'Channel', 'Value_Abs', 'Value_Norm']]
        
        if save_csv_path:
            final_df.to_csv(save_csv_path, index=False)
            print(f"Profile data saved to: {save_csv_path}")
            
        return final_df
    else:
        print("No data extracted.")
        return pd.DataFrame()

# ==========================================
# 실행 예시 (사용자 환경에 맞춰 경로 수정 필요)
# ==========================================
if __name__ == "__main__":
    # 분석할 이미지가 있는 디렉토리 경로
    target_dir = "/Users/petrenko/Documents/Program_study/Antigravity_Works/ImageClassifier" 
    
    # 결과 저장 경로
    output_path = os.path.join(target_dir, "rgb_analysis_result.csv")
    
    # 함수 실행
    df_result = extract_window_rgb_profile(target_dir, output_path)
    
    # 결과 확인
    print(df_result.head())
    print(f"Total Rows: {len(df_result)}")
