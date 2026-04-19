import os
import shutil
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# [구조 정의] 현장 데이터 및 환경에 맞게 수정 필요
BASE_PATH = "/data_home/user/2025/username/Python"
IMAGE_FOLDER = os.path.join(BASE_PATH, "EMG_IMAGE")
BATCH_SIZE = 16

# 1. 설비별 개별 설정 (모델 경로 및 파라미터 분리)
CONFIG = {
    "A": {
        "MODEL_PATH": os.path.join(BASE_PATH, "Model_A.keras"),
        "IMG_SIZE": (256, 256),
        "THRESHOLD": 0.01,
        "RESIDUAL_THRESH": -6.0,
        # 추가적인 설비 A 전용 파라미터 (Crop, Template 등)
    },
    "B": {
        "MODEL_PATH": os.path.join(BASE_PATH, "Model_B.keras"),
        "IMG_SIZE": (256, 256),
        "THRESHOLD": 0.01,
        "RESIDUAL_THRESH": -6.0,
        # 추가적인 설비 B 전용 파라미터
    }
}

def parse_machine_id(filename):
    """
    파일명에서 설비 정보(A 또는 B)를 추출하는 로직
    {LOT}_{GLASS}...{MACHINE}...{SEQ}.jpg
    """
    try:
        parts = filename.split("_")
        # 예: 특정 인덱스에 MACHINE 정보가 있는 경우
        # machine_id = parts[2] 
        # 임시 로직: 파일명에 A가 포함되면 A, 아니면 B로 반환 (현장 규칙에 맞게 수정)
        if "A" in filename: return "A"
        if "B" in filename: return "B"
        return None
    except Exception:
        return None

def preprocess_for_inference(file_path, config):
    """
    5-Channel 변환 등 설비별 config를 반영한 전처리 로직
    (기본 3채널 로드 예시)
    """
    img = cv2.imread(file_path)
    img = cv2.resize(img, config["IMG_SIZE"])
    img = img / 255.0
    # 여기서 Sobel, Darkness 채널 추가 로직 수행 가능
    return img

def main():
    # [Step 1] 다중 모델 로드
    print("Loading Models for Machine A and B...")
    models = {}
    for m_id in CONFIG.keys():
        if os.path.exists(CONFIG[m_id]["MODEL_PATH"]):
            models[m_id] = load_model(CONFIG[m_id]["MODEL_PATH"])
        else:
            print(f"[Warning] Model file not found for Machine {m_id}")

    # [Step 2] 파일 리스트 사전 분리
    all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files_by_machine = {m_id: [] for m_id in models.keys()}

    for f in all_files:
        m_id = parse_machine_id(f)
        if m_id in files_by_machine:
            files_by_machine[m_id].append(f)

    results_list = []

    # [Step 3] 설비별 독립적 Batch 추론 루프
    for m_id, file_list in files_by_machine.items():
        if not file_list:
            continue
            
        print(f"\n[Machine {m_id}] Starting Inference: {len(file_list)} images")
        current_model = models[m_id]
        current_config = CONFIG[m_id]

        # 설비별 리스트를 Batch 단위로 순회
        for i in tqdm(range(0, len(file_list), BATCH_SIZE)):
            batch_filenames = file_list[i : i + BATCH_SIZE]
            batch_tensors = []
            
            # Batch 구성
            for filename in batch_filenames:
                file_path = os.path.join(IMAGE_FOLDER, filename)
                try:
                    tensor = preprocess_for_inference(file_path, current_config)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Preprocess Error: {filename} - {e}")

            if not batch_tensors: continue
            
            input_batch = np.array(batch_tensors)
            
            # 추론 실행
            preds = current_model.predict(input_batch, verbose=0)

            # 결과 처리 및 데이터 누적
            for filename, pred in zip(batch_filenames, preds):
                score = float(pred[0])
                
                # Rule-based 보조 판단 (Placeholder)
                # residual = get_residual_score(filename) 
                
                # 최종 판정 로직
                is_ok = score >= current_config["THRESHOLD"]
                result_label = "OK" if is_ok else "ESD"

                results_list.append({
                    "Filename": filename,
                    "Machine": m_id,
                    "Score": round(score, 4),
                    "Result": result_label
                })

                # 파일 이동 (필요 시)
                dest_dir = os.path.join(BASE_PATH, result_label)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(os.path.join(IMAGE_FOLDER, filename), os.path.join(dest_dir, filename))

    # [Step 4] 결과 저장
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_excel(os.path.join(BASE_PATH, "Inference_Results.xlsx"), index=False)
        print("\n[Complete] All processes finished. Results saved.")

if __name__ == "__main__":
    main()
