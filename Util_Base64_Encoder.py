import base64
import os

# --- 여기에 변환할 템플릿 파일 경로를 입력하세요 ---
# 예: r"D:\Template_A.bmp"
# 예: r"D:\Template_B.bmp"
TEMPLATE_FILE_PATH = r"D:\Template_A.bmp" 

# ------------------------------------------------

def encode_image_to_base64(filepath):
    """이미지 파일을 base64 문자열로 인코딩합니다."""
    if not os.path.exists(filepath):
        print(f"파일을 찾을 수 없습니다: {filepath}")
        return None
        
    try:
        with open(filepath, "rb") as image_file:
            # 바이너리 읽기 -> base64 인코딩 -> utf-8 문자열로 디코딩
            b64_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        print(f"--- [ {os.path.basename(filepath)} ] 변환 성공 ---")
        print("아래 문자열 전체를 복사하여 메인 스크립트의 CONFIG 사전에 붙여넣으세요.")
        print("-" * 70)
        print(b64_string)
        print("-" * 70)
        
    except Exception as e:
        print(f"파일 변환 중 오류 발생: {e}")

if __name__ == "__main__":
    encode_image_to_base64(TEMPLATE_FILE_PATH)