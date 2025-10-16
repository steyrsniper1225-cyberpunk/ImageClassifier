import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

def get_orientation_value(image_path):
    """
    이미지 파일 경로를 받아 'Orientation' 태그 값을 반환합니다.
    EXIF 정보나 Orientation 태그가 없으면 'N/A'를 반환합니다.
    """
    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()

            if exif_raw:
                # Orientation 태그의 숫자 ID는 274입니다.
                orientation_tag_id = 274
                return exif_raw.get(orientation_tag_id, 'Not Found') # 태그가 없으면 'Not Found'
            else:
                return 'No EXIF' # EXIF 데이터 자체가 없음
    except Exception:
        return 'Error' # 파일을 읽는 중 오류 발생

# --- 메인 코드 실행 부분 ---
if __name__ == "__main__":
    # EXIF 정보를 확인할 이미지들이 있는 폴더 경로를 지정하세요.
    folder_path = r"C:\Users\LGPC\Desktop\ESD" # 예시 경로

    if not os.path.isdir(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
    else:
        exif_data_list = []
        supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')

        print(f"Reading images from: {folder_path}")
        # 폴더 내의 모든 파일 순회
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_extensions):
                image_full_path = os.path.join(folder_path, filename)
                
                # (1) Orientation 값을 가져와 tuple 생성
                orientation = get_orientation_value(image_full_path)
                exif_data_list.append((filename, orientation))

        if not exif_data_list:
            print("해당 폴더에 분석할 이미지가 없습니다.")
        else:
            # (2) & (3) DataFrame 생성 및 컬럼명 설정
            df = pd.DataFrame(exif_data_list, columns=["FileName", "Orientation"])

            # (4) DataFrame을 .xlsx 파일로 저장
            # os.path.dirname()으로 폴더 경로를 얻고, 파일명을 지정합니다.
            output_excel_path = os.path.join(os.path.dirname(folder_path), "exif_orientation_results.xlsx")
            
            try:
                df.to_excel(output_excel_path, index=False, engine='openpyxl')
                print(f"\n EXIF 정보가 성공적으로 저장되었습니다.")
                print(f"   -> {output_excel_path}")
                print("\n--- DataFrame Preview ---")
                print(df.head())
                
            except Exception as e:
                print(f"\n Excel 파일을 저장하는 중 오류가 발생했습니다: {e}")
