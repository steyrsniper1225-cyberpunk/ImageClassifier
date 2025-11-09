import pandas as pd
import glob
import os

# --- 설정 ---

# 1. 엑셀 파일이 저장된 기본 경로
BASE_PATH = r"D:\\"

# 2. 검색할 파일 패턴 (예: GLS0001.xlsx, GLS0002.xlsx ...)
# "GLS*.xlsx"는 "GLS"로 시작하고 ".xlsx"로 끝나는 모든 파일을 의미합니다.
FILE_PATTERN = "GLS*.xlsx"

# 3. 저장할 최종 통합 파일 이름
OUTPUT_FILENAME = "EMG_Classification_Summary.xlsx"

# ----------------

def combine_gls_excels():
    """
    지정된 경로에서 GLS*.xlsx 파일을 찾아 하나의 파일로 병합합니다.
    """
    
    # 1. 검색할 파일 패턴 정의
    search_path = os.path.join(BASE_PATH, FILE_PATTERN)
    
    # 2. 패턴과 일치하는 모든 엑셀 파일 리스트 검색
    # glob은 파일 목록을 자동으로 오름차순 정렬하지 않을 수 있으므로, sorted()를 사용합니다.
    excel_files = sorted(glob.glob(search_path))
    
    if not excel_files:
        print(f"'{search_path}' 경로에서 패턴과 일치하는 파일을 찾지 못했습니다.")
        print("파일 경로나 FILE_PATTERN 설정을 확인하세요.")
        return

    print(f"총 {len(excel_files)}개의 엑셀 파일 병합을 시작합니다.")
    print("-" * 30)
    for f in excel_files:
        # 경로가 아닌 파일명만 깔끔하게 출력
        print(f"  -> {os.path.basename(f)}")
    print("-" * 30)

    # 3. 모든 엑셀 파일을 읽어 DataFrame 리스트에 저장
    all_dfs = []
    for file in excel_files:
        try:
            # os.path.basename(file)을 사용하여 현재 파일 이름 로깅
            print(f"  ... 읽는 중: {os.path.basename(file)}")
            df = pd.read_excel(file, engine='openpyxl')
            all_dfs.append(df)
        except Exception as e:
            print(f"[경고] 파일을 읽는 중 오류 발생: {os.path.basename(file)}. 오류: {e}")

    # 4. DataFrame 리스트가 비어있는지 확인 (모든 파일 읽기 실패 시)
    if not all_dfs:
        print("병합할 데이터가 없습니다. 모든 파일을 읽는데 실패했습니다.")
        return

    # 5. 모든 DataFrame을 하나로 병합 (위아래로 붙이기)
    # ignore_index=True: 각 파일의 원래 인덱스를 무시하고 0부터 새로 부여합니다.
    try:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    except ValueError as e:
        print(f"DataFrame 병합 실패: {e}")
        print("읽어들인 파일이 없거나 데이터 형식이 맞지 않을 수 있습니다.")
        return

    # 6. 최종 엑셀 파일로 저장
    output_path = os.path.join(BASE_PATH, OUTPUT_FILENAME)
    
    try:
        combined_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n✅ 성공: 모든 데이터가 {output_path} 파일에 저장되었습니다.")
        print(f"  (총 {len(excel_files)}개 파일, {len(combined_df)} 행)")
    except Exception as e:
        print(f"최종 엑셀 파일 저장 실패: {e}")
        print(f"파일이 열려있는지 확인하세요: {output_path}")

if __name__ == "__main__":
    combine_gls_excels()
