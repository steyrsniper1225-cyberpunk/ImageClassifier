import os
import shutil
import pandas as pd
from pathlib import Path

# --- (신규) X, Y 좌표 포맷팅을 위한 헬퍼 함수 ---
def format_coord_str(coord_str):
    """
    Excel에서 읽은 문자열(예: '100' or '-40.123' or '193.227')을 
    float으로 변환 후, 항상 소수점 3자리(예: '100.000', '-40.123', '193.227')
    형식의 문자열로 반환합니다.
    """
    if pd.isna(coord_str):
        return "" # 빈 값 처리
    try:
        # 문자열을 float으로 변환
        f_val = float(coord_str)
        # f-string formatting을 사용하여 소수점 3자리로 강제
        return f"{f_val:.3f}"
    except (ValueError, TypeError):
        # float 변환이 불가능한 문자열(예: 'N/A')은 원본 반환
        return str(coord_str)
# ---------------------------------------------


# 1. 기본 경로 설정
base_path = Path("/data_home/user/2025/username/Python/emgimages/AP5_Reservoir/FileSort")
excel_path = base_path / "MoveList.xlsx"

print(f"작업 대상 경로: {base_path}")
print(f"Excel 파일 경로: {excel_path}")

# 2. DataFrame1 (FullFileList) 생성 (정렬 적용됨)
print("1. 전체 파일 목록 스캔 중...")
all_files = list(base_path.glob("GLASS*/*.JPG"))
all_files_filtered = [f for f in all_files if f.parent.name != "Moved"]

if not all_files_filtered:
    print("[경고] 'GLASSxxx' 폴더에서 JPG 파일을 찾을 수 없습니다. (Moved 폴더 제외)")
    df1 = pd.DataFrame(columns=['FullFileList', 'FullPath'])
else:
    df1 = pd.DataFrame({
        'FullFileList': [f.name for f in all_files_filtered],
        'FullPath': [f for f in all_files_filtered]
    })
    df1 = df1.sort_values(by='FullFileList', ascending=True).reset_index(drop=True)

print(f"DataFrame1 생성 완료 (정렬됨). 찾은 파일 수: {len(df1)}")


# 3. DataFrame2 (ToMoveFileList) 생성
print("\n2. 'MoveList.xlsx' 파일 읽는 중...")
if not excel_path.exists():
    print(f"[오류] Excel 파일을 찾을 수 없습니다: {excel_path}")
    # raise SystemExit("프로세스 중단됨.")
else:
    try:
        # 1. 모든 컬럼을 문자열(str)로 먼저 읽어옵니다 (기존과 동일)
        df2 = pd.read_excel(excel_path, sheet_name="Sheet1", dtype=str)
        
        print("DataFrame2 생성 완료. 'ToMoveFileList' 컬럼 생성 중 (좌표 형식 포맷팅 적용)...")
        
        # 4. 'ToMoveFileList' 컬럼 생성 (수정된 부분)
        # --- .apply(format_coord_str) ---
        # X와 Y 컬럼에 헬퍼 함수를 적용하여 소수점 3자리 문자열로 변환합니다.
        
        df2['ToMoveFileList'] = (
            df2['LOT'] + '_' + df2['GLS'] + '_' +
            df2['PNL'] + '_' + df2['EQP'] + '_' +
            df2['PROCESS'] + '___' +
            df2['X'].apply(format_coord_str) + '_' +  # <--- 수정
            df2['Y'].apply(format_coord_str) + '___' + # <--- 수정
            df2['Seq'] + '_' + df2['FileExt']
        )
        # ----------------------------------------
        
        print(f"이동 대상 파일 목록 수: {len(df2)}")

        # 5. 이동할 파일 목록 식별
        files_to_move_set = set(df2['ToMoveFileList'])
        
        files_to_process_df = df1[df1['FullFileList'].isin(files_to_move_set)].copy()
        
        total_to_move = len(files_to_process_df)
        print(f"\n3. 파일 이동 작업 시작... (총 {total_to_move} 개)")

        if total_to_move > 0:
            moved_count = 0
            # 6. 'Moved' 폴더 생성 및 7. 파일 이동
            for index, row in files_to_process_df.iterrows():
                src_path = row['FullPath']
                file_name = row['FullFileList']
                
                parent_dir = src_path.parent      
                moved_dir = parent_dir / "Moved"  
                dest_path = moved_dir / file_name 
                
                try:
                    moved_dir.mkdir(exist_ok=True)
                    shutil.move(src_path, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"  [오류] {file_name} 이동 실패: {e}")
                    
            print(f"\n[완료] 총 {total_to_move} 개의 대상 파일 중 {moved_count} 개를 'Moved' 폴더로 이동했습니다.")
        else:
            print("[완료] 이동할 파일이 없습니다.")

    except Exception as e:
        print(f"[오류] DataFrame2 생성 또는 파일 이동 중 오류 발생: {e}")