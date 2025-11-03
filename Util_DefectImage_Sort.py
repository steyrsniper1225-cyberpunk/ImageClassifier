# ===========================
# ========= Gemini ==========
# ===========================

import os
import shutil
import pandas as pd
from pathlib import Path

# 1. 기본 경로 설정
base_path = Path("/data_home/user/2025/username/Python/emgimages/AP5_Reservoir/FileSort")
excel_path = base_path / "MoveList.xlsx"

print(f"작업 대상 경로: {base_path}")
print(f"Excel 파일 경로: {excel_path}")

# 2. DataFrame1 (FullFileList) 생성
print("1. 전체 파일 목록 스캔 중...")
all_files = list(base_path.glob("GLASS*/*.JPG"))

# 'Moved' 폴더 내 파일은 제외
all_files_filtered = [f for f in all_files if f.parent.name != "Moved"]

if not all_files_filtered:
    print("[경고] 'GLASSxxx' 폴더에서 JPG 파일을 찾을 수 없습니다. (Moved 폴더 제외)")
    # 스크립트를 중단하지 않고 빈 DataFrame으로 계속 진행 (이후 단계에서 어차피 이동할 파일 없음)
    
df1 = pd.DataFrame({
    'FullFileList': [f.name for f in all_files_filtered],
    'FullPath': [f for f in all_files_filtered]
})
print(f"DataFrame1 생성 완료. 찾은 파일 수: {len(df1)}")


# 3. DataFrame2 (ToMoveFileList) 생성
print("\n2. 'MoveList.xlsx' 파일 읽는 중...")
if not excel_path.exists():
    print(f"[오류] Excel 파일을 찾을 수 없습니다: {excel_path}")
    # 여기서 스크립트를 중단하는 것이 적절함
    # raise SystemExit("프로세스 중단됨.")
else:
    try:
        # 모든 컬럼을 문자열(str)로 읽어와 형식 문제를 방지
        df2 = pd.read_excel(excel_path, sheet_name="Sheet1", dtype=str)
        
        print("DataFrame2 생성 완료. 'ToMoveFileList' 컬럼 생성 중...")
        
        # 4. 'ToMoveFileList' 컬럼 생성
        # '___', '_' 구분자 및 형식에 맞춰 문자열 결합
        df2['ToMoveFileList'] = (
            df2['LOT'] + '_' + df2['GLS'] + '_' +
            df2['PNL'] + '_' + df2['EQP'] + '_' +
            df2['PROCESS'] + '___' + df2['X'] + '_' +
            df2['Y'] + '___' + df2['Seq'] + '_' + df2['FileExt']
        )
        print(f"이동 대상 파일 목록 수: {len(df2)}")

        # 5. 이동할 파일 목록 식별 (Set 활용)
        files_to_move_set = set(df2['ToMoveFileList'])
        
        # df1의 'FullFileList'가 Set에 포함되어 있는지 확인하여 이동할 파일 필터링
        files_to_process_df = df1[df1['FullFileList'].isin(files_to_move_set)].copy()
        
        total_to_move = len(files_to_process_df)
        print(f"\n3. 파일 이동 작업 시작... (총 {total_to_move} 개)")

        if total_to_move > 0:
            moved_count = 0
            # 6. 'Moved' 폴더 생성 및 7. 파일 이동
            for index, row in files_to_process_df.iterrows():
                src_path = row['FullPath']  # Path 객체 (예: .../GLASS001/file.JPG)
                file_name = row['FullFileList']
                
                parent_dir = src_path.parent      # .../GLASS001
                moved_dir = parent_dir / "Moved"  # .../GLASS001/Moved
                dest_path = moved_dir / file_name # .../GLASS001/Moved/file.JPG
                
                try:
                    # 6. 'Moved' 폴더 생성 (이미 있어도 오류 없음)
                    moved_dir.mkdir(exist_ok=True)
                    
                    # 7. 파일 이동
                    shutil.move(src_path, dest_path)
                    # print(f"  [성공] {file_name} -> {moved_dir.name}/")
                    moved_count += 1
                except Exception as e:
                    print(f"  [오류] {file_name} 이동 실패: {e}")
                    
            print(f"\n[완료] 총 {total_to_move} 개의 대상 파일 중 {moved_count} 개를 'Moved' 폴더로 이동했습니다.")
        else:
            print("[완료] 이동할 파일이 없습니다.")

    except Exception as e:
        print(f"[오류] DataFrame2 생성 또는 파일 이동 중 오류 발생: {e}")


# =============================
# ========== ChatGPT ==========
# =============================

# -*- coding: utf-8 -*-
# Jupyter / Python 3.x (Linux)
# 조건에 맞는 JPG 파일을 각 GLASSxxx 폴더 내 Moved/로 이동

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# ================== 설정 ==================
BASE = Path("/data_home/user/2025/username/Python/emgimages/AP5_Reservoir/FileSort")
EXCEL_PATH = BASE / "MoveList.xlsx"    # Sheet1 사용
SHEET_NAME = "Sheet1"
DRY_RUN = True                         # 먼저 True로 확인 -> OK면 False로 변경하여 실제 이동
LOG_TO_CSV = True

# ================== (1) 폴더 내 전체 파일 스캔 -> DataFrame1 ==================
# 확장자 대소문자 섞여 있어도 인식
all_imgs = [p for p in BASE.glob("GLASS*/*.[Jj][Pp][Gg]") if p.is_file()]

df1 = pd.DataFrame({
    "FullFileList": [p.name for p in all_imgs],  # 파일 이름(확장자 포함)
    "Dir":         [str(p.parent) for p in all_imgs],
    "FullPath":    [str(p) for p in all_imgs],
})
# 확인용
print(f"[INFO] 발견한 이미지 수: {len(df1)}")
display(df1.head(3))

# ================== (2) MoveList.xlsx 읽기 -> 원본 그대로 문자열로 ==================
# Excel에서 숫자/소수 형식을 건드리면 파일명 매칭이 틀어질 수 있으니 dtype=str로 고정
df2_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, dtype=str)

# 기대 컬럼 순서: LOT, GLS, PNL, EQP, PROCESS, X, Y, Seq, FileExt
expected_cols = ["LOT","GLS","PNL","EQP","PROCESS","X","Y","Seq","FileExt"]
missing = [c for c in expected_cols if c not in df2_raw.columns]
if missing:
    raise ValueError(f"[ERROR] 엑셀에 다음 컬럼이 없습니다: {missing}\n현재 컬럼: {df2_raw.columns.tolist()}")

# 공백 제거 및 누락값 처리
def _clean(s):
    if pd.isna(s): 
        return ""
    return str(s).strip()

df2 = df2_raw.copy()
for c in expected_cols:
    df2[c] = df2[c].map(_clean)

# ================== (3)(4) 파일명 문자열 합치기 -> DataFrame2(ToMoveFileList) ==================
# 규칙: LOT_GLS_PNL_EQP_PROCESS___X_Y___Seq_FileExt
df2["ToMoveFileList"] = (
    df2["LOT"] + "_" + df2["GLS"] + "_" + df2["PNL"] + "_" + df2["EQP"] + "_" + df2["PROCESS"]
    + "___" + df2["X"] + "_" + df2["Y"]
    + "___" + df2["Seq"] + "_" + df2["FileExt"]
)

# 확인용
print(f"[INFO] 이동 대상(엑셀) 행 수: {len(df2)}")
display(df2[["ToMoveFileList"]].head(5))

# ================== (5) 매칭: DataFrame1.FullFileList ↔ DataFrame2.ToMoveFileList ==================
matched = df1.merge(df2[["ToMoveFileList"]], left_on="FullFileList", right_on="ToMoveFileList", how="inner")

print(f"[INFO] 엑셀 지정 파일 중 실제 디스크에서 발견된 개수: {len(matched)}")
if len(matched) == 0:
    print("[WARN] 일치하는 파일이 없습니다. 엑셀 값(소수점/대소문자/언더바 3개 규칙)을 다시 확인하세요.")

display(matched.head(10))

# ================== (6)(7)(8) 각 GLASSxxx 폴더에 Moved/ 생성 후 이동 ==================
log = []
for row in matched.itertuples(index=False):
    src = Path(row.FullPath)
    dst_dir = src.parent / "Moved"
    dst = dst_dir / src.name

    action = "SKIP"
    reason = ""

    try:
        if not dst_dir.exists():
            if not DRY_RUN:
                dst_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            action = "MISS"
            reason = "원본없음"
        elif dst.exists():
            action = "SKIP"
            reason = "이미_이동됨(동일이름_존재)"
        else:
            action = "MOVE" if not DRY_RUN else "DRY"
            if not DRY_RUN:
                shutil.move(str(src), str(dst))
    except Exception as e:
        action = "ERROR"
        reason = repr(e)

    log.append({
        "FullFileList": row.FullFileList,
        "From": str(src),
        "To": str(dst),
        "Action": action,
        "Reason": reason
    })

df_log = pd.DataFrame(log)
print(f"[RESULT] DRY_RUN={DRY_RUN} | MOVE:{(df_log['Action']=='MOVE').sum()} DRY:{(df_log['Action']=='DRY').sum()} SKIP:{(df_log['Action']=='SKIP').sum()} MISS:{(df_log['Action']=='MISS').sum()} ERROR:{(df_log['Action']=='ERROR').sum()}")
display(df_log.head(20))

# 로그 저장(선택)
if LOG_TO_CSV:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = BASE / f"move_log_{stamp}.csv"
    df_log.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[LOG] 저장: {out_csv}")
