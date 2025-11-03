# ============================
# ========== Gemini ==========
# ============================

import os
from pathlib import Path

# 1. 기본 경로 설정 (이전 스크립트와 동일)
base_path = Path("/data_home/user/2025/username/Python/emgimages/AP_Reservoir/FileSort")
print(f"작업 대상 기본 경로: {base_path}\n")

# (3) Path 하위의 모든 GLASSxxx 폴더 순회
# base_path.glob('GLASS*') -> .../FileSort/GLASS001, .../FileSort/GLASS002 ...
# .is_dir()를 통해 디렉터리인지 확인
glass_dirs = [d for d in base_path.glob('GLASS*') if d.is_dir()]

if not glass_dirs:
    print("[정보] 'GLASSxxx' 형태의 하위 폴더를 찾을 수 없습니다.")
    # 이 경우 스크립트가 더 이상 진행되지 않도록 종료하는 것이 나을 수 있습니다.
    # raise SystemExit("작업 대상 폴더 없음")

total_ok_renamed = 0
total_esd_renamed = 0

for parent_dir in glass_dirs:
    print(f"--- {parent_dir.name} 폴더 작업 시작 ---")
    
    # (1) "Moved" 폴더 내 파일명 변경 (-> _ESD)
    moved_dir = parent_dir / "Moved"
    esd_renamed_count = 0
    
    if moved_dir.is_dir():
        # .glob('*.JPG')는 해당 폴더의 JPG만 찾음 (재귀X)
        for file_path in moved_dir.glob('*.JPG'):
            # 이미 이름이 변경되었는지 확인 (중복 실행 방지)
            if not file_path.stem.endswith('_ESD') and not file_path.stem.endswith('_OK'):
                try:
                    stem = file_path.stem    # 예: LOT01..._108_1
                    suffix = file_path.suffix # 예: .JPG
                    
                    # (1) 예시: "LOT01..._108_1_ESD.JPG"
                    new_name = f"{stem}_ESD{suffix}"
                    new_path = file_path.with_name(new_name)
                    
                    file_path.rename(new_path)
                    esd_renamed_count += 1
                except Exception as e:
                    print(f"  [오류] {file_path.name} -> _ESD 변경 실패: {e}")
        
        if esd_renamed_count > 0:
            print(f"  [Moved] {esd_renamed_count} 개 파일 -> '_ESD' 추가 완료.")
        # 'Moved' 폴더가 있으나 변경할 파일이 없는 경우(모두 변경됨) 조용히 넘어감
        
    else:
        # 'Moved' 폴더 자체가 없는 경우 (파일이 모두 _OK이거나, 아직 이동 작업 안 함)
        print(f"  [Moved] 'Moved' 폴더가 없습니다. (_ESD 작업 스킵)")

    # (2) "parent_dir" (GLASSxxx) 폴더 내 파일명 변경 (-> _OK)
    # .glob('*.JPG')는 'parent_dir' 바로 하위의 JPG 파일만 검색
    # 'Moved' 폴더 안의 파일은 검색되지 않음
    ok_renamed_count = 0
    
    for file_path in parent_dir.glob('*.JPG'):
        # 이미 이름이 변경되었는지 확인 (중복 실행 방지)
        if not file_path.stem.endswith('_ESD') and not file_path.stem.endswith('_OK'):
            try:
                stem = file_path.stem    # 예: LOT01..._1239_1
                suffix = file_path.suffix # 예: .JPG
                
                # (2) 예시: "LOT01..._1239_1_OK.JPG"
                new_name = f"{stem}_OK{suffix}"
                new_path = file_path.with_name(new_name)
                
                file_path.rename(new_path)
                ok_renamed_count += 1
            except Exception as e:
                print(f"  [오류] {file_path.name} -> _OK 변경 실패: {e}")

    if ok_renamed_count > 0:
        print(f"  [OK] {ok_renamed_count} 개 파일 -> '_OK' 추가 완료.")
    
    total_esd_renamed += esd_renamed_count
    total_ok_renamed += ok_renamed_count
    print(f"--- {parent_dir.name} 폴더 작업 완료 ---")

print("\n--- 전체 작업 요약 ---")
print(f"총 {total_esd_renamed} 개의 파일에 '_ESD'가 추가되었습니다.")
print(f"총 {total_ok_renamed} 개의 파일에 '_OK'가 추가되었습니다.")
print("[스크립트 완료]")


# =============================
# ========== ChatGPT ==========
# =============================

from pathlib import Path

BASE = Path("/data_home/user/2025/username/Python/emgimages/AP5_Reservoir/FileSort")
DRY_RUN = True  # 먼저 True로 테스트 → OK면 False로 변경

def rename_file_add_suffix(file_path: Path, suffix: str):
    """
    파일 확장자 바로 앞에 suffix를 넣어서 rename
    ex) name.JPG -> name_suffix.JPG
    """
    if not file_path.is_file():
        return
    
    new_name = f"{file_path.stem}_{suffix}{file_path.suffix}"
    new_path = file_path.with_name(new_name)

    if DRY_RUN:
        print(f"[DRY] {file_path.name} -> {new_path.name}")
    else:
        file_path.rename(new_path)

# ==============================
# (1) Moved 폴더 → _ESD
# ==============================
for glass_dir in BASE.glob("GLASS*"):
    moved_dir = glass_dir / "Moved"
    
    if not moved_dir.exists():
        continue
    
    for img in moved_dir.glob("*.JPG"):
        rename_file_add_suffix(img, "ESD")
    for img in moved_dir.glob("*.jpg"):
        rename_file_add_suffix(img, "ESD")

# ==============================
# (2) 원본 폴더(=GLASSxxx) → _OK
# ==============================
for glass_dir in BASE.glob("GLASS*"):
    moved_dir = glass_dir / "Moved"

    for img in glass_dir.glob("*.JPG"):
        # Moved 폴더에 있는 파일은 제외
        if img.parent.name == "Moved":
            continue
        rename_file_add_suffix(img, "OK")

    for img in glass_dir.glob("*.jpg"):
        if img.parent.name == "Moved":
            continue
        rename_file_add_suffix(img, "OK")

print("=== Done ===")
print(f"DRY_RUN = {DRY_RUN}")
