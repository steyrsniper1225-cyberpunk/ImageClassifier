import os
import random
from PIL import Image

# --- 경로 설정 ---
# 원본 이미지가 있는 폴더 경로
base_dir = r"C:\Users\LGPC\Desktop\TestAug"
# 저장할 폴더 경로
output_dir = os.path.join(base_dir, "saveimages")

# --- 1. 폴더 및 파일 존재 여부 확인 ---
# 저장 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 원본 이미지 폴더가 존재하는지 확인
if not os.path.isdir(base_dir):
    print(f"오류: 원본 이미지 폴더를 찾을 수 없습니다. 경로를 다시 확인해 주세요.")
    print(f"-> 확인 경로: {base_dir}")
else:
    # --- 2. 이미지 증강 작업 ---
    # base_dir 내 모든 파일 목록 가져오기
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    
    total_generated_count = 0
    # 각 파일에 대해 증강 작업 수행
    for filename in files:
        # 이미지 파일인지 확장자 확인 (png, jpeg 등 추가 가능)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_image_path = os.path.join(base_dir, filename)
            
            try:
                # 원본 이미지 열기
                original_image = Image.open(input_image_path)
                base_filename, ext = os.path.splitext(filename)

                # 좌/우 반전 10장 생성
                for i in range(1, 11):
                    flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT) # 좌/우 반전
                    x_offset = random.randint(-20, 20) # X축 shift
                    y_offset = random.randint(-20, 20) # Y축 shift
                    background = Image.new(flipped_image.mode, flipped_image.size)
                    background.paste(flipped_image, (x_offset, y_offset))              
                    output_filename = f"{base_filename}_flip_{i:04d}{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    background.save(output_path)
                    total_generated_count += 1

                # 90도 단위 회전 10장 생성
                for i in range(1, 11):
                    angle = random.choice([90, 180, 270])
                    rotated_image = original_image.rotate(angle, expand=True)
                    x_offset = random.randint(-20, 20)
                    y_offset = random.randint(-20, 20)
                    background = Image.new(rotated_image.mode, rotated_image.size)
                    background.paste(rotated_image, (x_offset, y_offset))
                    output_filename = f"{base_filename}_rot_{i:04d}{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    background.save(output_path)
                    total_generated_count += 1
                
                print(f"성공: '{filename}' 파일에 대한 이미지 증강 완료")

            except Exception as e:
                print(f"오류: '{filename}' 처리 중 예상치 못한 문제가 발생했습니다: {e}")

    print(f"\n작업 완료: 총 {total_generated_count}개의 이미지를 성공적으로 생성했습니다.")
    print(f"-> 저장 경로: '{output_dir}'")
