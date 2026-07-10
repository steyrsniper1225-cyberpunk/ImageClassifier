import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ======================================================
# 1. 설정
# ======================================================

base_path = "/data_home/user/2025/username/Python"
save_excel_path = os.path.join(base_path, "residual_analysis.xlsx")

rule_config = {

    "CHANNEL": "Green",

    # "VERTICAL" 또는 "HORIZONTAL"
    "DIRECTION": "VERTICAL",

    # 분석할 영역
    "START": 100,
    "END": 111
}


# ======================================================
# 2. Residual 계산
# ======================================================

def calculate_residual_array(img_pil, rule_config):

    try:

        img_arr = np.array(img_pil)

        ch_map = {
            "Red": 0,
            "Green": 1,
            "Blue": 2
        }

        ch_idx = ch_map.get(rule_config["CHANNEL"], 1)

        start = rule_config["START"]
        end = rule_config["END"]

        direction = rule_config["DIRECTION"].upper()

        # -----------------------------
        # Vertical
        # -----------------------------
        if direction == "VERTICAL":

            if start >= img_arr.shape[1] or end > img_arr.shape[1]:
                return None

            # Height 전체 × Width 일부
            roi = img_arr[:, start:end, ch_idx]

            results = []

            for col_idx in range(roi.shape[1]):

                profile = roi[:, col_idx].astype(np.float64)

                y = np.arange(len(profile))

                slope, intercept = np.polyfit(y, profile, 1)

                fitted = slope * y + intercept

                residual = profile - fitted

                for row_idx, value in enumerate(residual):

                    results.append(
                        (
                            col_idx,
                            row_idx,
                            value
                        )
                    )

            columns = ["Primary_Index", "Secondary_Index", "Value"]

        # -----------------------------
        # Horizontal
        # -----------------------------
        elif direction == "HORIZONTAL":

            if start >= img_arr.shape[0] or end > img_arr.shape[0]:
                return None

            # Height 일부 × Width 전체
            roi = img_arr[start:end, :, ch_idx]

            results = []

            for row_idx in range(roi.shape[0]):

                profile = roi[row_idx, :].astype(np.float64)

                x = np.arange(len(profile))

                slope, intercept = np.polyfit(x, profile, 1)

                fitted = slope * x + intercept

                residual = profile - fitted

                for col_idx, value in enumerate(residual):

                    results.append(
                        (
                            row_idx,
                            col_idx,
                            value
                        )
                    )

            columns = ["Primary_Index", "Secondary_Index", "Value"]

        else:

            raise ValueError(
                "DIRECTION은 VERTICAL 또는 HORIZONTAL 이어야 합니다."
            )

        return results, columns

    except Exception:

        return None


# ======================================================
# 3. Main
# ======================================================

def main():

    image_files = glob.glob(os.path.join(base_path, "*.jpg"))

    if not image_files:

        print("이미지가 없습니다.")
        return

    data_buffer = []

    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

    for file_path in tqdm(image_files):

        filename = os.path.basename(file_path)

        try:

            with Image.open(file_path) as img:

                result = calculate_residual_array(
                    img,
                    rule_config
                )

                if result is None:
                    continue

                residuals, columns = result

                temp_df = pd.DataFrame(
                    residuals,
                    columns=columns
                )

                temp_df.insert(
                    0,
                    "FileName",
                    filename
                )

                data_buffer.append(temp_df)

        except Exception as e:

            print(f"{filename} : {e}")

    if not data_buffer:

        print("저장할 데이터가 없습니다.")
        return

    final_df = pd.concat(
        data_buffer,
        ignore_index=True
    )

    final_df.to_excel(
        save_excel_path,
        index=False
    )

    print()

    print("저장 완료")
    print(save_excel_path)
    print(final_df.shape)


if __name__ == "__main__":
    main()