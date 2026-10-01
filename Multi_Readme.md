Multi-ROI baseline — Legacy 리팩토링 안내

1. 파일 구성

  Main_Training_MultiROI.py   : 수정 학습 스크립트
  Main_Inference_MultiROI.py  : 수정 추론 스크립트
  multi_roi_common.py        : 공용 ROI CONFIG / 5채널 전처리 / 모델 / weight 이식
  check_multi_roi.py         : 실제 데이터 없이 실행하는 구조·학습·저장 검증

세 구현 파일은 반드시 같은 폴더에 두세요. 원본 두 파일은 수정하지 않았습니다.
실제 학습 데이터, 운영 template, Legacy 학습 weight는 첨부되지 않았습니다.
이 패키지에는 학습 완료 모델이 포함되어 있지 않습니다.

2. 처음 실행하는 순서

  (1) multi_roi_common.py 상단 CONFIG의 INTERNAL_ROIS를 실제 defect zone으로 수정.
      제공 좌표는 예시이며 실제 공정에서 검증된 좌표가 아닙니다.
      global 1개는 자동 포함됩니다. 리스트에 global을 다시 넣지 마세요.
      기본 local은 64x64 4개 + 32x32 3개입니다.

      {"name": "mid_1", "x": 32, "y": 32, "width": 64, "height": 64}
      => 정렬된 global 이미지의 image[32:96, 32:96] 영역.

      x=가로/열, y=세로/행. 원점은 왼쪽 위. 끝 좌표는 미포함.
      항목 추가/삭제로 개수 변경, width/height로 크기 변경, x/y로 위치 변경.
      리스트 순서가 concat 순서입니다. 중복 이름/음수/범위 초과는 오류 처리.

  (2) Main_Training_MultiROI.py 상단 CONFIG에서 DATA_ROOT, WORK_DIR 수정.
      DATA_ROOT/OK, DATA_ROOT/ESD에 이미 crop/회전 정렬된 256x256 RGB 이미지 배치.
      Training에는 원본 full-frame template matching을 추가하지 않았습니다.
      다른 크기는 조용히 resize하지 않고 오류를 냅니다.

  (3) python check_multi_roi.py
      python Main_Training_MultiROI.py

      WORK_DIR/실행시각/ 아래에 Stage1, Stage2, Final .keras 파일과 각각의
      .keras.roi.json, split.csv, training_log.csv, training_config.json 생성.
      Final은 stage 2에서 val_accuracy가 가장 높았던 weight입니다.
      기존 모델을 덮어쓰지 않도록 실행마다 새 하위 폴더를 만듭니다.

  (4) Main_Inference_MultiROI.py의 BASE_PATH와 TEMPLATE_A_BASE64를 실제 값으로 수정.
      CONFIG["A"]["MODEL_PATH"]에 학습이 출력한 MultiROI_Final.keras 절대 경로 설정.
      모델을 다른 폴더로 옮길 때 .keras.roi.json도 함께 복사하세요.

      python Main_Inference_MultiROI.py

      기존처럼 BASE_PATH/EMG_IMAGE에서 읽고, Glass별 OK/ESD 폴더로 원본 파일을
      이동합니다(shutil.move 유지). 최초 비교는 테스트 이미지 복사본으로 실행하세요.
      Excel 컬럼, rule/model 결합 판정, focus_out 파일명 조건도 원본 유지.

3. 구현 구조

  aligned RGB 256x256
    -> Legacy per_image_standardization
    -> global min/max normalization (+1e-6)
    -> RGB + RGB별 Sobel magnitude 평균 + inverse gray = 5채널
    -> global 1개 + fixed-coordinate local crops
    -> local 5채널 전체를 bilinear로 256x256 확대
    -> 동일한 shared encoder: Conv1x1 5->16 ReLU -> Conv1x1 16->3
                               -> ResNet50V2(include_top=False) -> GAP 2048
    -> [global, INTERNAL_ROIS 리스트 순서]의 feature concat
    -> Dropout(0.3) -> Dense(128, ReLU, L2=1e-4) -> Dropout(0.3)
    -> Dense(1, sigmoid, float32) = P(OK)

  기본 8개 view는 2048*8=16384차원 concat -> 128 -> 1입니다.
  encoder를 여러 번 호출하지만 encoder 인스턴스와 weight는 하나입니다.
  attention, FPN/BiFPN, ROI별 classifier, projection은 추가하지 않았습니다.
  학습/추론 입력은 둘 다 [B,256,256,5], 출력은 [B,1]입니다.
  crop/resize가 .keras 모델 안에 저장되므로 별도 view 배열을 만들 필요가 없습니다.

4. 그대로 유지한 것과 필요한 차이

  - ESD=0, OK=1. 출력은 Defect 확률이 아니라 OK 확률입니다.
    추론 THRESHOLD=0.01과 판정 방향을 보존했습니다. Defect 확률은 1-P(OK).
  - 전처리 수식과 연산 순서를 그대로 유지. Sobel은 [0,1]로 추가 정규화하지 않음.
  - 원본의 preprocess_input은 import만 되고 사용되지 않았으므로 새 코드에도 적용하지 않음.
  - 원본 ADD_CANNY 변수와 달리 실제 5번째 채널은 Canny가 아니라 inverse gray였음.
  - 파일 정렬/shuffle/stratified split, class weight, loss, 학습률,
    epoch 수, EarlyStopping, ReduceLROnPlateau monitor/patience 유지.
  - Stage1: mapper/head만 학습. Stage2: mapper/head + conv5_* 중 BN 이외 학습.
    BN은 고정하고 shared ResNet 호출에도 training=False를 명시.
  - 배치는 16 -> 2로 변경. 8 views 때문에 연산량은 단일 view보다 대략 8배이며,
    실제 메모리/속도는 GPU와 학습 단계에 따라 달라집니다. 부족하면 1로 낮추세요.
    encoder를 공유해도 activation 메모리와 연산량이 같아지는 것은 아닙니다.
  - Legacy JPEG decoder 유지 + PNG decoder 지원. Inference PIL 이미지는 RGB로 변환.
  - 추론 매칭 실패 반환값을 (None,None,None)으로 통일하고 누락된 조기 return 보완.
  - model output NaN/Inf 검사와 저장된 ROI CONFIG/concat 순서 검사 추가.
  - 원본에서 주석 처리되어 있던 history/final model 저장을 실제 동작하도록 정리.

5. augmentation과 비교 시 주의

  원본은 5채널을 만든 뒤 flip/brightness/contrast를 적용했습니다.
  새 코드도 이 순서를 유지하며, global 5채널에 한 번 적용한 다음 모델에서 crop합니다.
  Sobel/inverse gray도 함께 증강되는 원본 동작을 의도적으로 유지했습니다.

  AUG_FLIP_LEFT_RIGHT=True가 기본입니다. 고정 defect zone이 좌우 비대칭이면
  flip이 결함을 local ROI 밖으로 옮길 수 있습니다. 그 경우 False로 설정하고,
  Base도 동일한 증강 조건으로 맞춰 비교하세요. view마다 독립적으로 flip하지 않습니다.

  같은 정렬 RGB 픽셀을 넣으면 Training/Inference의 5채널 결과는 동일합니다.
  JPEG 학습 파일을 별도로 재압축했거나 PIL/TF decoder, EXIF 처리, crop 정렬이 다르면
  입력 픽셀 자체가 달라질 수 있습니다. 학습용 crop은 동일한 방향으로 저장되어야 하며,
  픽셀 단위 재현이 필요하면 동일 pipeline에서 추출한 무손실 PNG를 사용하세요.

  Base vs Multi-ROI만 비교하면 됩니다. 같은 split/test set과 rule 조건을 사용하고,
  최종 OK/ESD뿐 아니라 CNN P(OK)를 따로 비교하세요. THRESHOLD=0.01은 Legacy 값으로,
  새 모델에서 같은 FPR을 보장하지 않습니다. validation에서 threshold를 정한 뒤
  고정된 test에서 동일 Normal FPR 기준 Defect Recall, FN/FP, 처리시간을 비교하세요.
  배치 변경으로 optimizer step 수와 학습 난수열은 달라질 수 있습니다.

6. Legacy weight 이식 (선택)

  Main_Training_MultiROI.py:
      CONFIG["LEGACY_MODEL_PATH"] = "/absolute/path/Legacy_Stage2.keras"

  시작할 때 전체 .keras에서 ch_mapper_16, ch_mapper_3와 ResNet50V2만 복사합니다.
  원본 classifier는 복사하지 않습니다. layer/weight shape가 안 맞으면 오류를 냅니다.
  함수는 multi_roi_common.transplant_legacy_weights()에 있습니다.
  weights-only 파일은 원본 모델을 구성해서 load_weights 후 전체 .keras로 먼저 저장하세요.

  기본 LEGACY_MODEL_PATH=None이면 Legacy 모델 없이 ImageNet 초기값으로 학습합니다.
  ImageNet 다운로드도 피하려면 BACKBONE_WEIGHTS=None으로 설정하세요.
  그 경우 random backbone을 고정하지 않고 두 단계 모두 전체 non-BN layer를 학습합니다.
  완전 scratch는 학습 가능하지만 기존 10+10 epoch만으로 수렴을 보장하지 않습니다.

7. 검증 범위

  Python 3.13 / TensorFlow 2.21.0 / Keras 3.15.1 CPU 환경에서 확인.
  Float32와 mixed_float16 양쪽에서 check_multi_roi.py 통과:
    전처리/회전 보정 일치, 좌표 crop/resize, encoder 공유, concat 크기,
    단계별 gradient update, BN 고정, 이식, save/load 출력 일치,
    batch 크기 변경, ROI 설정 불일치 거부.
  원본 코드를 별도로 비교하여 5채널 수식, split 결과, 네 가지 rule/model 판정,
  파일 분류, 마지막 잔여 배치, Excel/focus_out 결과의 동등성을 확인.
  합성 데이터로 Training main 전체를 두 단계 각 1 epoch씩 실행하여 graph-mode fit,
  class weight, callback, Stage1/Stage2/Final 저장·재로드, CSV 생성도 확인.

  실제 공정 이미지의 정확도/recall/FPR, GPU 메모리/속도, 사용 중인 Legacy checkpoint의
  실제 이식은 해당 데이터/weight가 없어 검증하지 않았습니다. 이식 경로 자체는
  원본 topology를 따른 합성 checkpoint로 검사했습니다.
  기존 실행 환경에서 check_multi_roi.py를 먼저 실행하세요.
  별도의 custom layer나 unsafe Lambda deserialization 설정은 필요 없습니다.

  구현 확인에 참고한 공식 문서:
  Keras transfer learning / BN 동작: https://keras.io/guides/transfer_learning/
  TensorFlow Resizing: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing
  TensorFlow Cropping2D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
