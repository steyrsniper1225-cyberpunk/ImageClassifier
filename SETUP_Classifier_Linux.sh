#!/bin/bash

# 에러 발생 시 스크립트 실행 중단
set -e

echo "=== 1. Conda 채널 설정 (conda-forge 고정) ==="
conda config --remove-key channels || true
conda config --add channels conda-forge
conda config --set channel_priority strict

echo "=== 2. 가상환경(ml11) 생성 (Python 3.11.13) ==="
# 이미 ml11 환경이 존재할 경우 에러가 발생할 수 있으므로 필요시 기존 환경 삭제 명령어 추가
# mamba env remove -n ml11 -y
mamba create -n ml11 python=3.11.13 -y

echo "=== 3. 가상환경 활성화 ==="
# 쉘 스크립트 내에서 conda activate를 사용하기 위한 설정
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml11

echo "=== 4. 지정된 버전의 패키지 설치 ==="
mamba install tensorflow=2.17.0 pandas=2.3.2 numpy=1.26.4 openpyxl=3.1.5 pillow=11.3.0 ipykernel tqdm -y

echo "=== 5. 환경변수(PATH, ptxas) 자동화 스크립트 생성 ==="
# 가상환경 활성화 시 실행될 스크립트
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat <<EOF > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
EOF

# 가상환경 비활성화 시 실행될 스크립트
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat <<EOF > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
#!/bin/bash
unset LD_LIBRARY_PATH
EOF

echo "=== 6. Jupyter Notebook 커널 등록 ==="
python -m ipykernel install --user --name ml11 --display-name "Python (ml11)"

echo "=== 7. 최종 검증 ==="
echo "- 현재 Python 경로:"
which python
echo "- 현재 ptxas 경로:"
which ptxas || echo "ptxas를 찾을 수 없습니다. (TensorFlow 설치 시 포함됨)"
echo "- TensorFlow GPU 인식 확인:"
python -c "import tensorflow as tf; print('사용 가능 GPU 목록:', tf.config.list_physical_devices('GPU'))"

echo "=== 가상환경(ml11) 구축 및 최적화가 완료되었습니다. ==="
