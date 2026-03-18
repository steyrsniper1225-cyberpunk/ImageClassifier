#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

echo "=================================================="
echo "   ML11 환경 구축 및 최적화 스크립트 시작"
echo "=================================================="

# [추가] Conda/Mamba 명령어 인식 설정
# 시스템에서 miniforge3 또는 anaconda 경로를 찾아 초기화 스크립트를 로드합니다.
CONDA_BASE_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")
if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
else
    echo "에러: Conda 설치 경로를 찾을 수 없습니다. Miniforge3가 설치되어 있는지 확인하세요."
    exit 1
fi

# 1. Conda 채널 및 우선순위 설정
echo "[1/7] Conda 채널 설정 중..."
conda config --remove-key channels || true
conda config --add channels conda-forge
conda config --set channel_priority strict

# 2. 가상환경 생성 (Python 3.11.13)
echo "[2/7] 가상환경(ml11) 생성 중..."
# mamba가 없을 경우를 대비해 conda로 대체 가능하도록 설정
BIN_MGR=$(command -v mamba || echo "conda")
$BIN_MGR create -n ml11 python=3.11.13 -y

# 3. 가상환경 활성화
echo "[3/7] 가상환경 활성화 중..."
conda activate ml11

# 4. 패키지 설치 (버전 고정)
echo "[4/7] 딥러닝 패키지 설치 중 (TF 2.17.0)..."
$BIN_MGR install tensorflow=2.17.0 pandas=2.3.2 numpy=1.26.4 openpyxl=3.1.5 pillow=11.3.0 ipykernel tqdm -y

# 5. 터미널용 환경변수 자동화 (activate.d)
echo "[5/7] 터미널 환경변수(ptxas 경로) 자동화 설정 중..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat <<EOF > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
EOF

mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat <<EOF > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
#!/bin/bash
unset LD_LIBRARY_PATH
EOF

# 6. Jupyter Kernel 등록 및 kernel.json 수정
echo "[6/7] Jupyter 커널 등록 및 env 블록 주입 중..."
python -m ipykernel install --user --name ml11 --display-name "Python (ml11)"

KERNEL_PATH="$HOME/.local/share/jupyter/kernels/ml11/kernel.json"
python3 - <<EOF
import json
import os

path = os.path.expanduser('$KERNEL_PATH')
with open(path, 'r') as f:
    data = json.load(f)

data['env'] = {
    "PATH": f"{os.environ['CONDA_PREFIX']}/bin:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{os.environ['CONDA_PREFIX']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
    "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={os.environ['CONDA_PREFIX']}",
    "CUDA_HOME": os.environ['CONDA_PREFIX'],
    "TF_CUDA_PATHS": os.environ['CONDA_PREFIX']
}

with open(path, 'w') as f:
    json.dump(data, f, indent=1)
EOF

# 7. 최종 검증
echo "[7/7] 설치 환경 최종 검증..."
echo "--------------------------------------------------"
echo "Python 위치: $(which python)"
echo "ptxas 위치: $(which ptxas || echo 'Not Found')"
python -c "import tensorflow as tf; print('GPU 인식 여부:', '성공' if tf.config.list_physical_devices('GPU') else '실패')"
echo "--------------------------------------------------"

echo "모든 설정이 완료되었습니다."