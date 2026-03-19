#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

echo "=================================================="
echo "   ML11 환경 구축 및 최적화 스크립트 시작"
echo "=================================================="

# [핵심] Conda/Mamba 경로 강제 로드
# 일반적인 Miniforge 설치 경로를 순차적으로 탐색합니다.
CONDA_PATHS=("$HOME/miniforge3" "$HOME/anaconda3" "/opt/miniforge3" "/data_home/user/2025/username/miniforge3")
CONDA_BASE=""

for path in "${CONDA_PATHS[@]}"; do
    if [ -f "$path/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$path"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "에러: Conda 설치 경로를 찾을 수 없습니다. 먼저 Miniforge를 설치하세요."
    exit 1
fi

# Conda 환경 초기화 로드
source "$CONDA_BASE/etc/profile.d/conda.sh"
export PATH="$CONDA_BASE/bin:$CONDA_BASE/condabin:$PATH"

# [핵심] Mamba 명령어 가용성 체크
if command -v mamba &> /dev/null; then
    BIN_MGR="mamba"
else
    echo "알림: mamba를 찾을 수 없어 conda를 사용합니다."
    BIN_MGR="conda"
fi

# 1. Conda 채널 설정
echo "[1/7] Conda 채널 설정 중..."
$BIN_MGR config --remove-key channels || true
$BIN_MGR config --add channels conda-forge
$BIN_MGR config --set channel_priority strict

# 2. 가상환경 생성 (Python 3.11.13)
echo "[2/7] 가상환경(ml11) 생성 중..."
$BIN_MGR create -n ml11 python=3.11.13 -y

# 3. 가상환경 활성화
echo "[3/7] 가상환경 활성화 중..."
conda activate ml11

# 4. 패키지 설치
echo "[4/7] 패키지 설치 중 ($BIN_MGR 사용)..."
$BIN_MGR install tensorflow=2.17.0 pandas=2.3.2 numpy=1.26.4 openpyxl=3.1.5 pillow=11.3.0 ipykernel tqdm -y

# 5. 터미널용 환경변수 자동화 (activate.d)
echo "[5/7] activate.d 설정 중..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat <<EOF > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
EOF

# 6. Jupyter Kernel 등록 및 kernel.json 수정
echo "[6/7] Jupyter 커널 및 환경변수 주입 중..."
python -m ipykernel install --user --name ml11 --display-name "Python (ml11)"

KERNEL_PATH="$HOME/.local/share/jupyter/kernels/ml11/kernel.json"
python3 - <<EOF
import json, os
path = os.path.expanduser('$KERNEL_PATH')
with open(path, 'r') as f: data = json.load(f)
data['env'] = {
    "PATH": f"{os.environ['CONDA_PREFIX']}/bin:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{os.environ['CONDA_PREFIX']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
    "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={os.environ['CONDA_PREFIX']}",
    "CUDA_HOME": os.environ['CONDA_PREFIX'],
    "TF_CUDA_PATHS": os.environ['CONDA_PREFIX']
}
with open(path, 'w') as f: json.dump(data, f, indent=1)
EOF

# 7. 최종 검증
echo "[7/7] 최종 검증..."
echo "--------------------------------------------------"
echo "Python: $(which python)"
echo "ptxas: $(which ptxas || echo 'Not Found')"
python -c "import tensorflow as tf; print('GPU 인식:', 'OK' if tf.config.list_physical_devices('GPU') else 'FAIL')"
echo "--------------------------------------------------"
