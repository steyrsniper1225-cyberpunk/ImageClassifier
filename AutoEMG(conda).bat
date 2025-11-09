@ECHO OFF
CHCP 65001

TITLE [자동화] 1. 이미지 추론 >> 2. Excel 병합

ECHO =======================================================
ECHO  Conda 가상환경 (tf_win)을 활성화합니다...
ECHO =======================================================

:: (수정) 'Scripts\activate.bat' 대신 'condabin\conda.bat'를 직접 호출합니다.
:: 이것이 배치 파일에서 Conda 환경을 활성화하는 가장 안정적인 방법입니다.
CALL D:\Miniforge3\condabin\conda.bat activate tf_win

:: (중요) 활성화 실패 시 오류 검사
IF %ERRORLEVEL% NEQ 0 (
    ECHO [오류] 'D:\Miniforge3\condabin\conda.bat activate tf_win' 실행에 실패했습니다.
    ECHO Miniforge3 설치 경로와 'tf_win' 환경 이름을 확인하세요.
    PAUSE
    EXIT /B
)

ECHO 가상환경이 성공적으로 활성화되었습니다. (활성화된 Python: %CONDA_PYTHON_EXE%)
ECHO =======================================================
ECHO  [1/2] 이미지 추론 스크립트를 시작합니다...
ECHO =======================================================

:: 'conda activate'가 PATH를 올바르게 설정했으므로 'python'만 호출해도 됩니다.
python D:\Test_Inference_AutoCrop.py

ECHO.
ECHO =======================================================
ECHO  [2/2] Excel 병합 유틸리티를 시작합니다...
ECHO =======================================================

python D:\Test_ResultFile_Concat.py

ECHO.
ECHO =======================================================
ECHO  모든 작업이 완료되었습니다.
ECHO =======================================================

PAUSE
