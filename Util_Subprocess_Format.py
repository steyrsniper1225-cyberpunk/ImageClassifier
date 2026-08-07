import subprocess
import sys

# =======================================================================

SOFT_MASK_DIR = r"C:\company_project\3O2H\model_fit\mask_v001\soft_masks"
ALIGNMENT_OUTPUT = r"C:\company_project\3O2H\model_fit\alignment_v001"

subprocess.run(
    [
        sys.executable,
        "Normal_Mask_Alignment.py",
        "--input-soft-dir", SOFT_MASK_DIR,
        "--output", ALIGNMENT_OUTPUT,
        "--rotations", "0,90,180,270",
        "--max-scale-deviation", "0.04",
    ],
    check=True,
)
