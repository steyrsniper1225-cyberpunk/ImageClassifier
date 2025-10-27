# === 현재 노트북 커널이 실제로 쓰는 경로/버전 확인 (단일 셀) ===
import os, sys, site, importlib, json
info = {}

# 워킹디렉토리/홈
info["cwd"] = os.getcwd()
info["HOME(~)"] = os.path.expanduser("~")

# 파이썬 실행 파일/버전
info["python_executable"] = sys.executable
info["python_version"] = sys.version

# site-packages 위치
sp = {}
sp["getsitepackages"] = site.getsitepackages() if hasattr(site, "getsitepackages") else []
sp["getusersitepackages"] = site.getusersitepackages() if hasattr(site, "getusersitepackages") else ""
info["site_packages"] = sp

# sys.path 앞쪽 10개만
info["sys.path_head"] = sys.path[:10]

# 주요 패키지 로딩 경로(있으면)
def mod_path(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__file__", "built-in or namespace")
    except Exception as e:
        return f"NOT FOUND: {e.__class__.__name__}: {e}"

mods = {}
for m in ["numpy", "pandas", "tensorflow", "jupyter", "ipykernel"]:
    mods[m] = mod_path(m)
info["module_files"] = mods

# Jupyter 경로 정보
try:
    from jupyter_core.paths import jupyter_config_dir, jupyter_data_dir, jupyter_path
    info["jupyter_config_dir"] = jupyter_config_dir()
    info["jupyter_data_dir"] = jupyter_data_dir()
    info["jupyter_path"] = jupyter_path()
except Exception as e:
    info["jupyter_paths_error"] = str(e)

# 커널 연결 파일 위치(현재 커널이 어디서 떴는지 힌트)
try:
    import ipykernel
    from ipykernel.connect import get_connection_file
    info["kernel_connection_file"] = get_connection_file()
except Exception as e:
    info["kernel_connection_file_error"] = str(e)

print(json.dumps(info, indent=2, ensure_ascii=False))

# ===============================================================
# ===============================================================
# ===============================================================

# === 설치된 커널스펙 vs 현재 커널(Python) 실행 경로 비교 ===
import sys, json
from jupyter_client.kernelspec import KernelSpecManager

current_py = sys.executable

rows = []
ksm = KernelSpecManager()
all_specs = ksm.get_all_specs()  # {name: {"resource_dir": str, "spec": KernelSpec}}

for name, info in all_specs.items():
    spec = info["spec"]
    argv = getattr(spec, "argv", [])
    display = getattr(spec, "display_name", name)
    resource_dir = info.get("resource_dir", "")
    match = (len(argv) > 0 and argv[0] == current_py)
    rows.append({
        "kernel_name": name,
        "display_name": display,
        "argv0": argv[0] if argv else "",
        "argv": argv,
        "resource_dir": resource_dir,
        "argv0_matches_current_python": match
    })

out = {
    "current_python_executable": current_py,
    "kernelspecs": rows
}
print(json.dumps(out, indent=2, ensure_ascii=False))

# ===============================================================
# ===============================================================
# ===============================================================

# === 설치된 Jupyter 커널들의 kernel.json 직접 검사 ===
import os, sys, json
from jupyter_client.kernelspec import find_kernel_specs

current_py = os.path.abspath(sys.executable)
rows = []

specs = find_kernel_specs()  # {kernel_name: resource_dir}
for name, res_dir in specs.items():
    kernel_json = os.path.join(res_dir, "kernel.json")
    try:
        with open(kernel_json, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        rows.append({
            "kernel_name": name,
            "resource_dir": res_dir,
            "error": f"READ_FAIL: {e}"
        })
        continue

    argv = spec.get("argv", [])
    display = spec.get("display_name", name)
    lang = spec.get("language", "")
    argv0 = argv[0] if argv else ""
    # argv[0]와 현재 파이썬 실행 파일이 동일한지 (가능한 경우 samefile로 비교)
    def same(a, b):
        try:
            return os.path.samefile(a, b)
        except Exception:
            return os.path.abspath(a) == os.path.abspath(b)

    match = bool(argv) and same(argv0, current_py)

    rows.append({
        "kernel_name": name,
        "display_name": display,
        "language": lang,
        "resource_dir": res_dir,
        "argv0": argv0,
        "argv_len": len(argv),
        "argv0_matches_current_python": match,
        "kernel_json_path": kernel_json
    })

out = {
    "current_python_executable": current_py,
    "kernels_found": len(rows),
    "kernels": rows
}
print(json.dumps(out, indent=2, ensure_ascii=False))
