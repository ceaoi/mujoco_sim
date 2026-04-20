import os
import sys
import argparse
import runpy

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORK_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 获取输入参数--filename
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="script file name to run, relative to the scripts directory")
args = parser.parse_args()
filename = args.filename
# 检查filename是否以".py"结尾
if not filename.endswith(".py"):
    filename += ".py"

if __name__ == "__main__":
    script_path = os.path.join(WORK_DIR, "scripts", filename)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"can not find script file: {script_path}")

    runpy.run_path(script_path, run_name="__main__")