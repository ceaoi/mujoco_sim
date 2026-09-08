import os
import sys
import argparse
import runpy

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORK_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mujoco_sim.utils.mujoco_gl import configure_headless_mujoco_gl


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="script file name to run, relative to the scripts directory",
    )
    args = parser.parse_args(argv)
    filename = args.filename
    if not filename.endswith(".py"):
        filename += ".py"

    script_path = os.path.join(WORK_DIR, "scripts", filename)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"can not find script file: {script_path}")

    if configure_headless_mujoco_gl():
        print("[run_script] DISPLAY is unavailable; using MUJOCO_GL=egl")

    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
