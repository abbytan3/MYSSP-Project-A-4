"""DAF Modal job (T15). Follows detectors/dtd/modal_job.py. Run from this laptop:

  0. Environment check (T4 GPU, a few minutes, no pages; builds the image, loads BOTH weight files and
     checks that each reports 0 missing and 0 unexpected keys, runs one random-noise image):
       modal run detectors/daf/modal_job.py --stage env
  1. DAF (one T4 GPU, one weight and one size at a time):
       modal run detectors/daf/modal_job.py --stage daf --weight anytext --size shrink512 --pages page_0001,page_0003,page_0009
       modal run detectors/daf/modal_job.py --stage daf --weight anytext --size full      --pages page_0001,page_0003,page_0009
       (same two lines with --weight diffste)
       modal run detectors/daf/modal_job.py --stage daf --weight anytext --size shrink512      (all pages)
       modal run detectors/daf/modal_job.py --stage daf --weight anytext --size full           (all pages)
  2. Check a finished weight and size (CPU):
       modal run detectors/daf/modal_job.py --stage check_daf --weight anytext --size shrink512

Reads pages from the private Volume `a4-data` (pages/<page_id>.<ext>) and writes
  out/daf_<weight>/<size>/<page_id>.npz      compact output (shared/output_format.py)
  out/daf_<weight>/<size>/run_info.json, run_log.csv
Weights must already be on the Volume under weights/daf/ (see detectors/daf/README.md).

This file runs on Modal's own Python (3.12). DAF needs Python 3.9, so the image holds a second,
separate Python 3.9 environment (/opt/daf_env) and detectors/daf/infer.py runs inside it as a subprocess.
Only the Volume `a4-data` is used. Nothing about a page is ever printed: only anonymous IDs and numbers.
"""

import json
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------- settings
# Name of the private Volume that holds pages, weights and outputs.
VOLUME_NAME = "a4-data"
# Where the Volume appears inside every container.
DATA = "/data"
# The authors' code, pinned to the commit used for this project (HEAD of the default branch on 2026-11-19).
REPO_URL = "https://github.com/qcf-568/OSTF.git"
REPO_COMMIT = "1fec955ccd85445f4fc231b0b36cf7844a17a46f"
REPO_DIR = "/opt/OSTF"
# The Python 3.9 environment DAF runs in, and the file listing the versions that must not change.
ENV_DIR = "/opt/daf_env"
ENV_PY = ENV_DIR + "/bin/python"
CONSTRAINTS = "/opt/daf_constraints.txt"
# Lines of the inference script that are shown on the console (everything else stays in the container).
SHOW_PREFIXES = ("model ready", "page ", "RESULT", "  ")

# Folder of this file, and the shared folder (both are copied into the image).
HERE = Path(__file__).resolve().parent
# Inside the Modal container this file sits at /root, which has no grandparent folder; the
# paths are only used on the laptop when the image is built, so any value is fine there.
SHARED = HERE.parents[1] / "shared" if len(HERE.parents) > 1 else HERE

app = modal.App("a4-daf")
volume = modal.Volume.from_name(VOLUME_NAME)

# ---------------------------------------------------------------- image
# Wheel sources (never build mmcv from source: it takes hours).
MMCV_FIND = "https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html"
NATTEN_FIND = "https://shi-labs.com/natten/wheels/"
UV = "uv pip install --python " + ENV_PY + " -c " + CONSTRAINTS
# Versions follow the authors' requirements.txt (Python 3.9, torch 1.13.0, mmcv 2.0.0, mmengine 0.8.5,
# timm 0.6.13) where they install; each difference is marked in detectors/daf/requirements.txt.
daf_image = (
    # Modal's own Python (3.12); the DAF environment below is separate and is Python 3.9.
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        # A fresh Python 3.9 environment (uv downloads Python 3.9 itself).
        f"uv venv --python 3.9 {ENV_DIR}",
        # Versions that later installs must not change.
        f"printf 'torch==1.13.0\\ntorchvision==0.14.0\\nnumpy==1.22.4\\n' > {CONSTRAINTS}",
        # torch 1.13.0 from PyPI (the Linux build bundles CUDA 11.7), and numpy.
        f"{UV} torch==1.13.0 torchvision==0.14.0 numpy==1.22.4",
        # setuptools (mmengine imports pkg_resources); yapf 0.40.1 (newer yapf breaks mmengine 0.8.5).
        f"{UV} 'setuptools<70' yapf==0.40.1",
        # OpenCV before mmcv, so mmcv does not pull a numpy-2 OpenCV.
        f"{UV} opencv-python==4.8.0.76 opencv-python-headless==4.8.0.76",
        f"{UV} mmengine==0.8.5",
        # mmcv with compiled CUDA ops: 2.0.0, else the closest 2.0.x prebuilt wheel.
        f"({UV} mmcv==2.0.0 --find-links {MMCV_FIND} --only-binary mmcv || "
        f"{UV} mmcv==2.0.1 --find-links {MMCV_FIND} --only-binary mmcv)",
        f"{UV} timm==0.6.13",
        # The other packages the DAF mmdet imports when it loads.
        f"{UV} Pillow==9.0.1 scipy==1.7.3 scikit-image==0.19.2 Shapely==2.0.6 pycocotools==2.0.7 "
        "terminaltables==3.1.10 six==1.16.0 rich==13.8.1 matplotlib==3.5.1 tqdm==4.64.0",
        # numba 0.58.1 needs numpy>=1.22, which matches the pin above; albumentations as documented.
        f"{UV} numba==0.58.1 albumentations==1.3.1",
        # natten is only needed by the NAT backbone, which the Faster R-CNN Swin configs never import: optional.
        f"({UV} natten==0.17.1 --find-links {NATTEN_FIND} --only-binary natten || echo 'natten not installed (not needed)')",
    )
    # The authors' code at the pinned commit; README inference step: mv mmdet mmdet_train; unzip mmdet_test.zip.
    .run_commands(
        f"git clone -q {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout -q {REPO_COMMIT}",
        f"cd {REPO_DIR} && mv mmdet mmdet_train && unzip -q mmdet_test.zip",
        f"find {REPO_DIR}/mmdet -name __pycache__ -type d -prune -exec rm -rf {{}} +",
    )
    # Our own files, copied in last so editing them does not rebuild the image.
    .add_local_file(str(HERE / "infer.py"), "/app/infer.py")
    .add_local_file(str(HERE / "tiling.py"), "/app/tiling.py")
    .add_local_file(str(SHARED / "output_format.py"), "/app/output_format.py")
)


def run_script(args):
    """Run /app/infer.py in the Python 3.9 environment and stream its output.

    The script prints COMMIT when the Volume should be saved, and one RESULT line with numbers.
    Returns the parsed RESULT, or raises with the end of the output if the script failed.
    """
    command = [ENV_PY, "-u", "/app/infer.py"] + list(args) + ["--repo-commit", REPO_COMMIT]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/app")
    last, result = [], None
    for line in process.stdout:
        line = line.rstrip()
        last = (last + [line])[-30:]
        if line == "COMMIT":
            # Save what the script has written so far, so a stopped run loses little.
            volume.commit()
        elif line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
            print(line, flush=True)
        elif line.startswith(SHOW_PREFIXES):
            print(line, flush=True)
    process.wait()
    volume.commit()
    # The script returns 1 when a check fails; the RESULT line then holds the details.
    if process.returncode != 0 and result is None:
        raise RuntimeError("DAF script failed (re-run to resume):\n" + "\n".join(last))
    return result, process.returncode


# ---------------------------------------------------------------- stage 0: environment check (T4)
@app.function(image=daf_image, gpu="T4", volumes={DATA: volume}, timeout=3600)
def env_check():
    """Load both weights, report missing/unexpected keys, run one random-noise image. No pages."""
    volume.reload()
    result, code = run_script(["--mode", "env"])
    result["exit_code"] = code
    return result


# ---------------------------------------------------------------- stage 1: DAF
@app.function(image=daf_image, gpu="T4", volumes={DATA: volume}, timeout=12 * 3600)
def run_daf(weight, size, limit=0, names=""):
    """Run DAF with one weight on every page at one size and save the compact outputs."""
    volume.reload()
    args = ["--mode", "run", "--weight", weight, "--size", size, "--limit", str(limit)]
    if names:
        args += ["--pages", names]
    result, _ = run_script(args)
    return result


# ---------------------------------------------------------------- stage check
@app.function(image=daf_image, volumes={DATA: volume}, timeout=3600)
def check_stage(weight, size):
    """Check one finished weight and size on the Volume. Returns numbers and anonymous IDs only."""
    volume.reload()
    result, code = run_script(["--mode", "check", "--weight", weight, "--size", size])
    result["exit_code"] = code
    return result


# ---------------------------------------------------------------- entry point
@app.local_entrypoint()
def main(stage: str, weight: str = "anytext", size: str = "shrink512", limit: int = 0, pages: str = ""):
    # stage: "env", "daf" or "check_daf". weight: anytext or diffste. limit: 0 = all pages.
    # pages: comma-separated anonymous IDs (the agreed test pages).
    if weight not in ("anytext", "diffste"):
        raise SystemExit("weight must be 'anytext' or 'diffste'")
    if stage == "env":
        result = env_check.remote()
        # One machine-readable line (numbers and weight file names only).
        print("ENV " + json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)
    elif stage == "daf":
        print(run_daf.remote(weight, size, limit, pages))
    elif stage == "check_daf":
        result = check_stage.remote(weight, size)
        print("CHECK " + json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)
    else:
        raise SystemExit("stage must be 'env', 'daf' or 'check_daf'")
