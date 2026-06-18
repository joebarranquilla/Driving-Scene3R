#!/usr/bin/env bash
# =============================================================================
# run_3dgs_e2e.sh  —  End-to-end driver for the semantic_gs Phase-0/1/2/3 test
#                     on the TUM i9 workstation (e.g. s0045@devcube2).
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Stages 1-3  : one-time setup (conda env, MSNet repo, MSNet checkpoint check)
# Stages 4-6  : generate the teammate inputs your module consumes
#                 4) Mask2Former panoptic NPZs   (GPU,  ~5-10 min for seq 04)
#                 5) MobileStereoNet depth NPZs  (GPU,  ~5-10 min for seq 04)
#                 6) lift_to_semantic_pointcloud.py  →  PLY + NPZ  (CPU, ~30 s)
# Stages 7-10 : >>> THE ACTUAL TESTS OF *YOUR* CODE <<<
#                 7) Phase 0  — pytest hermetic test suite              (CPU, ~5 s)
#                 8) Phase 1  — smoke_test_adapters on real KITTI       (CPU, ~10 s)
#                 9) Phase 2  — init_pointcloud on the real lift output (CPU, ~5 s)
#                10) Phase 3  — gsplat training (RGB only) on real KITTI
#                              → renders, PSNR/SSIM, Inria-format PLY   (GPU, ~3-5 min)
#
# Every stage is idempotent: re-running skips work that's already done.
#
# PYTHON ENVIRONMENT
# ------------------
# The script prefers conda when available (env name $CONDA_ENV, default
# `driving-scene3r`). When conda is NOT installed it transparently falls
# back to a plain venv at $VENV_DIR (default $AFM_ROOT/.venv), built from
# the first python3.10/3.11/3.12 on PATH. Either path is fine for testing;
# conda is recommended for reproducibility and to match the main README.
# To install conda without sudo, see Option A in the chat instructions
# (miniforge → $AFM_ROOT/miniforge3, ~3 min).
#
# QUICK START
# -----------
#     chmod +x run_3dgs_e2e.sh
#     ./run_3dgs_e2e.sh                       # full pipeline, seq 04, 50 frames
#     ./run_3dgs_e2e.sh -s 05 -n 100          # seq 05, 100 frames for lift
#     ./run_3dgs_e2e.sh --only-test           # rerun just the Phase 0/1/2 tests
#     ./run_3dgs_e2e.sh -h                    # all options
#
# DEFAULTS (override with flags or env vars)
# ------------------------------------------
#   AFM_ROOT    = ~/afm/x                     (your main working dir)
#   Outputs     = $AFM_ROOT/outputs/          (depth, panoptic, clouds, viz)
#   MSNet repo  = $AFM_ROOT/mobilestereonet/
#   MSNet ckpt  = $AFM_ROOT/checkpoints/MSNet2D_SF_KITTI2015.ckpt
#   HF cache    = $AFM_ROOT/hf_cache/         (~1 GB for Mask2Former weights)
#   conda env   = driving-scene3r
#   KITTI mirror= /storage/group/dataset_mirrors/kitti_odom_color/...
#
# =============================================================================
set -Eeuo pipefail

# Loud failure: if anything exits non-zero under set -e, print which
# command and on which line. Without this, set -e aborts SILENTLY.
trap 'rc=$?; printf "\n\033[1;31m[ABORT]\033[0m line %s: command \"%s\" exited with status %s\n" \
      "${BASH_LINENO[0]}" "${BASH_COMMAND}" "$rc" >&2; exit $rc' ERR

# Tolerant `find` for "count NPZs in a maybe-missing dir". Returns 0 when
# the dir does not exist, instead of crashing the script under pipefail.
count_npz() {
  local dir="$1"
  [[ -d "$dir" ]] || { echo 0; return 0; }
  find "$dir" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l
}

# -----------------------------------------------------------------------------
# CUDA toolkit (nvcc) discovery + install.
#
# gsplat 1.5.x ships a Python sdist whose CUDA kernels are JIT-compiled at
# first use. That JIT step needs `nvcc` on PATH and a working CUDA_HOME.
# Most TUM workstations DO have a system CUDA toolkit, but it isn't always
# on PATH; if it isn't, we fall back to NVIDIA's official PyPI wheel
# (`nvidia-cuda-nvcc-cu13` / `cu12`) which works entirely inside the venv
# with no sudo / no module-load.
#
# Sets PATH and CUDA_HOME on success and propagates them to subsequent
# stages (including Stage 10's training process).
# -----------------------------------------------------------------------------
ensure_nvcc() {
  if command -v nvcc &>/dev/null; then
    log "nvcc already on PATH: $(command -v nvcc)  ($(nvcc --version | grep release || true))"
    [[ -z "${CUDA_HOME:-}" ]] && export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    return 0
  fi

  local d
  for d in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.0 \
           /usr/local/cuda-12 /usr/local/cuda-12.1 /usr/local/cuda-12.4 \
           /opt/cuda /usr/lib/cuda; do
    if [[ -x "$d/bin/nvcc" ]]; then
      log "Found system nvcc at $d/bin/nvcc — adding to PATH."
      export PATH="$d/bin:$PATH"
      export CUDA_HOME="$d"
      return 0
    fi
  done

  log "No system nvcc found — installing it via pip (no sudo required)."
  log "(Trying nvidia-cuda-nvcc-cu13 first to match torch+cu130, then cu12 as fallback.)"
  local pkg
  for pkg in nvidia-cuda-nvcc-cu13 nvidia-cuda-nvcc-cu12; do
    if pip install --progress-bar off "$pkg" 2>&1 | tail -n 5; then
      local nvcc_bin
      nvcc_bin=$(python - <<'PY'
import importlib, os, glob
for name in ("nvidia.cuda_nvcc",):
    try:
        m = importlib.import_module(name)
    except Exception:
        continue
    cands = glob.glob(os.path.join(os.path.dirname(m.__file__), "bin", "nvcc"))
    if cands:
        print(cands[0]); break
PY
)
      if [[ -n "$nvcc_bin" && -x "$nvcc_bin" ]]; then
        log "Installed nvcc at $nvcc_bin"
        export PATH="$(dirname "$nvcc_bin"):$PATH"
        export CUDA_HOME="$(dirname "$(dirname "$nvcc_bin")")"
        return 0
      fi
    fi
  done

  err "Could not find or install nvcc. Phase 3 (gsplat) cannot JIT-compile."
  err "Manual fix on this machine:"
  err "    module load cuda    # if your cluster uses Lmod"
  err "  OR"
  err "    export PATH=/path/to/cuda/bin:\$PATH"
  err "    export CUDA_HOME=/path/to/cuda"
  return 1
}

# Real end-to-end gsplat sanity check: triggers JIT compile + a tiny
# rasterization on the GPU. Fails loud if `_C` is None or kernels won't
# build, instead of letting Stage 10 die 100 lines deeper.
gsplat_smoke_test() {
  python - <<'PY'
import sys
import torch
try:
    from gsplat.rendering import rasterization
except Exception as e:
    print(f"[gsplat smoke] import failed: {e}", file=sys.stderr); sys.exit(1)

if not torch.cuda.is_available():
    print("[gsplat smoke] CUDA not available", file=sys.stderr); sys.exit(2)

d = torch.device("cuda")
n = 4
means     = torch.tensor([[0,0,5.],[1,0,5.],[0,1,5.],[-1,0,5.]], device=d)
quats     = torch.zeros(n, 4, device=d); quats[:, 0] = 1.0
scales    = torch.full((n, 3), 0.1, device=d)
opac      = torch.full((n,), 0.5, device=d)
colors    = torch.tensor([[1,0,0.],[0,1,0.],[0,0,1.],[1,1,0.]], device=d)
viewmats  = torch.eye(4, device=d).unsqueeze(0)
Ks        = torch.tensor([[[100.,0,32.],[0,100.,32.],[0,0,1.]]], device=d)
print("[gsplat smoke] running first call (JIT compile may take ~30-60 s)...",
      flush=True)
r, a, _ = rasterization(means, quats, scales, opac, colors,
                         viewmats, Ks, 64, 64, sh_degree=None)
assert r.shape == (1, 64, 64, 3), r.shape
print(f"[gsplat smoke] OK — rendered {tuple(r.shape)} from {n} Gaussians")
PY
}

# Locate the teammate's lift_to_semantic_pointcloud.py + its sibling
# utils.py. The script lives on the `lift-semantic` branch; if it has
# been merged into main it's in scripts/, otherwise we check the
# unpacked branch-export directories.
find_lift_script() {
  local candidates=(
    "$REPO_DIR/scripts/lift_to_semantic_pointcloud.py"
    "$REPO_DIR/Driving-Scene3R-lift-semantic/Driving-Scene3R-lift-semantic/scripts/lift_to_semantic_pointcloud.py"
    "$REPO_DIR/Driving-Scene3R-lift-semantic/scripts/lift_to_semantic_pointcloud.py"
    "$AFM_ROOT/Driving-Scene3R-lift-semantic/Driving-Scene3R-lift-semantic/scripts/lift_to_semantic_pointcloud.py"
    "$AFM_ROOT/Driving-Scene3R-lift-semantic/scripts/lift_to_semantic_pointcloud.py"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" && -f "$(dirname "$c")/utils.py" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
AFM_ROOT="${AFM_ROOT:-$HOME/afm/x}"
SEQUENCE="04"
N_FRAMES_LIFT=50
BATCH_SIZE=4
DATA_ROOT="/storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences"
POSES_ROOT=""    # empty → auto-detect; KITTI poses live in a SEPARATE archive,
                 # typically …/data_odometry_poses/dataset/poses/<seq>.txt
CONDA_ENV="driving-scene3r"
VENV_DIR="$AFM_ROOT/.venv"
OUT_DIR="$AFM_ROOT/outputs"
MSNET_REPO="$AFM_ROOT/mobilestereonet"
MSNET_CKPT="$AFM_ROOT/checkpoints/MSNet2D_SF_KITTI2015.ckpt"
HF_HOME_DIR="$AFM_ROOT/hf_cache"
SKIP_SETUP=0
SKIP_GEN=0
SKIP_TRAIN=0
FORCE_VENV=0    # if 1, ignore conda even when present

# Phase 3 (gsplat training) hyper-params -------------------------------------
TRAIN_MAX_ITERS=10000
TRAIN_EVAL_EVERY=1000
TRAIN_EVAL_STRIDE=10
TRAIN_MAX_FRAMES=0    # 0 = use all loader frames
TRAIN_SKY_DOME=1     # 1 = add synthetic sky dome + ground shell to the export

usage() {
  cat <<EOF
Usage: $0 [options]

End-to-end driver: generates teammate inputs (steps 4-6) and runs YOUR
semantic_gs code (steps 7-10) against them on a real KITTI sequence.

Options:
  -s, --sequence SEQ      KITTI sequence (default: $SEQUENCE; valid: 00-10)
  -n, --n-frames N        Frames to aggregate in lift script (default: $N_FRAMES_LIFT)
  -b, --batch-size N      GPU batch size for M2F / MSNet (default: $BATCH_SIZE)
  -o, --out-dir DIR       Output root (default: $OUT_DIR)
      --env NAME          Conda env name (default: $CONDA_ENV)
      --venv-dir DIR      venv path (used when conda is absent, default: $VENV_DIR)
      --force-venv        Always use venv, even if conda is available
      --msnet-repo DIR    MobileStereoNet repo (default: $MSNET_REPO)
      --msnet-ckpt FILE   MSNet2D checkpoint (default: $MSNET_CKPT)
      --data-root DIR     KITTI sequences dir (default: $DATA_ROOT)
      --poses-root DIR    KITTI poses dir (default: auto-detect sibling
                          data_odometry_poses/dataset/poses). The lift
                          script accepts either layout
                          <root>/<seq>.txt or <root>/<seq>/poses.txt.
      --skip-setup        Skip Python env + dependency install (stage 1)
      --skip-gen          Skip teammate-script stages (4-6), just test your code
      --skip-train        Skip Phase-3 GS training (stage 10) — quick re-runs
      --only-test         Same as --skip-setup --skip-gen
      --max-iters N       Phase-3 training iterations (default: $TRAIN_MAX_ITERS)
      --eval-every N      Phase-3 eval/render interval (default: $TRAIN_EVAL_EVERY)
      --eval-stride N     Every Nth loader frame held out for eval (default: $TRAIN_EVAL_STRIDE)
      --max-train-frames N  Cap loader to N frames for faster POC (0 = all)
      --no-sky-dome       Do NOT add the synthetic sky dome to the export
  -h, --help              Show this help

Re-runs are safe: each stage skips work already done.
EOF
}

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--sequence)        SEQUENCE="$2"; shift 2 ;;
    -n|--n-frames)        N_FRAMES_LIFT="$2"; shift 2 ;;
    -b|--batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    -o|--out-dir)         OUT_DIR="$2"; shift 2 ;;
    --env)                CONDA_ENV="$2"; shift 2 ;;
    --venv-dir)           VENV_DIR="$2"; shift 2 ;;
    --force-venv)         FORCE_VENV=1; shift ;;
    --msnet-repo)         MSNET_REPO="$2"; shift 2 ;;
    --msnet-ckpt)         MSNET_CKPT="$2"; shift 2 ;;
    --data-root)          DATA_ROOT="$2"; shift 2 ;;
    --poses-root)         POSES_ROOT="$2"; shift 2 ;;
    --skip-setup)         SKIP_SETUP=1; shift ;;
    --skip-gen)           SKIP_GEN=1; shift ;;
    --skip-train)         SKIP_TRAIN=1; shift ;;
    --only-test)          SKIP_SETUP=1; SKIP_GEN=1; shift ;;
    --max-iters)          TRAIN_MAX_ITERS="$2"; shift 2 ;;
    --eval-every)         TRAIN_EVAL_EVERY="$2"; shift 2 ;;
    --eval-stride)        TRAIN_EVAL_STRIDE="$2"; shift 2 ;;
    --max-train-frames)   TRAIN_MAX_FRAMES="$2"; shift 2 ;;
    --no-sky-dome)        TRAIN_SKY_DOME=0; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# Derived paths
# -----------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_DIR="$OUT_DIR/depth_predictions"
PANO_DIR="$OUT_DIR/panoptic_predictions"
CLOUDS_DIR="$OUT_DIR/semantic_clouds"
CLOUD_PLY="$CLOUDS_DIR/seq${SEQUENCE}_static.ply"
CLOUD_NPZ="$CLOUDS_DIR/seq${SEQUENCE}_static.npz"
SMOKE_OUT="$OUT_DIR/semantic_gs_smoke"
PANO_SEQ_DIR="$PANO_DIR/$SEQUENCE"
DEPTH_SEQ_DIR="$DEPTH_DIR/$SEQUENCE"
# Phase-3 outputs (one subdir per sequence so reruns don't clobber)
GS_RUN_DIR="$OUT_DIR/semantic_gs_runs/seq${SEQUENCE}"

# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------
_ts()    { date +"%H:%M:%S"; }
log()    { printf '\033[1;36m[%s]\033[0m \033[1m%s\033[0m\n' "$(_ts)" "$*"; }
warn()   { printf '\033[1;33m[%s] WARN:\033[0m %s\n' "$(_ts)" "$*" >&2; }
err()    { printf '\033[1;31m[%s] ERROR:\033[0m %s\n' "$(_ts)" "$*" >&2; }
header() { printf '\n\033[1;35m============================================================\n  %s\n============================================================\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
header "Pre-flight"
log "Repo dir        : $REPO_DIR"
log "AFM_ROOT        : $AFM_ROOT"
log "Sequence        : $SEQUENCE  (n_frames lift = $N_FRAMES_LIFT, batch = $BATCH_SIZE)"
log "Output dir      : $OUT_DIR"
log "KITTI mirror    : $DATA_ROOT"
log "Conda env       : $CONDA_ENV"
log "MSNet repo      : $MSNET_REPO"
log "MSNet checkpoint: $MSNET_CKPT"

# KITTI mirror sanity
if [[ ! -d "$DATA_ROOT/$SEQUENCE" ]]; then
  err "KITTI sequence not found: $DATA_ROOT/$SEQUENCE"
  err "Are you on the i9 workstation? Is /storage/group/... mounted on this node?"
  exit 2
fi

if [[ ! -f "$DATA_ROOT/$SEQUENCE/calib.txt" ]]; then
  err "No calib.txt at $DATA_ROOT/$SEQUENCE/calib.txt"
  exit 2
fi

# -----------------------------------------------------------------------------
# Locate the REAL KITTI odometry poses file.
#
# Standard KITTI layout puts poses in a SEPARATE archive — they are NOT under
# sequences/<seq>/. The file inside sequences/<seq>/poses.txt (if it exists at
# all) is usually empty or a leftover placeholder. Always prefer the canonical
# locations and verify the chosen file is non-empty.
# -----------------------------------------------------------------------------
resolve_poses_file() {
  local seq="$1"
  local user_root="$2"           # may be empty
  local data_root="$3"
  local cand
  local candidates=()

  if [[ -n "$user_root" ]]; then
    candidates+=( "$user_root/${seq}.txt"
                  "$user_root/poses/${seq}.txt"
                  "$user_root/$seq/poses.txt"
                  "$user_root/$seq/${seq}.txt" )
  fi
  # Auto-detect: assumes data_odometry_color/dataset/sequences layout and
  # tries the sibling data_odometry_poses archive AND the i9-mirror habit of
  # storing the KITTI ground-truth poses inside each sequence as <seq>.txt
  # (alongside an empty / broken sequences/<seq>/poses.txt).
  local color_dataset_root
  color_dataset_root="$(dirname "$data_root")"                  # .../dataset
  local mirror_root
  mirror_root="$(dirname "$(dirname "$color_dataset_root")")"   # .../kitti_odom_color
  candidates+=(
    "$data_root/$seq/${seq}.txt"   # i9 mirror: sequences/<seq>/<seq>.txt  ← seq 01-10
    "$color_dataset_root/poses/${seq}.txt"
    "$mirror_root/data_odometry_poses/dataset/poses/${seq}.txt"
    "$(dirname "$mirror_root")/kitti_odom_grey/data_odometry_poses/dataset/poses/${seq}.txt"
    "$data_root/$seq/poses.txt"    # last-resort: the (likely empty/broken) one
  )

  for cand in "${candidates[@]}"; do
    # -s : exists AND non-empty. Follows symlinks, so broken symlinks
    # (the seq 01-10 poses.txt -> /storage/local/... entries on this mirror)
    # are correctly rejected here.
    if [[ -s "$cand" ]]; then
      # Sanity-check first non-blank line has 12 whitespace-separated floats
      # (KITTI 3x4 row-major pose). Skips OKVIS CSV / timestamp files.
      local first_n
      first_n=$(awk 'NF { print NF; exit }' "$cand" 2>/dev/null || echo 0)
      if [[ "$first_n" == "12" ]]; then
        echo "$cand"
        return 0
      fi
    fi
  done
  # Nothing worked — print what we tried with reason per candidate.
  err "Could not find a valid (12-column, non-empty) KITTI poses file for sequence $seq. Tried:"
  for cand in "${candidates[@]}"; do
    local reason
    if [[ -L "$cand" && ! -e "$cand" ]]; then
      reason="broken symlink"
    elif [[ ! -e "$cand" ]]; then
      reason="missing"
    elif [[ ! -s "$cand" ]]; then
      reason="empty"
    else
      reason="first line has $(awk 'NF{print NF; exit}' "$cand" 2>/dev/null) cols (expected 12)"
    fi
    err "  - $cand   [$reason]"
  done
  if [[ "$seq" == "00" ]]; then
    err ""
    err "NOTE: on the TUM i9 KITTI mirror, sequence 00 is missing the KITTI"
    err "ground-truth pose file (only OKVIS '*_trajectory.csv' files exist in"
    err "sequences/00/). Pick another sequence (e.g. -s 04) or supply"
    err "--poses-root pointing at a directory that contains 00.txt / poses/00.txt."
  fi
  return 1
}

POSES_FILE="$(resolve_poses_file "$SEQUENCE" "$POSES_ROOT" "$DATA_ROOT")" || exit 2
POSES_DIR_FOR_LIFT="$(dirname "$POSES_FILE")"   # lift script needs <root>/<seq>.txt
log "Poses file      : $POSES_FILE ($(wc -l < "$POSES_FILE") lines)"

N_PNGS=$(find "$DATA_ROOT/$SEQUENCE/image_2" -maxdepth 1 -name '*.png' | wc -l)
if (( N_PNGS == 0 )); then
  err "No PNGs found in $DATA_ROOT/$SEQUENCE/image_2/"
  exit 2
fi
log "Frames in seq   : $N_PNGS"

# Workspace — pre-create per-sequence subdirs so later `find` calls always succeed
mkdir -p "$OUT_DIR" "$DEPTH_DIR" "$PANO_DIR" "$CLOUDS_DIR" "$SMOKE_OUT" "$HF_HOME_DIR" \
         "$PANO_SEQ_DIR" "$DEPTH_SEQ_DIR" \
         "$(dirname "$MSNET_CKPT")"

# -----------------------------------------------------------------------------
# Stage 1 : Python environment (conda preferred, venv fallback)
# -----------------------------------------------------------------------------
ENV_BACKEND=""   # set by activate_env: "conda" or "venv"

detect_python_for_venv() {
  # Pick the newest acceptable interpreter on PATH (>= 3.10, < 3.14 to stay
  # compatible with torch wheels and avoid 3.13/3.14 ecosystem gaps).
  local cand
  for cand in python3.12 python3.11 python3.10; do
    if command -v "$cand" &>/dev/null; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

create_or_activate_env() {
  # Conda path
  if [[ $FORCE_VENV -eq 0 ]] && command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      log "Creating conda env '$CONDA_ENV' (python 3.10)..."
      conda create -n "$CONDA_ENV" python=3.10 -y
    else
      log "Conda env '$CONDA_ENV' already exists."
    fi
    conda activate "$CONDA_ENV"
    ENV_BACKEND="conda"
    return 0
  fi

  # venv path
  local py
  if ! py=$(detect_python_for_venv); then
    err "Neither conda nor a python3.10+ interpreter found on PATH."
    cat <<EOF >&2

Install one of:
  (A) Miniforge into your home dir (no sudo, recommended):
        cd \$HOME && curl -L -o mf.sh \\
          https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash mf.sh -b -p \$HOME/afm/x/miniforge3 && rm mf.sh
        \$HOME/afm/x/miniforge3/bin/conda init bash && exec bash
        # then re-run this script.

  (B) Ask the cluster admin to load a conda or python 3.10+ module.

EOF
    exit 2
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv with $py → $VENV_DIR"
    "$py" -m venv "$VENV_DIR"
  else
    log "venv already exists: $VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --quiet --upgrade pip
  ENV_BACKEND="venv"
}

reactivate_env_only() {
  # Used in --skip-setup mode to put us back into the right env.
  if [[ $FORCE_VENV -eq 0 ]] && command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" && { ENV_BACKEND="conda"; return 0; } \
      || warn "could not activate conda env '$CONDA_ENV'"
  fi
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ENV_BACKEND="venv"
    return 0
  fi
  warn "no python env detected; relying on whatever python is on PATH"
}

if [[ $SKIP_SETUP -eq 0 ]]; then
  header "Stage 1 / 10 : Python environment + dependencies"
  create_or_activate_env
  log "Backend: $ENV_BACKEND  |  python: $(command -v python)  ($(python --version 2>&1))"

  log "Installing project requirements + extras (visible output so dep conflicts surface)..."
  pip install --progress-bar off -r "$REPO_DIR/requirements.txt"
  pip install --progress-bar off scipy open3d pytest plyfile
  pip install --progress-bar off -e "$REPO_DIR"

  python - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "n/a"
mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3 if ok else 0
print(f"[gpu] CUDA available: {ok} | device: {name} | total VRAM: {mem:.1f} GB | torch: {torch.__version__}")
PY

  # ---- Phase-3 backend: gsplat (CUDA only — this project targets Linux+NVIDIA) ----
  if [[ $SKIP_TRAIN -eq 0 ]]; then
    # 1) Make sure nvcc is available so gsplat can JIT-compile its kernels.
    ensure_nvcc || {
      warn "Phase 3 (Stage 10) will be skipped because nvcc could not be set up."
      SKIP_TRAIN=1
    }

    if [[ $SKIP_TRAIN -eq 0 ]]; then
      # 2) Install gsplat if not already present. We do NOT trust `import gsplat`
      #    as a "working" signal — gsplat's _C extension can be None even after
      #    a successful import (which is what happened in the previous run.log).
      if ! python -c "import gsplat" 2>/dev/null; then
        log "Installing gsplat..."
        pip install --progress-bar off "gsplat>=1.4,<2.0"
      else
        log "gsplat already installed."
      fi

      # 3) Real end-to-end check: JIT-compile + tiny rasterization. If this
      #    fails, Stage 10 cannot work and we want to know NOW.
      log "Running gsplat smoke test (will JIT-compile CUDA kernels the first time)..."
      if ! gsplat_smoke_test; then
        warn "gsplat smoke test failed — Stage 10 will be skipped."
        warn "  Re-check the install log above. Common causes:"
        warn "    - nvcc + torch CUDA versions don't match"
        warn "    - CUDA driver too old for the installed gsplat wheels"
        warn "    - GPU compute capability not supported by the installed kernels"
        SKIP_TRAIN=1
      fi
    fi
  fi
else
  header "Stage 1 SKIPPED (--skip-setup)"
  reactivate_env_only
fi

# -----------------------------------------------------------------------------
# Stage 2-3 : MobileStereoNet repo + checkpoint
# -----------------------------------------------------------------------------
if [[ $SKIP_GEN -eq 0 ]]; then
  header "Stage 2 / 10 : MobileStereoNet repo"
  if [[ -d "$MSNET_REPO/.git" ]]; then
    log "Already cloned at $MSNET_REPO"
  else
    log "Cloning MobileStereoNet → $MSNET_REPO"
    git clone https://github.com/cogsys-tuebingen/mobilestereonet "$MSNET_REPO"
  fi

  header "Stage 3 / 10 : MobileStereoNet checkpoint"
  if [[ ! -f "$MSNET_CKPT" ]]; then
    err "MSNet2D checkpoint not found at: $MSNET_CKPT"
    cat <<EOF >&2

ACTION REQUIRED (one-time, ~50 MB)
----------------------------------
The checkpoint isn't on Hugging Face, only on Google Drive. Manual steps:

  1. Open  https://github.com/cogsys-tuebingen/mobilestereonet  in a browser.
  2. In the evaluation table, click the hyperlinked model name for the
     MSNet2D row, column "SF + KITTI2015". This opens Google Drive.
  3. Download the .ckpt file.
  4. Save / scp it to exactly this path:
         $MSNET_CKPT
  5. Re-run this script:  $0 $*
     (everything done so far will be skipped automatically)

EOF
    exit 3
  fi
  log "Checkpoint OK: $MSNET_CKPT ($(du -h "$MSNET_CKPT" | cut -f1))"
fi

# -----------------------------------------------------------------------------
# Stage 4 : Mask2Former panoptic
# -----------------------------------------------------------------------------
if [[ $SKIP_GEN -eq 0 ]]; then
  header "Stage 4 / 10 : Mask2Former panoptic inference  (GPU)"
  N_DONE=$(count_npz "$PANO_SEQ_DIR")
  if (( N_DONE >= N_PNGS )); then
    log "All $N_PNGS panoptic NPZs already exist — skipping."
  else
    log "Status: $N_DONE / $N_PNGS already done. Running Mask2Former (batch=$BATCH_SIZE)..."
    log "(weights cached under HF_HOME=$HF_HOME_DIR — ~1 GB the first time)"
    HF_HOME="$HF_HOME_DIR" python "$REPO_DIR/scripts/run_mask2former_inference.py" \
      --dataset_root "$DATA_ROOT" \
      --sequences    "$SEQUENCE" \
      --output_dir   "$PANO_DIR" \
      --batch_size   "$BATCH_SIZE"
  fi
fi

# -----------------------------------------------------------------------------
# Stage 5 : MobileStereoNet depth
# -----------------------------------------------------------------------------
if [[ $SKIP_GEN -eq 0 ]]; then
  header "Stage 5 / 10 : MobileStereoNet depth inference  (GPU)"
  N_DONE=$(count_npz "$DEPTH_SEQ_DIR")
  if (( N_DONE >= N_PNGS )); then
    log "All $N_PNGS depth NPZs already exist — skipping."
  else
    log "Status: $N_DONE / $N_PNGS already done. Running MobileStereoNet (batch=$BATCH_SIZE)..."
    python "$REPO_DIR/scripts/run_mobilestereonet_inference.py" \
      --msnet_path   "$MSNET_REPO" \
      --dataset_root "$DATA_ROOT" \
      --sequences    "$SEQUENCE" \
      --checkpoint   "$MSNET_CKPT" \
      --output_dir   "$DEPTH_DIR" \
      --batch_size   "$BATCH_SIZE"
  fi
fi

# -----------------------------------------------------------------------------
# Stage 6 : lift to static semantic point cloud
# -----------------------------------------------------------------------------
if [[ $SKIP_GEN -eq 0 ]]; then
  header "Stage 6 / 10 : lift to static semantic point cloud  (CPU, ~30 s)"
  if [[ -f "$CLOUD_PLY" && -f "$CLOUD_NPZ" ]]; then
    log "Lift output already exists. Delete to regenerate:"
    log "  $CLOUD_PLY"
    log "  $CLOUD_NPZ"
  else
    LIFT_SCRIPT="$(find_lift_script)" || {
      err "lift_to_semantic_pointcloud.py + utils.py not found in any known location."
      cat <<EOF >&2

ACTION REQUIRED — one of:

  (a) Copy the two files into the main scripts/ dir of this repo:
        cp <unpacked-branch>/scripts/lift_to_semantic_pointcloud.py $REPO_DIR/scripts/
        cp <unpacked-branch>/scripts/utils.py                       $REPO_DIR/scripts/
      then re-run me with --skip-setup --skip-gen=0   (or just --skip-setup).

  (b) Merge the 'lift-semantic' branch into your working branch so the
      files live at scripts/ permanently:
        git fetch origin
        git checkout -b merged-lift-semantic
        git merge origin/lift-semantic       # resolve any conflicts
      then re-run me.

EOF
      exit 4
    }
    log "Using lift script:  $LIFT_SCRIPT"
    log "Aggregating $N_FRAMES_LIFT frames → $CLOUD_PLY (+ .npz)"
    python "$LIFT_SCRIPT" \
      --dataset_root "$DATA_ROOT" \
      --sequence     "$SEQUENCE" \
      --poses_root   "$POSES_DIR_FOR_LIFT" \
      --depth_dir    "$DEPTH_DIR" \
      --panoptic_dir "$PANO_DIR" \
      --output       "$CLOUD_PLY" \
      --n_frames     "$N_FRAMES_LIFT" \
      --aggregation  voxel \
      --voxel_size   0.10 \
      --color_mode   rgb \
      --depth_trunc  35.0 \
      --save_npz
  fi
fi

# =============================================================================
#   TESTING YOUR CODE (Phases 0, 1, 2 of semantic_gs)
# =============================================================================

header "Stage 7 / 10 : Phase 0 — pytest (YOUR hermetic test suite)"
( cd "$REPO_DIR" && python -m pytest -q )

header "Stage 8 / 10 : Phase 1 — YOUR KITTI loader against REAL teammate output"
if [[ ! -d "$PANO_SEQ_DIR" || ! -d "$DEPTH_SEQ_DIR" ]]; then
  warn "Teammate per-frame outputs missing — skipping Phase 1 real-data test."
  warn "Run without --skip-gen to generate them first."
else
  python -m semantic_gs.scripts.smoke_test_adapters \
    --kitti-odom-seq "$DATA_ROOT/$SEQUENCE" \
    --depth-dir      "$DEPTH_SEQ_DIR" \
    --pano-dir       "$PANO_SEQ_DIR" \
    --pose-path      "$POSES_FILE" \
    --frames 0 50 100 200 \
    --out            "$SMOKE_OUT"
fi

header "Stage 9 / 10 : Phase 2 — YOUR point-cloud adapter on REAL lift output"
if [[ ! -f "$CLOUD_NPZ" ]]; then
  warn "Lift NPZ missing at $CLOUD_NPZ — skipping Phase 2 real-data test."
else
  python -m semantic_gs.scripts.init_pointcloud \
    --input "$CLOUD_NPZ" \
    --out   "$SMOKE_OUT"
fi

# -----------------------------------------------------------------------------
# Stage 10 : Phase 3 — vanilla 3DGS training on real KITTI
# -----------------------------------------------------------------------------
header "Stage 10 / 10 : Phase 3 — YOUR 3DGS trainer on REAL KITTI (GPU)"
if [[ $SKIP_TRAIN -eq 1 ]]; then
  warn "Phase 3 training SKIPPED (--skip-train, no CUDA, or gsplat install failed)."
elif [[ ! -f "$CLOUD_NPZ" ]]; then
  warn "Lift NPZ missing at $CLOUD_NPZ — skipping Phase 3 training."
elif ! python -c "import gsplat" 2>/dev/null; then
  warn "gsplat not importable — skipping Phase 3 training."
  warn "  Re-run without --skip-setup to retry the install, or install manually:"
  warn "    pip install 'gsplat>=1.4,<2.0'"
else
  mkdir -p "$GS_RUN_DIR"
  log "Training vanilla 3DGS (max_iters=$TRAIN_MAX_ITERS, eval_every=$TRAIN_EVAL_EVERY)"
  log "Output dir: $GS_RUN_DIR"
  EXTRA_ARGS=()
  if (( TRAIN_MAX_FRAMES > 0 )); then
    EXTRA_ARGS+=( --max-frames "$TRAIN_MAX_FRAMES" )
  fi
  if (( TRAIN_SKY_DOME == 1 )); then
    EXTRA_ARGS+=( --sky-dome --sky-dome-ground )
  fi
  python -m semantic_gs.scripts.train_gs \
    --kitti-odom-seq "$DATA_ROOT/$SEQUENCE" \
    --depth-dir      "$DEPTH_SEQ_DIR" \
    --pano-dir       "$PANO_SEQ_DIR" \
    --pose-path      "$POSES_FILE" \
    --init-pc        "$CLOUD_NPZ" \
    --out            "$GS_RUN_DIR" \
    --max-iters      "$TRAIN_MAX_ITERS" \
    --eval-every     "$TRAIN_EVAL_EVERY" \
    --eval-stride    "$TRAIN_EVAL_STRIDE" \
    "${EXTRA_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
header "ALL DONE"
echo "Phase 0/1/2 visualisations  :  $SMOKE_OUT"
if compgen -G "$SMOKE_OUT/*.png" > /dev/null; then
  ls -lh "$SMOKE_OUT"/*.png
else
  echo "  (no PNGs yet — check the warnings above)"
fi
echo
if [[ -d "$GS_RUN_DIR" && -f "$GS_RUN_DIR/summary.json" ]]; then
  echo "Phase 3 training outputs    :  $GS_RUN_DIR"
  echo "  3DGS PLY (open in SuperSplat: https://playcanvas.com/supersplat/editor):"
  ls -lh "$GS_RUN_DIR"/exports/*.ply 2>/dev/null || true
  echo "  Eval render PNGs:"
  ls -lh "$GS_RUN_DIR"/renders/*.png 2>/dev/null | tail -n 6 || true
  echo "  Final metrics summary:"
  python - <<PY
import json, pathlib
p = pathlib.Path("$GS_RUN_DIR") / "summary.json"
d = json.loads(p.read_text())
print(f"    N gaussians         : {d['n_gaussians']:,}")
print(f"    Final eval PSNR mean: {d['final_eval_psnr_mean']:.2f} dB")
print(f"    Final eval SSIM mean: {d['final_eval_ssim_mean']:.3f}")
PY
  echo
fi
echo "To copy outputs to your Windows machine for viewing:"
echo "  scp '$USER@$(hostname):${SMOKE_OUT}/*.png' ."
if [[ -d "$GS_RUN_DIR" ]]; then
  echo "  scp '$USER@$(hostname):${GS_RUN_DIR}/exports/gaussians_final.ply' ."
  echo "  scp '$USER@$(hostname):${GS_RUN_DIR}/renders/*.png' ."
fi
echo
echo "Re-run just the tests later with:  $0 --only-test"
echo "Skip training (quick rerun)    :  $0 --only-test --skip-train"
