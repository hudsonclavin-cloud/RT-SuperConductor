#!/usr/bin/env bash
# One-time setup on Hazel's LOGIN node: a conda environment with Quantum
# ESPRESSO, EPW and Wannier90, installed in application space (never $HOME).
#
# Usage:
#   scripts/hpc/setup_qe_env.sh <app_space_dir>
#   e.g. scripts/hpc/setup_qe_env.sh /usr/local/usrapps/<group>/<unity_id>
#
# What it does:
#   1. Refuses to install under $HOME (15 GB and a file-count limit).
#   2. Uses conda, mamba or micromamba if one is already on PATH (on Hazel try
#      `module load conda` first). Otherwise downloads micromamba, a single
#      file, into <app_space_dir>/bin.
#   3. Creates <app_space_dir>/envs/rtsc-qe from conda-forge with the pinned
#      Quantum ESPRESSO version, keeping the package cache in app space too.
#   4. Checks that every program the EPW route needs is in the environment,
#      and numpy, which the EPW example's pp.py imports.
#
# Installing software on a login node is allowed. Running calculations there is
# not, so this script only installs and lists files; it runs no QE program.
#
# The pinned version was checked on 2026-11-11: the conda-forge linux-64
# package qe 7.6 (OpenMPI build) contains epw.x, wannier90.x, pw2wannier90.x,
# ph.x, q2r.x and matdyn.x, and it ran the Stage 0 Pb example end to end.
set -euo pipefail

app_space="${1:?usage: setup_qe_env.sh <app_space_dir>   (e.g. /usr/local/usrapps/<group>/<unity_id>)}"
qe_version="${QE_VERSION:-7.6}"
mpi_variant="${QE_MPI:-openmpi}"          # openmpi or mpich
env_name="${QE_ENV_NAME:-rtsc-qe}"
dry_run="${DRY_RUN:-0}"                   # DRY_RUN=1 prints the commands only
required=(pw.x ph.x q2r.x matdyn.x epw.x wannier90.x pw2wannier90.x)
# micromamba is itself a conda-forge package; this is the file the 2026-11-11 check used.
micromamba_url="${MICROMAMBA_URL:-https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2}"

case "$app_space" in
  "$HOME"|"$HOME"/*) echo "refusing: $app_space is under \$HOME; use application space" >&2; exit 3 ;;
esac

env_dir="$app_space/envs/$env_name"
pkgs_dir="$app_space/conda-pkgs"
run() { if [ "$dry_run" = 1 ]; then echo "+ $*"; else "$@"; fi; }

run mkdir -p "$app_space/bin" "$app_space/envs" "$pkgs_dir"
export CONDA_PKGS_DIRS="$pkgs_dir"        # keep the package cache out of $HOME

tool=""
for candidate in mamba conda micromamba; do
  if command -v "$candidate" >/dev/null 2>&1; then tool="$candidate"; break; fi
done
if [ -z "$tool" ]; then
  tool="$app_space/bin/micromamba"
  if [ ! -x "$tool" ]; then
    echo "no conda found; downloading micromamba to $tool" >&2
    run bash -c "curl -fsSL '$micromamba_url' | tar -xj -C '$app_space' bin/micromamba"
  fi
  export MAMBA_ROOT_PREFIX="$app_space/mamba-root"
fi
echo "using: $tool" >&2

if [ -d "$env_dir/bin" ]; then
  echo "environment already exists: $env_dir (delete it to reinstall)" >&2
else
  run "$tool" create -y -p "$env_dir" -c conda-forge --override-channels \
    "qe=$qe_version=mpi_${mpi_variant}_*" python=3.12 numpy
fi

missing=0
for exe in "${required[@]}"; do
  if [ -x "$env_dir/bin/$exe" ]; then
    echo "ok       $exe"
  elif [ "$dry_run" = 1 ]; then
    echo "check    $exe"
  else
    echo "MISSING  $exe"; missing=1
  fi
done
[ "$missing" = 0 ] || { echo "the environment lacks programs the EPW route needs; ask HPC support" >&2; exit 4; }
# The EPW example's pp.py imports numpy; an environment made before this was
# added to the list above fails here instead of in the third Slurm job.
if [ "$dry_run" != 1 ]; then
  "$env_dir/bin/python3" -c "import numpy" 2>/dev/null \
    || { echo "numpy is missing from $env_dir; add it with: $tool install -y -p $env_dir -c conda-forge numpy" >&2; exit 4; }
  echo "ok       numpy (for pp.py)"
fi

cat <<EOF

Environment: $env_dir
To use it in a shell or a job script, put its programs first on PATH:

  export PATH="$env_dir/bin:\$PATH"

Next: submit scripts/hpc/check_qe_mpi.sbatch to confirm how MPI programs start
on Hazel before running Stage 0 (docs/hazel-runbook.md, section 5).
EOF
