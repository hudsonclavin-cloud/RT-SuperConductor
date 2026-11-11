#!/usr/bin/env bash
# Stage 1 on Hazel: the tantalum benchmark (JARVIS JVASP-1014) as a chain of
# Slurm jobs. Run it only after Stage 0 has passed.
#
# Usage:
#   scripts/hpc/submit_stage1_ta.sh <run_dir> <partition> <qos> [ntasks]
#   e.g. scripts/hpc/submit_stage1_ta.sh /share/<group>/<unity_id>/rtsc/ta-jvasp-1014 compute normal 16
#
# Run it twice:
#   1. as written (smoke mode): coarse meshes, a setup check only;
#   2. with CHAIN_MODE=full in front: the benchmark settings in
#      data/benchmarks/epw-ta-jvasp-1014/. Only this result is compared with
#      the reference (lambda = 0.7572, omega_log = 164.99 K).
#
# This is a thin wrapper around scripts/hpc/submit_epw_chain.sh; every switch
# described there (LAUNCHER, START_AT, PH_TIME, EPW_TIME, LOCAL, DRY_RUN) works
# here too. When the last job finishes, copy <run_dir>/epw/epw.out and
# ta-bcc.cif into results/ta-jvasp-1014/, write review.json from the package's
# review.template.json, and run
#   python scripts/promote_dft_results.py --results-root results
set -euo pipefail
: "${1:?usage: submit_stage1_ta.sh <run_dir> <partition> <qos> [ntasks]}"
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/submit_epw_chain.sh" "$here/../../data/benchmarks/epw-ta-jvasp-1014" "$@"
