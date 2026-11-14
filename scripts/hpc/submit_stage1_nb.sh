#!/usr/bin/env bash
# Stage 1, second benchmark: niobium as a chain of Slurm jobs, after tantalum.
#
# Usage:
#   scripts/hpc/submit_stage1_nb.sh <run_dir> <partition> <qos> [ntasks]
#
# Run it as written first (smoke mode: coarse meshes, a setup check only), then
# with CHAIN_MODE=full in front: the settings in data/benchmarks/epw-nb-bcc/.
# Judge only the full result, by the rule in docs/hazel-runbook.md, section 7
# ("Stage 1 pass rule"): reference lambda = 1.248, omega_log = 147.1 K.
#
# A thin wrapper around scripts/hpc/submit_epw_chain.sh; every switch described
# there (LAUNCHER, START_AT, PH_TIME, EPW_TIME, LOCAL, DRY_RUN) works here too.
set -euo pipefail
: "${1:?usage: submit_stage1_nb.sh <run_dir> <partition> <qos> [ntasks]}"
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/submit_epw_chain.sh" "$here/../../data/benchmarks/epw-nb-bcc" "$@"
