#!/usr/bin/env bash
# Stage 0 on Hazel: run the official EPW Pb example as a chain of Slurm jobs.
#
# Run on the LOGIN node, with the QE/EPW conda environment already activated
# (Slurm passes the submitting shell's environment, including PATH, to jobs).
#
# Usage:
#   scripts/hpc/submit_stage0_pb.sh <run_dir> <partition> <qos> [ntasks]
#   e.g. scripts/hpc/submit_stage0_pb.sh /share/<group>/<unity_id>/rtsc/pb-tutorial <partition> <qos> 8
#
# This is a thin wrapper around scripts/hpc/submit_epw_chain.sh, which holds the
# logic and is shared with the tantalum benchmark (submit_stage1_ta.sh).
#
# Two sizes (STAGE0_MODE):
#   smoke (default)  the same inputs with coarser meshes: phonon k-mesh 8x8x8
#                    instead of 14x14x14, q-mesh 3x3x3 instead of 6x6x6, and
#                    EPW fine q-mesh 8x8x8 instead of 20x20x20. It runs every
#                    program in the chain. Stage 0 checks the tools, not the
#                    physics, so this is all it needs.
#   full             the official example unchanged. Its phonon step is heavy:
#                    16 q-points on a 14x14x14 k-mesh, on the order of 100+
#                    core-hours. Use many cores and the long time limit below.
# The vendored files are never edited; smoke mode changes the copies in
# <run_dir> and records the changes in <run_dir>/STAGE0_MODE.txt.
#
# LAUNCHER replaces srun inside the jobs, for example LAUNCHER="mpirun -np 8"
# when scripts/hpc/check_qe_mpi.sbatch shows that srun does not start a
# conda-installed MPI program correctly. PH_TIME overrides the phonon job's
# time limit.
#
# START_AT=<step> resumes an existing run directory at that step (ph-scf, ph,
# pp, epw-scf, nscf or epw) without recopying inputs, so a failed late step
# does not cost the phonons again.
#
# Without a scheduler (a laptop, or a rehearsal before the first Hazel run), the
# same six steps run one after another with mpirun instead of sbatch/srun:
#   LOCAL=1 scripts/hpc/submit_stage0_pb.sh <run_dir> - - 4
# There LAUNCHER defaults to "mpirun -np <ntasks>". Never use LOCAL=1 on a Hazel
# login node: calculations there break the Acceptable Use Policy.
#
# Prints the job IDs. When the last job finishes, copy
#   <run_dir>/woSOC/epw/epw.out, the CIF and logs/ into results/pb-tutorial/
# in the repository, write results/pb-tutorial/review.json from
# data/benchmarks/epw-pb-tutorial/review.template.json, and run
#   python scripts/promote_dft_results.py --results-root results
# Stage 0 passes when that report shows a tools_check with lambda, omega_log
# and an EPW consistency check of true.
set -euo pipefail
: "${1:?usage: submit_stage0_pb.sh <run_dir> <partition> <qos> [ntasks]}"
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/submit_epw_chain.sh" "$here/../../data/benchmarks/epw-pb-tutorial" "$@"
