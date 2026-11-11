#!/bin/bash
# Two-minute check, run once after scripts/hpc/setup_qe_env.sh: do Quantum
# ESPRESSO programs from the conda environment start under Slurm, and which
# launcher gives real parallel runs?
#
# A conda-installed MPI does not always talk to a cluster's srun. When it does
# not, srun starts N separate one-process copies instead of one N-process run.
# The job would still "work", only N times slower and with outputs overwriting
# each other, so check before Stage 0.
#
# Submit from the work area, with the environment's bin directory on PATH:
#   export PATH=/usr/local/usrapps/<group>/<unity_id>/envs/rtsc-qe/bin:$PATH
#   sbatch scripts/hpc/check_qe_mpi.sbatch
#
# Read check_qe_mpi.<jobid>.out. Use the launcher whose line reads
# "1 x Number of MPI processes: 2" for Stage 0. "2 x Number of MPI processes: 1"
# is the failure described above: two separate one-process copies.
#   srun works     -> scripts/hpc/submit_stage0_pb.sh ... as written
#   only mpirun    -> LAUNCHER="mpirun -np 8" scripts/hpc/submit_stage0_pb.sh ...
#   neither        -> send the output file to HPC support (help@ncsu.edu)
#
#SBATCH --job-name=check_qe_mpi
#SBATCH --output=check_qe_mpi.%j.out
#SBATCH --error=check_qe_mpi.%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "host=$(hostname) date=$(date -Is) ntasks=${SLURM_NTASKS:-unset}"
for exe in pw.x ph.x epw.x wannier90.x; do
  printf '%-14s %s\n' "$exe" "$(command -v "$exe" || echo NOT-ON-PATH)"
done
command -v pw.x >/dev/null || { echo "pw.x not on PATH; export PATH as described above" >&2; exit 2; }

# pw.x prints how many MPI processes it sees before it reads any input, so an
# empty input is enough. It then stops with an input error, which is expected.
report() {  # report <label> <launcher...>
  local label="$1"; shift
  local seen
  seen="$("$@" pw.x -in /dev/null 2>&1 | grep 'Number of MPI processes' | tr -s ' ' | sort | uniq -c \
          | awk '{count=$1; $1=""; printf "%s x%s; ", count, $0}' || true)"
  printf '%-8s %s\n' "$label" "${seen:-no MPI line printed (launcher missing or failed)}"
}
export OMP_NUM_THREADS=1
report srun   srun -n 2
report mpirun mpirun -np 2
echo "done"
