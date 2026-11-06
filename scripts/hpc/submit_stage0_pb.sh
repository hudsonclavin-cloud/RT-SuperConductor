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
# Prints the job IDs. When the last job finishes, copy
#   <run_dir>/woSOC/epw/epw.out, the CIF and logs/ into results/pb-tutorial/
# in the repository, write results/pb-tutorial/review.json from
# data/benchmarks/epw-pb-tutorial/review.template.json, and run
#   python scripts/promote_dft_results.py --results-root results
# Stage 0 passes when that report shows a tools_check with lambda, omega_log
# and an EPW consistency check of true.
set -euo pipefail

run_dir="${1:?usage: submit_stage0_pb.sh <run_dir> <partition> <qos> [ntasks]}"
partition="${2:?partition required (check Hazel docs or the HPC chatbot)}"
qos="${3:?qos required}"
ntasks="${4:-8}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
package="$repo_root/data/benchmarks/epw-pb-tutorial"
dry_run="${DRY_RUN:-0}"   # DRY_RUN=1 prints sbatch commands instead of submitting

case "$run_dir" in
  "$HOME"|"$HOME"/*) echo "refusing: run_dir is under \$HOME; use the work area" >&2; exit 3 ;;
esac
for exe in pw.x ph.x epw.x python3; do
  command -v "$exe" >/dev/null || [ "$dry_run" = 1 ] || { echo "$exe not on PATH; activate the QE/EPW environment first" >&2; exit 2; }
done

mkdir -p "$run_dir"
cp -r "$package/pp" "$package/woSOC" "$package/pb-fcc.cif" "$run_dir/"
mkdir -p "$run_dir/logs"
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/woSOC/phonons" scf ph
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/woSOC/epw" scf nscf epw

previous=""
submit() {  # submit <name> <subdir> <ntasks> <time> <command>
  local name="$1" subdir="$2" n="$3" time="$4" cmd="$5"
  local dep=()
  [ -n "$previous" ] && dep=(--dependency="afterok:$previous")
  local args=(--parsable --job-name="rtsc-pb-$name" --partition="$partition" --qos="$qos"
              --nodes=1 --ntasks="$n" --time="$time" --chdir="$run_dir/woSOC/$subdir"
              --output="$run_dir/logs/%x.%j.out" --error="$run_dir/logs/%x.%j.err" "${dep[@]}")
  if [ "$dry_run" = 1 ]; then
    echo "sbatch ${args[*]} --wrap=\"$cmd\"" >&2
    previous="dry-$name"
  else
    previous="$(sbatch "${args[@]}" --wrap="$cmd")"
  fi
  echo "$name $previous"
}

submit ph-scf  phonons "$ntasks" 00:30:00 "srun pw.x -in scf.in > scf.out"
submit ph      phonons "$ntasks" 01:50:00 "srun ph.x -in ph.in > ph.out"
# pp.py asks for the prefix and the number of irreducible q-points; ph.x writes
# one pb.dynN file per irreducible q-point (pb.dyn0 is the list itself).
submit pp      phonons 1         00:10:00 "nq=\$(ls pb.dyn* | grep -vc 'pb.dyn0\$'); printf 'pb\n%s\n' \"\$nq\" | python3 pp.py > pp.out"
submit epw-scf epw     "$ntasks" 00:30:00 "srun pw.x -in scf.in > scf.out"
submit nscf    epw     "$ntasks" 00:50:00 "srun pw.x -in nscf.in > nscf.out"
submit epw     epw     "$ntasks" 01:50:00 "srun epw.x -npool $ntasks -input epw.in > epw.out"
