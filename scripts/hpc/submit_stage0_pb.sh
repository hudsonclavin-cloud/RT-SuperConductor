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

run_dir="${1:?usage: submit_stage0_pb.sh <run_dir> <partition> <qos> [ntasks]}"
partition="${2:?partition required (check Hazel docs or the HPC chatbot)}"
qos="${3:?qos required}"
ntasks="${4:-8}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
package="$repo_root/data/benchmarks/epw-pb-tutorial"
dry_run="${DRY_RUN:-0}"   # DRY_RUN=1 prints sbatch commands instead of submitting
local_run="${LOCAL:-0}"   # LOCAL=1 runs the steps here, in order, without Slurm
mode="${STAGE0_MODE:-smoke}"
case "$mode" in
  smoke) ph_time="${PH_TIME:-01:50:00}" ;;
  full)  ph_time="${PH_TIME:-24:00:00}" ;;
  *) echo "STAGE0_MODE must be smoke or full, not '$mode'" >&2; exit 2 ;;
esac
if [ "$local_run" = 1 ]; then
  launcher="${LAUNCHER:-mpirun -np $ntasks}"
else
  launcher="${LAUNCHER:-srun}"
fi

case "$run_dir" in
  "$HOME"|"$HOME"/*) echo "refusing: run_dir is under \$HOME; use the work area" >&2; exit 3 ;;
esac
for exe in pw.x ph.x epw.x python3; do
  command -v "$exe" >/dev/null || [ "$dry_run" = 1 ] || { echo "$exe not on PATH; activate the QE/EPW environment first" >&2; exit 2; }
done
# pp.py, which gathers the phonon files for EPW, imports numpy. Without it the
# third job fails after the phonons have already been paid for.
[ "$dry_run" = 1 ] || python3 -c "import numpy" 2>/dev/null \
  || { echo "python3 on PATH has no numpy (pp.py needs it); rerun scripts/hpc/setup_qe_env.sh or install numpy" >&2; exit 2; }

steps=(ph-scf ph pp epw-scf nscf epw)
start_at="${START_AT:-ph-scf}"
case " ${steps[*]} " in
  *" $start_at "*) ;;
  *) echo "START_AT must be one of: ${steps[*]}" >&2; exit 2 ;;
esac

if [ "$start_at" != ph-scf ]; then
  [ -d "$run_dir/woSOC" ] || { echo "START_AT=$start_at needs an existing run directory: $run_dir" >&2; exit 2; }
  echo "resuming at $start_at; inputs in $run_dir are left as they are" >&2
else
mkdir -p "$run_dir"
cp -r "$package/pp" "$package/woSOC" "$package/pb-fcc.cif" "$run_dir/"
mkdir -p "$run_dir/logs"

# Replace one exact piece of text in a copied input, and stop if it is not there
# exactly once, so a changed upstream file cannot be half-edited silently.
swap() {  # swap <file> <old> <new>
  local file="$1" old="$2" new="$3" count
  count="$(grep -cF -- "$old" "$file" || true)"
  [ "$count" = 1 ] || { echo "smoke edit failed: expected one '$old' in $file, found $count" >&2; exit 5; }
  python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
  echo "$file: '$old' -> '$new'" >> "$run_dir/STAGE0_MODE.txt"
}
echo "mode=$mode" > "$run_dir/STAGE0_MODE.txt"
if [ "$mode" = smoke ]; then
  swap "$run_dir/woSOC/phonons/scf.in" "14 14 14 1 1 1" "8 8 8 1 1 1"
  for n in 1 2 3; do
    swap "$run_dir/woSOC/phonons/ph.in" "nq$n=6," "nq$n=3,"
    swap "$run_dir/woSOC/epw/epw.in" "nq$n         = 6" "nq$n         = 3"
    # EPW's last step interpolates onto fine meshes; 20x20x20 for both k and q
    # is 64 million pairs, far more than a tools check needs. Only the q-mesh
    # is coarsened. The fine k-mesh stays at 20x20x20: at 8x8x8 EPW finds no
    # states at the Fermi level with this smearing and lambda divides by zero.
    swap "$run_dir/woSOC/epw/epw.in" "nqf$n         = 20" "nqf$n         = 8"
  done
fi
fi   # end of the fresh-run setup skipped by START_AT
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/woSOC/phonons" scf ph
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/woSOC/epw" scf nscf epw

previous=""
started=0
submit() {  # submit <name> <subdir> <ntasks> <time> <command>
  local name="$1" subdir="$2" n="$3" time="$4" cmd="$5"
  [ "$name" = "$start_at" ] && started=1
  [ "$started" = 1 ] || { echo "$name skipped (START_AT=$start_at)"; return 0; }
  local dep=()
  [ -n "$previous" ] && dep=(--dependency="afterok:$previous")
  local args=(--parsable --job-name="rtsc-pb-$name" --partition="$partition" --qos="$qos"
              --nodes=1 --ntasks="$n" --time="$time" --chdir="$run_dir/woSOC/$subdir"
              --output="$run_dir/logs/%x.%j.out" --error="$run_dir/logs/%x.%j.err" "${dep[@]}")
  # The commands below are written with srun; swap in the chosen launcher.
  # Serial steps (pp) have no srun and run as they are.
  cmd="${cmd//srun /$launcher }"
  if [ "$local_run" = 1 ]; then
    local local_cmd="$cmd"
    if [ "$dry_run" = 1 ]; then
      echo "(cd $run_dir/woSOC/$subdir && $local_cmd)" >&2
    else
      local started=$SECONDS
      ( cd "$run_dir/woSOC/$subdir" && bash -euo pipefail -c "$local_cmd" ) \
        > "$run_dir/logs/rtsc-pb-$name.local.out" 2> "$run_dir/logs/rtsc-pb-$name.local.err" \
        || { echo "$name FAILED; see $run_dir/logs/rtsc-pb-$name.local.err" >&2; exit 4; }
      echo "$name finished in $((SECONDS - started)) s" >&2
    fi
    previous="local-$name"
  elif [ "$dry_run" = 1 ]; then
    echo "sbatch ${args[*]} --wrap=\"$cmd\"" >&2
    previous="dry-$name"
  else
    previous="$(sbatch "${args[@]}" --wrap="$cmd")"
  fi
  echo "$name $previous"
}

submit ph-scf  phonons "$ntasks" 00:30:00 "srun pw.x -in scf.in > scf.out"
submit ph      phonons "$ntasks" "$ph_time" "srun ph.x -in ph.in > ph.out"
# pp.py asks for the prefix and the number of irreducible q-points; ph.x writes
# one pb.dynN file per irreducible q-point (pb.dyn0 is the list itself).
submit pp      phonons 1         00:10:00 "nq=\$(ls pb.dyn* | grep -vc 'pb.dyn0\$'); printf 'pb\n%s\n' \"\$nq\" | python3 pp.py > pp.out"
submit epw-scf epw     "$ntasks" 00:30:00 "srun pw.x -in scf.in > scf.out"
submit nscf    epw     "$ntasks" 00:50:00 "srun pw.x -in nscf.in > nscf.out"
submit epw     epw     "$ntasks" 01:50:00 "srun epw.x -npool $ntasks -input epw.in > epw.out"
