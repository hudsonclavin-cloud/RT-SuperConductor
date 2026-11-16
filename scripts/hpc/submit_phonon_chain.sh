#!/usr/bin/env bash
# Run one candidate's phonon and electron-phonon package as a chain of six Slurm jobs:
#   dense ground state -> ground state -> phonons with electron-phonon coupling
#   -> force constants -> a2F and lambda -> report.
# The method is Quantum ESPRESSO's interpolated electron-phonon coupling (ph.x,
# q2r.x, matdyn.x). It uses no Wannier functions. Packages are made by
# scripts/prepare_phonon_candidate.py and live in data/benchmarks/candidates/.
#
# Run on the LOGIN node, with the QE environment's bin directory on PATH.
#
# Usage:
#   scripts/hpc/submit_phonon_chain.sh <package_dir> <run_dir> <partition> <qos> [ntasks]
#
# CHAIN_MODE=smoke (default) applies the package's smoke-edits.tsv to the copies
# in <run_dir>: coarse meshes that run every program quickly and check the setup,
# not the physics. CHAIN_MODE=full runs the package unchanged.
#
# The other switches work as in submit_epw_chain.sh: DRY_RUN=1 prints the sbatch
# commands; LAUNCHER="mpirun -np 8" replaces srun; PH_TIME overrides the phonon
# job's time limit; START_AT=<step> resumes an existing run directory at
# scf-fit, scf, ph, q2r, matdyn or report; LOCAL=1 runs the steps in order with
# mpirun and no scheduler (a personal machine only, never a Hazel login node).
#
# ph.in sets recover = .true., so a phonon job that hits its time limit can be
# resubmitted with START_AT=ph and continues where it stopped.
set -euo pipefail

usage="usage: submit_phonon_chain.sh <package_dir> <run_dir> <partition> <qos> [ntasks]"
package="${1:?$usage}"
run_dir="${2:?$usage}"
partition="${3:?partition required (see docs/hazel-runbook.md, section 3)}"
qos="${4:?qos required}"
ntasks="${5:-16}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$package/chain.conf" ] || { echo "not a candidate package (no chain.conf): $package" >&2; exit 2; }
package="$(cd "$package" && pwd)"
FULL_PH_TIME="24:00:00"
# shellcheck source=/dev/null
. "$package/chain.conf"
: "${CHAIN_NAME:?chain.conf must set CHAIN_NAME}" "${PREFIX:?chain.conf must set PREFIX}"
: "${RUN_DIR:?chain.conf must set RUN_DIR}" "${COPY_ITEMS:?chain.conf must set COPY_ITEMS}"
dry_run="${DRY_RUN:-0}"
local_run="${LOCAL:-0}"
mode="${CHAIN_MODE:-smoke}"
case "$mode" in
  smoke) ph_time="${PH_TIME:-01:50:00}" ;;
  full)  ph_time="${PH_TIME:-$FULL_PH_TIME}" ;;
  *) echo "CHAIN_MODE must be smoke or full, not '$mode'" >&2; exit 2 ;;
esac
if [ "$local_run" = 1 ]; then
  launcher="${LAUNCHER:-mpirun -np $ntasks}"
else
  launcher="${LAUNCHER:-srun}"
fi

case "$run_dir" in
  "$HOME"|"$HOME"/*) echo "refusing: run_dir is under \$HOME; use the work area" >&2; exit 3 ;;
esac
for exe in pw.x ph.x q2r.x matdyn.x python3; do
  command -v "$exe" >/dev/null || [ "$dry_run" = 1 ] || { echo "$exe not on PATH; activate the QE environment first" >&2; exit 2; }
done

steps=(scf-fit scf ph q2r matdyn report)
start_at="${START_AT:-scf-fit}"
case " ${steps[*]} " in
  *" $start_at "*) ;;
  *) echo "START_AT must be one of: ${steps[*]}" >&2; exit 2 ;;
esac

if [ "$start_at" != scf-fit ]; then
  [ -d "$run_dir/$RUN_DIR" ] || { echo "START_AT=$start_at needs an existing run directory: $run_dir" >&2; exit 2; }
  echo "resuming at $start_at; inputs in $run_dir are left as they are" >&2
else
  mkdir -p "$run_dir"
  for item in $COPY_ITEMS; do
    cp -r "$package/$item" "$run_dir/"
  done
  cp "$package/package.json" "$run_dir/"
  mkdir -p "$run_dir/logs"
  swap() {  # swap <file> <old> <new>: one exact replacement, or stop
    local file="$1" old="$2" new="$3" count
    count="$(grep -cF -- "$old" "$file" || true)"
    [ "$count" = 1 ] || { echo "smoke edit failed: expected one '$old' in $file, found $count" >&2; exit 5; }
    python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
    echo "$file: '$old' -> '$new'" >> "$run_dir/CHAIN_MODE.txt"
  }
  echo "mode=$mode" > "$run_dir/CHAIN_MODE.txt"
  if [ "$mode" = smoke ]; then
    while IFS=$'\t' read -r edit_file edit_old edit_new; do
      case "$edit_file" in ""|"#"*) continue ;; esac
      swap "$run_dir/$edit_file" "$edit_old" "$edit_new"
    done < "$package/smoke-edits.tsv"
  fi
fi
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/$RUN_DIR" scf_fit scf ph q2r matdyn

previous=""
started=0
submit() {  # submit <name> <ntasks> <time> <command>
  local name="$1" n="$2" time="$3" cmd="$4"
  [ "$name" = "$start_at" ] && started=1
  [ "$started" = 1 ] || { echo "$name skipped (START_AT=$start_at)"; return 0; }
  local dep=()
  [ -n "$previous" ] && dep=(--dependency="afterok:$previous")
  local args=(--parsable --job-name="rtsc-$CHAIN_NAME-$name" --partition="$partition" --qos="$qos"
              --nodes=1 --ntasks="$n" --time="$time" --chdir="$run_dir/$RUN_DIR"
              --output="$run_dir/logs/%x.%j.out" --error="$run_dir/logs/%x.%j.err" "${dep[@]}")
  cmd="${cmd//srun /$launcher }"
  if [ "$local_run" = 1 ]; then
    if [ "$dry_run" = 1 ]; then
      echo "(cd $run_dir/$RUN_DIR && $cmd)" >&2
    else
      local began=$SECONDS
      ( cd "$run_dir/$RUN_DIR" && bash -euo pipefail -c "$cmd" ) \
        > "$run_dir/logs/rtsc-$CHAIN_NAME-$name.local.out" 2> "$run_dir/logs/rtsc-$CHAIN_NAME-$name.local.err" \
        || { echo "$name FAILED; see $run_dir/logs/rtsc-$CHAIN_NAME-$name.local.err" >&2; exit 4; }
      echo "$name finished in $((SECONDS - began)) s" >&2
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

submit scf-fit "$ntasks" 01:50:00 "srun pw.x -in scf_fit.in > scf_fit.out"
submit scf     "$ntasks" 01:00:00 "srun pw.x -in scf.in > scf.out"
submit ph      "$ntasks" "$ph_time" "srun ph.x -in ph.in > ph.out"
submit q2r     1         00:20:00 "q2r.x -in q2r.in > q2r.out"
submit matdyn  1         00:50:00 "matdyn.x -in matdyn.in > matdyn.out"
submit report  1         00:05:00 "python3 $repo_root/scripts/phonon_chain_report.py . > report.json"
