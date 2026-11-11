#!/usr/bin/env bash
# Run one QE/EPW benchmark package as a chain of six Slurm jobs:
#   phonon ground state -> phonons -> gather phonon files -> EPW ground state
#   -> band step -> EPW.
# scripts/hpc/submit_stage0_pb.sh (lead, Stage 0) and
# scripts/hpc/submit_stage1_ta.sh (tantalum, Stage 1) call this script with
# their package; a new material needs only a new package directory.
#
# Run on the LOGIN node, with the QE/EPW environment's bin directory on PATH
# (Slurm passes the submitting shell's environment, including PATH, to jobs).
#
# Usage:
#   scripts/hpc/submit_epw_chain.sh <package_dir> <run_dir> <partition> <qos> [ntasks]
#
# A package directory holds the inputs plus two small files:
#   chain.conf        shell variables: CHAIN_NAME (job-name tag), PREFIX (the QE
#                     prefix), PHONON_DIR and EPW_DIR (relative to the package),
#                     COPY_ITEMS (what to copy into the run directory), and
#                     optionally MODE_FILE, FULL_PH_TIME, FULL_EPW_TIME.
#   smoke-edits.tsv   one edit per line, tab-separated: file, text to find,
#                     replacement. Applied to the copies in smoke mode only.
#
# Two sizes (CHAIN_MODE; STAGE0_MODE is accepted as an older name):
#   smoke (default)  the package's inputs with the coarser meshes listed in its
#                    smoke-edits.tsv. Runs every program in the chain quickly.
#                    It checks the tools and the setup, not the physics.
#   full             the package's inputs unchanged.
# The package files are never edited; smoke mode changes the copies in
# <run_dir> and records the changes in <run_dir>/<MODE_FILE>.
#
# LAUNCHER replaces srun inside the jobs, for example LAUNCHER="mpirun -np 8"
# when scripts/hpc/check_qe_mpi.sbatch shows that srun does not start a
# conda-installed MPI program correctly. PH_TIME and EPW_TIME override the
# phonon and EPW jobs' time limits.
#
# START_AT=<step> resumes an existing run directory at that step (ph-scf, ph,
# pp, epw-scf, nscf or epw) without recopying inputs, so a failed late step
# does not cost the phonons again.
#
# Without a scheduler (a personal machine, or a rehearsal before a Hazel run),
# the same six steps run one after another with mpirun instead of sbatch/srun:
#   LOCAL=1 scripts/hpc/submit_epw_chain.sh <package_dir> <run_dir> - - 4
# There LAUNCHER defaults to "mpirun -np <ntasks>". Never use LOCAL=1 on a Hazel
# login node: calculations there break the Acceptable Use Policy.
#
# Prints the job IDs. When the last job finishes, copy <run_dir>/<EPW_DIR>/epw.out,
# the package's CIF and logs/ into results/<id>/ in the repository, write
# results/<id>/review.json from the package's review.template.json, and run
#   python scripts/promote_dft_results.py --results-root results
set -euo pipefail

usage="usage: submit_epw_chain.sh <package_dir> <run_dir> <partition> <qos> [ntasks]"
package="${1:?$usage}"
run_dir="${2:?$usage}"
partition="${3:?partition required (see docs/hazel-runbook.md, section 3)}"
qos="${4:?qos required}"
ntasks="${5:-8}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$package/chain.conf" ] || { echo "not a benchmark package (no chain.conf): $package" >&2; exit 2; }
package="$(cd "$package" && pwd)"
MODE_FILE="CHAIN_MODE.txt"; FULL_PH_TIME="24:00:00"; FULL_EPW_TIME="04:00:00"
# shellcheck source=/dev/null
. "$package/chain.conf"
: "${CHAIN_NAME:?chain.conf must set CHAIN_NAME}" "${PREFIX:?chain.conf must set PREFIX}"
: "${PHONON_DIR:?chain.conf must set PHONON_DIR}" "${EPW_DIR:?chain.conf must set EPW_DIR}"
: "${COPY_ITEMS:?chain.conf must set COPY_ITEMS}"
dry_run="${DRY_RUN:-0}"   # DRY_RUN=1 prints sbatch commands instead of submitting
local_run="${LOCAL:-0}"   # LOCAL=1 runs the steps here, in order, without Slurm
mode="${CHAIN_MODE:-${STAGE0_MODE:-smoke}}"
case "$mode" in
  smoke) ph_time="${PH_TIME:-01:50:00}"; epw_time="${EPW_TIME:-01:50:00}" ;;
  full)  ph_time="${PH_TIME:-$FULL_PH_TIME}"; epw_time="${EPW_TIME:-$FULL_EPW_TIME}" ;;
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
  [ -d "$run_dir/$EPW_DIR" ] || { echo "START_AT=$start_at needs an existing run directory: $run_dir" >&2; exit 2; }
  echo "resuming at $start_at; inputs in $run_dir are left as they are" >&2
else
mkdir -p "$run_dir"
for item in $COPY_ITEMS; do
  cp -r "$package/$item" "$run_dir/"
done
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
  echo "$file: '$old' -> '$new'" >> "$run_dir/$MODE_FILE"
}
echo "mode=$mode" > "$run_dir/$MODE_FILE"
if [ "$mode" = smoke ]; then
  [ -f "$package/smoke-edits.tsv" ] || { echo "package has no smoke-edits.tsv; use CHAIN_MODE=full" >&2; exit 2; }
  while IFS=$'\t' read -r edit_file edit_old edit_new; do
    case "$edit_file" in ""|"#"*) continue ;; esac
    swap "$run_dir/$edit_file" "$edit_old" "$edit_new"
  done < "$package/smoke-edits.tsv"
fi
fi   # end of the fresh-run setup skipped by START_AT
# EPW needs the band step run with symmetry on. With nosym = .true. the saved
# file records one symmetry operation, EPW then finds 48 for itself, and
# epw.x dies with "free(): invalid pointer" and no explanation (seen with QE
# 7.6 on the first tantalum rehearsal).
if grep -Eiq "^[[:space:]]*nosym[[:space:]]*=[[:space:]]*\.true\." "$run_dir/$EPW_DIR/nscf.in"; then
  echo "refusing: $EPW_DIR/nscf.in sets nosym = .true.; EPW crashes on such a band step. Remove that line." >&2
  exit 6
fi
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/$PHONON_DIR" scf ph
"$repo_root/scripts/hpc/stage_inputs.sh" "$run_dir/$EPW_DIR" scf nscf epw

previous=""
started=0
submit() {  # submit <name> <subdir> <ntasks> <time> <command>
  local name="$1" subdir="$2" n="$3" time="$4" cmd="$5"
  [ "$name" = "$start_at" ] && started=1
  [ "$started" = 1 ] || { echo "$name skipped (START_AT=$start_at)"; return 0; }
  local dep=()
  [ -n "$previous" ] && dep=(--dependency="afterok:$previous")
  local args=(--parsable --job-name="rtsc-$CHAIN_NAME-$name" --partition="$partition" --qos="$qos"
              --nodes=1 --ntasks="$n" --time="$time" --chdir="$run_dir/$subdir"
              --output="$run_dir/logs/%x.%j.out" --error="$run_dir/logs/%x.%j.err" "${dep[@]}")
  # The commands below are written with srun; swap in the chosen launcher.
  # Serial steps (pp) have no srun and run as they are.
  cmd="${cmd//srun /$launcher }"
  if [ "$local_run" = 1 ]; then
    local local_cmd="$cmd"
    if [ "$dry_run" = 1 ]; then
      echo "(cd $run_dir/$subdir && $local_cmd)" >&2
    else
      local started=$SECONDS
      ( cd "$run_dir/$subdir" && bash -euo pipefail -c "$local_cmd" ) \
        > "$run_dir/logs/rtsc-$CHAIN_NAME-$name.local.out" 2> "$run_dir/logs/rtsc-$CHAIN_NAME-$name.local.err" \
        || { echo "$name FAILED; see $run_dir/logs/rtsc-$CHAIN_NAME-$name.local.err" >&2; exit 4; }
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

submit ph-scf  "$PHONON_DIR" "$ntasks" 00:30:00 "srun pw.x -in scf.in > scf.out"
submit ph      "$PHONON_DIR" "$ntasks" "$ph_time" "srun ph.x -in ph.in > ph.out"
# pp.py asks for the prefix and the number of irreducible q-points; ph.x writes
# one <prefix>.dynN file per irreducible q-point (<prefix>.dyn0 is the list itself).
submit pp      "$PHONON_DIR" 1         00:10:00 "nq=\$(ls $PREFIX.dyn* | grep -vc '$PREFIX.dyn0\$'); printf '$PREFIX\n%s\n' \"\$nq\" | python3 pp.py > pp.out"
submit epw-scf "$EPW_DIR"    "$ntasks" 00:30:00 "srun pw.x -in scf.in > scf.out"
submit nscf    "$EPW_DIR"    "$ntasks" 00:50:00 "srun pw.x -in nscf.in > nscf.out"
submit epw     "$EPW_DIR"    "$ntasks" "$epw_time" "srun epw.x -npool $ntasks -input epw.in > epw.out"
