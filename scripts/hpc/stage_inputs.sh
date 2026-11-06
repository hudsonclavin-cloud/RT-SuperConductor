#!/usr/bin/env bash
# Run on the Hazel LOGIN node (or the HPC VCL image) before submitting jobs.
# Compute nodes have no internet, so everything a job reads must already be in
# the run directory. This script checks that, records checksums, and refuses to
# continue if anything is missing. It downloads nothing: fetch tutorial inputs
# and pseudopotentials yourself from their official pages, then run this.
#
# Usage: scripts/hpc/stage_inputs.sh <run_dir> <step> [<step> ...]
#   e.g. scripts/hpc/stage_inputs.sh /share/<group>/<unity_id>/rtsc/pb-tutorial scf ph q2r matdyn nscf epw
set -euo pipefail

run_dir="${1:?usage: stage_inputs.sh <run_dir> <step> [<step> ...]}"
shift
[ "$#" -gt 0 ] || { echo "name at least one step (scf ph q2r matdyn nscf epw)" >&2; exit 2; }
cd "$run_dir"

case "$PWD" in
  "$HOME"|"$HOME"/*) echo "refusing: run directory is under \$HOME; use the work area" >&2; exit 3 ;;
esac

missing=0
for step in "$@"; do
  [ -f "$step.in" ] || { echo "missing input: $step.in" >&2; missing=1; }
done

# Every pseudopotential named in ATOMIC_SPECIES must be present locally.
pseudo_dir=$(grep -h -o "pseudo_dir *= *'[^']*'" ./*.in 2>/dev/null | head -1 | sed "s/.*'\(.*\)'/\1/")
pseudo_dir="${pseudo_dir:-.}"
for upf in $(awk '/ATOMIC_SPECIES/{f=1;next} f&&NF>=3{print $3} f&&NF<3{f=0}' ./*.in 2>/dev/null | sort -u); do
  [ -f "$pseudo_dir/$upf" ] || { echo "missing pseudopotential: $pseudo_dir/$upf" >&2; missing=1; }
done

# Jobs must not try to reach the network.
if grep -l -E "https?://" ./*.in >/dev/null 2>&1; then
  echo "an input file contains a URL; compute nodes cannot download it" >&2
  missing=1
fi

[ "$missing" -eq 0 ] || { echo "staging incomplete; fix the items above before sbatch" >&2; exit 1; }

mkdir -p logs
{
  echo "staged=$(date -Is)"
  echo "host=$(hostname)"
  echo "steps=$*"
  sha256sum ./*.in
  find "$pseudo_dir" -maxdepth 1 -name '*.UPF' -o -maxdepth 1 -name '*.upf' | sort | xargs -r sha256sum
} > logs/staging-manifest.txt
echo "staged OK: $(wc -l < logs/staging-manifest.txt) lines written to logs/staging-manifest.txt"
