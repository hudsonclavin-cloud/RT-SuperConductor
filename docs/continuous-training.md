# Continuous training and evidence loop

`.github/workflows/continuous-training.yml` is the software side of a human-in-
the-loop discovery cycle. It runs weekly, on code or evidence changes, and from
the GitHub Actions **Run workflow** button. A loop is useful when a run consumes
new evidence, emits review or calculation decisions, and a later run includes
verified results. Re-fitting the same unchanged data is reproducibility, not
learning.

Each run validates reviewed rows and the harmonized schema, then downloads and
checksums the UCI and NIMS SuperCon2 snapshots. It checks out the public SuperMat
repository and records that source revision and input checksums. It creates up
to 1,000 SuperCon2 paper-review leads and a full SuperMat review queue, grouping
records into within-ceiling, high-pressure, and unknown-pressure buckets. It
then trains the grouped UCI composition baseline and attempts the pressure-aware
surrogate using only confirmed empirical rows. Metrics, source coverage, queues,
predictions, and blocked-data reports are uploaded in a run-specific artifact.

Automatically extracted SuperCon2 and SuperMat rows do not enter training until
a reviewer confirms the transition, measurement method, pressure, and sample
context against the source paper. The queues are leads, not newly discovered
superconductors. SuperMat's CSV lacks DOIs; its paper filename is a trace-back
aid, so some leads need additional source resolution before review. Synthetic
demonstration rows remain excluded from model fitting and evaluation.

The current safe cycle is public-source refresh → hashed snapshots and pending
review queues → human-confirmed evidence → validation and grouped evaluation →
updated model and decision artifacts. A researcher must still supply real,
versioned DFT or laboratory results. Calculated quantities are not experimental
`Tc` labels. GitHub Actions prepares and evaluates work; it does not run QE,
phonons, EPW, Bader, ELF, or COHP calculations.

Minute-scale runs are expected while the workflow downloads modest data,
validates records, and fits CPU baselines. Artificial delays add no science.
Larger training jobs and first-principles calculations belong on a workstation
or HPC allocation with a resource-aware runner. Runtime should grow when actual
data volume or calculation cost grows, not just to make a run take longer.
