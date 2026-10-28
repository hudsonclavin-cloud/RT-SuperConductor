# Autonomous Superconductor Discovery Engine

## Executive research summary for NC State faculty review

### Research question and design principle

The RT-SuperConductor project is a physics-constrained discovery workflow for
identifying superconducting materials that merit experimental or first-principles
follow-up. Its central design choice is to keep evidence, statistical prediction,
thermodynamic stability, and electronic-structure calculations as separate but
auditable stages. A model score is therefore treated as a prioritization signal,
not as evidence that a material is superconducting.

The current Phase 2/3 implementation harmonizes literature and benchmark data
from SuperCon2, 3DSC, SuperMat, and the UCI superconductivity dataset. Every row
has a deterministic record identifier, normalized formula, critical temperature
in kelvin, applied/hydrostatic pressure in GPa, measurement method, sample form,
DOI or source identifier, review status, and a chemical-family label. Controlled
vocabularies and range checks make malformed or ambiguous rows fail at the data
boundary. Pending and excluded rows remain visible for review and gate testing;
they are not silently promoted into the training set.

### Data curation and Materials Project enrichment

The harmonized table is intentionally provenance-preserving. A record can be
traced back to its source database and DOI, while formula normalization gives
the downstream model a stable composition key. Confirmed rows support modeling;
pending rows are candidates for subsequent paper verification; excluded rows
provide negative-control references for the promotion gate.

The enrichment step queries the Materials Project API for every unique formula.
When multiple polymorphs are returned, it selects the document with the lowest
finite energy above hull and appends the Materials Project identifier,
\(\Delta E_{\mathrm{hull}}\) in eV/atom, and crystallographic space-group
number. API errors are retained in a machine-readable report rather than being
silently interpreted as instability. This enables a future version to add
structure identifiers or relaxed geometries without changing the evidence
schema.

### Pressure-aware surrogate model

The surrogate predicts reported \(T_c\) in kelvin from two classes of inputs:

1. Matminer's Magpie elemental-composition statistics, which summarize the
   chemical makeup without leaking a material's measured target; and
2. applied pressure and Materials Project energy above hull, with an explicit
   missing-stability indicator.

The default estimator is a reproducible random-forest regressor. Evaluation uses
GroupKFold with groups defined strictly by chemical family (for example,
cuprates, iron-based pnictides, hydrides, and simple elements/alloys). This
prevents records from the same chemistry family appearing in both the training
and validation folds, which is the relevant test for cross-family discovery.
The output records the feature backend, data checksum, held-out families, fold
metrics, out-of-fold predictions, and the full-fit artifact used for screening.

The model is not intended to replace domain judgment. In particular, a literature
transition can reflect sample quality, pressure history, stoichiometry, or an
indirect measurement. Those uncertainties remain attached to the row and should
be considered when deciding whether a candidate is ready for calculation.

### Promotion and reproducibility gate

The CI workflow trains a fresh artifact, validates the harmonized schema, and
evaluates a strict promotion gate before any deployment step can consume the
model. Positive metrics are computed from family-held-out out-of-fold
predictions; excluded controls are scored with the fitted artifact. The gate
requires:

- ambient-pressure \(T_c\) MAE no greater than 12 K for rows at
  \(P \le 0.0001\) GPa;
- Pearson correlation \(r \ge 0.80\) for the high-pressure \(T_c(P)\) trends
  of benchmark H\(_3\)S and LaH\(_{10}\); and
- a false-positive rate below 5% on excluded, non-superconducting reference
  materials under the 77 K screening threshold.

The gate is fail-closed when a model artifact, benchmark family, or negative
control is missing. That behavior makes dataset gaps visible instead of turning
missing evidence into an apparent pass. Artifact reports and predictions are
uploaded by CI for review and later comparison with a promoted model.

### DFT/EPW hand-off

Candidates pass to the computational queue only when the surrogate predicts
\(T_c > 77\) K and the Materials Project stability value satisfies
\(\Delta E_{\mathrm{hull}} < 0.05\) eV/atom. The orchestrator writes a per-candidate
folder containing Quantum ESPRESSO SCF, phonon, and EPW starter inputs plus a
machine-readable record of the screening decision. The SCF template specifies
cell handling, atomic species and positions, plane-wave cutoffs, smearing,
electron convergence, stress/force output, and an 8\(\times\)8\(\times\)8 k-grid.

Because a formula alone does not define a crystal structure, the current
generator emits deterministic placeholder fractional positions and marks them
as requiring replacement by a relaxed Materials Project/CIF structure before
production calculations. This is a deliberate safety boundary: it makes the
handoff automatable while preventing a placeholder geometry from being mistaken
for a validated structure. The EPW input is correspondingly a starter template
for electron-phonon matrix elements and \(\lambda\), not a completed calculation.

### Faculty collaboration and next research increments

For discussion with NC State researchers such as Dr. Alexander Kemper in Physics
and Dr. Jagdish Narayan in Materials Science, the most useful next increments
are: (i) independent paper-level review of the highest-value records, (ii)
structure-aware enrichment and relaxed geometries for selected candidates, (iii)
family-balanced external benchmarks, and (iv) a closed loop in which DFT/EPW
outputs are fed back as explicitly labeled evidence rather than mixed with
experimental \(T_c\) labels. These steps would connect data-centric discovery
with microscopic questions about pairing, lattice stability, defects, pressure
response, and synthesis feasibility.

The repository is therefore best viewed as a reproducible research instrument:
it narrows a large literature search into an auditable queue, exposes uncertainty
and data gaps, and creates a controlled bridge from candidate ranking to
calculation. Its success criterion is not a single high prediction, but a series
of traceable decisions that survive chemistry-aware validation and faculty review.
