# Current study summary for faculty review

**Target:** a reproducibly synthesizable superconductor at **273.15–298.15 K and 1 atm**, or below a separately justified engineering pressure ceiling. No model score or retrieved reference structure is evidence that this target has been achieved.

The completed study includes **5,773 3DSC records and reference CIFs**. The formula baseline's cross-family MAE is **10.4523 K**, compared with **16.1169 K** for the mean predictor. Initial lattice/spin additions changed that error only slightly; frozen CHGNet, Tc-fine-tuned CHGNet and scratch ALIGNN representations scored **10.5140**, **11.4483** and **12.2198 K**. These are grouped, embargoed tests with disclosed proxy structures and short training pilots, not a claim that structural physics is irrelevant. Primary MP collection supplies DOS/orbital fractions for **5,243 rows** and bulk moduli for **2,757**. The completed Phase 3.5 combination scored **10.4464 K**, versus the **10.4523 K** baseline; orbital features alone scored **10.6626 K** globally but improved within-Oxides MAE from **3.9724 to 3.5957 K**. These results show a family-dependent benefit, not a general improvement. The constrained campaign prepared **60 hypotheses**: **28** ordered inputs completed ML phonons (**20** imaginary-mode signals, **8** without that signal); **32** remain blocked by disorder. Al control false positives prevent treating these as confirmed instability verdicts. The EPC extrapolation audit had **14/1,057** above-training-maximum predictions, but only **1** reference-derived target also exceeded that maximum. Independent calibration helped some tasks and worsened the global family-shift task. No first-principles or experimental room-temperature result was obtained.

The current benchmark does not establish ambient-pressure performance: 3DSC does not provide the needed measurement-pressure labels. Reference magnetism is DFT-computed; neutral-atom shell/d-electron counts and DOS-window fractions are different descriptors, not measured electron occupations. The valid **5,243** DOS rows passed a parent-reference element-set check, which does not prove phase/sample equivalence. All **14 MVC rows** remain quarantined: returned curves for 11 rows mismatched their parent element sets and three rows had no returned DOS. No MVC numerical properties were used. The 3DSC zero-Tc convention also requires source review before interpreting a row as a tested negative.

The next useful faculty contribution is to approve an ordered conventional reference and a converged DFPT setup, then evaluate a small blinded candidate/control panel. See the [one-page brief](physics-features-deep-one-page.md), [complete tables](modeling.md), and [proposed tests with failure decisions](next-experiments.md). New first-principles and experimental evidence must remain distinct from ML predictions and prepared calculation templates.

The previous summary is preserved below as project history; current source checks, limitations and metrics above supersede older descriptions of the evaluated feature set or assumed pressure conditions.

---

# Autonomous Superconductor Discovery Engine

## Executive research summary for NC State faculty review

### Research question and design principle

The RT-SuperConductor project is a physics-constrained discovery workflow for
identifying superconducting materials that merit experimental or first-principles
follow-up. Its central design choice is to keep evidence, statistical prediction,
thermodynamic stability, and electronic-structure calculations as separate but
auditable stages. A model score is therefore treated as a prioritization signal,
not as evidence that a material is superconducting.

The current Phase 4 implementation harmonizes literature and benchmark data
from SuperCon2, 3DSC, SuperMat, and the UCI superconductivity dataset. Every row
has a deterministic record identifier, normalized formula, critical temperature
in kelvin, applied/hydrostatic pressure in GPa, measurement method, sample form,
DOI or source identifier, review status, provenance, `is_synthetic`, and a
chemical-family label. Controlled
vocabularies and range checks make malformed or ambiguous rows fail at the data
boundary. Pending and excluded rows remain visible for review and gate testing;
synthetic demonstration rows remain available for fixture coverage but are not
silently promoted into empirical model weights. The current trainer admits
confirmed literature and curated-database rows only when `is_synthetic=0`, and
excludes automatic SuperCon2 extraction. Confirmed reviewed records are bridged
with `provenance="literature"` and `is_synthetic=0`.

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

The surrogate predicts reported \(T_c\) in kelvin from two classes of inputs,
after filtering to `review_status` of `confirmed` or `curated_database` and
`is_synthetic=0`:

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

- ambient-pressure \(T_c\) MAE no greater than 12 K for confirmed or curated,
  non-synthetic rows at \(P \le 0.000101325\) GPa;
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

### NC State outreach and compute-access plan

The following contact ranking and approach are leads supplied by the student;
verify current appointments, research fit, and availability from NC State's
current faculty/group pages before outreach. No contact has endorsed the
project, confirmed Quantum ESPRESSO use, or offered compute access.

1. **Alexander “Lex” Kemper (Physics)** is the first scientific-fit request
because his reported work includes computational superconductivity, cuprates,
and iron-based systems. Ask for a 15–20 minute reality check on the YBCO
structure ambiguity and whether the proposed descriptors or DFT route make
physical sense.
2. **Douglas Irving (Materials Science and Engineering)** is a strong methods
and materials-informatics contact for guidance on narrowing the workflow,
choosing defensible structures, and setting convergence criteria. Ask whether
there is a suitable course or group opportunity for learning DFT.
3. **Benjamin Beeler (Nuclear Engineering)** is a relevant undergraduate-path
and atomistic/DFT contact. Ask for a learning path or referral to a
superconductivity researcher rather than assuming his group works on this
project's material families.

In parallel, contact NC State Libraries' Research Facilitation Service about
trial access to Hazel HPC and its onboarding process. Faculty mentorship and
compute access are separate requests; an undergraduate may be able to explore
trial access without presenting the full project as established research.

Keep the first message modest and candid: introduce yourself as a first-year
engineering student intending to study nuclear engineering; explain that this
is an independent project organizing superconductivity evidence and exploring
whether computational descriptors can support candidate ranking; say you are
new to DFT and want to validate the physics before spending compute; ask for a
short conversation or graduate-student referral. Bring one concrete case,
oxygen-deficient YBa₂Cu₃O₆.₉₄, and ask how vacancy ordering and phase identity
should be represented. Do not lead with a claim that an AI found a material or
ask a professor to review the entire database.

While compute access is pending, continue paper-level verification of
measurement method, pressure, sample form, DOI, and the definition of the
reported transition. Once a mentor reviews the structure, jointly choose the
method, pseudopotentials, and convergence tests. Reviewed observations and
genuine computational outputs can return to the project with provenance and
uncertainty; extracted literature leads remain pending.

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
