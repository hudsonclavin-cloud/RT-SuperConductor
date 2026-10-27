# Architecture

## Design principles

- Preserve unknown values. Missing transition-temperature evidence is not a
  measured zero, and missing pressure is not ambient pressure.
- Keep every normalized assertion traceable to a source record.
- Separate experimental measurements, reported calculations, and model
  predictions.
- Treat composition-to-structure matching and duplicate detection as uncertain,
  reviewable operations.
- Report interpolation within known families separately from extrapolation to
  unfamiliar chemistry.
- Rank candidates for a specific laboratory and next experiment, rather than by
  predicted transition temperature alone.

## Components

### Evidence ingestion

Source-specific adapters will ingest records from sources such as SuperCon,
3DSC, the UCI superconductivity dataset, and Materials Project. Each adapter
stores an immutable raw-record reference, then emits normalized evidence with
explicit caveats. Importers must never turn an absent transition into `0 K`.

### Evidence store and review

The normalized model records composition, structure, material family,
transition temperature, pressure, synthesis conditions, measurement method,
evidence kind, source, sample identity, confidence, and caveats. A later review
interface will expose duplicate groups and ambiguous structure matches.

### Modeling and evaluation

The first model should be a transparent composition-based baseline with calibrated
uncertainty. Evaluation splits must include:

- grouped splits that keep duplicates and near-duplicates together;
- entire held-out material families;
- chronological splits where publication dates exist; and
- separate metrics for interpolation and extrapolation.

Model artifacts will include the dataset snapshot, split definition, feature
code version, parameters, metrics, and random seed.

### Candidate ranking

Ranking combines model output with uncertainty, novelty, structural plausibility,
synthesis feasibility, pressure requirements, and the information value of a
result. Hard constraints exclude candidates outside a collaborator's available
chemistry, equipment, or pressure range.

### Physics workflows

Adapters will run selected candidates through reproducible calculation manifests.
The first planned lane is electron–phonon superconductivity using Quantum
ESPRESSO, Wannier90, and EPW. These results apply to that physical mechanism and
must not be presented as universal superconductivity predictions.

### Experimental feedback

Every proposed experiment receives a stable identifier. Measurements, failures,
unsuccessful syntheses, and inconclusive results return to the evidence store
without being collapsed into the same label.

## Initial domain model

`EvidenceRecord` is the normalized scientific assertion. `CandidateAssessment`
is a decision-support result derived from evidence and models. This separation
prevents predictions from silently becoming observations.

## Planned persistence boundary

The starter code uses in-memory Python values. A later milestone will introduce
versioned tabular storage (Parquet plus a manifest) before adding a service or UI.
This makes early pipelines reproducible without committing to an operational
database prematurely.
