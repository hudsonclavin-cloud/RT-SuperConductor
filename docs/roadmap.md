# Roadmap

## Phase 1: trustworthy evidence (days 1–15)

- Finalize controlled vocabularies and evidence validation rules.
- Add one source adapter and retain raw source identifiers.
- Create duplicate, near-duplicate, and ambiguous-structure review queues.
- Export a versioned dataset manifest with provenance and data-quality summaries.

Exit criterion: every normalized label is traceable to its source, with pressure
conditions and unknown values represented explicitly.

## Phase 2: honest baselines (days 16–30)

- Select one material family with a prospective research partner.
- Implement a composition baseline and uncertainty calibration.
- Add grouped, family-held-out, and chronological evaluation splits.
- Publish failure analysis alongside aggregate metrics.

Exit criterion: results can distinguish interpolation from extrapolation and can
be reproduced from a dataset manifest and model configuration.

## Phase 3: reproduce physics (days 31–60)

- Select a published electron–phonon calculation with available inputs.
- Capture code versions, pseudopotentials, convergence settings, and compute
  requirements in a calculation manifest.
- Reproduce the reported result within documented assumptions and tolerances.

Exit criterion: an independent run produces comparable output and preserves the
full provenance needed to explain differences.

## Phase 4: choose the next experiment (days 61–90)

- Record a collaborator's synthesis and measurement constraints.
- Rank a small candidate set using predicted value, uncertainty, feasibility,
  novelty, and expected information gain.
- Have a domain expert review every candidate and define the next measurement.

Exit criterion: each shortlisted candidate has a traceable rationale, explicit
risks, and a feasible test that can return useful evidence even if negative.

## Deferred work

Generative crystal design, large custom foundation models, and broad searches
across unrelated material families should follow trusted evaluation and a working
rejection pipeline.
