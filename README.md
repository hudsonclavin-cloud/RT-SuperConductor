# RT-SuperConductor

RT-SuperConductor is an early-stage research software project for proposing and
evaluating superconducting-material candidates. Its long-term research target is
a reproducibly synthesizable material that superconducts near 293–300 K at
approximately atmospheric pressure.

The software does not claim to predict or certify superconductivity by itself.
It is intended to support a closed research loop:

```text
published evidence and structure databases
                    |
                    v
       provenance-tracked evidence store
                    |
                    v
      predictive models with uncertainty
                    |
                    v
        constrained candidate ranking
                    |
                    v
       physics calculations and review
                    |
                    v
          experimental measurements
                    |
                    +----> evidence store
```

## Initial scope

The first release will focus on one narrow, testable question:

> Given a defined material family, which previously unseen candidates are most
> valuable to investigate next?

The first useful deliverables are:

1. An evidence schema that preserves provenance, pressure, synthesis details,
   measurement method, uncertainty, and unknown values.
2. Importers that retain source records and flag ambiguous or duplicate entries.
3. Leakage-resistant composition baselines with family-held-out and chronological
   evaluation.
4. A candidate ranking layer that combines predicted performance, uncertainty,
   physical plausibility, novelty, and synthesis feasibility.
5. Interfaces for established calculations such as Quantum ESPRESSO, Wannier90,
   and EPW, followed by expert and experimental feedback.

See the [founding charter](docs/founding-charter.md),
[architecture](docs/architecture.md), and [roadmap](docs/roadmap.md) for the
working design. The end goal and decision gates are defined in
[goal-and-plan.md](docs/goal-and-plan.md).

## Repository layout

```text
docs/                    Design and delivery plans
src/rt_superconductor/   Core evidence and ranking types
tests/                   Fast tests for scientific-data invariants
```

## Development

The starter package requires Python 3.11 or newer and has no runtime
dependencies. From the repository root:

```bash
python -m unittest discover -s tests -v
```

To normalize the two received datasets and write a quality report:

```bash
PYTHONPATH=src python -m rt_superconductor.pipeline \
  --uci data/raw/uci/extracted/unique_m.csv \
  --3dsc-mp data/raw/3dsc/MP/3DSC_MP.csv \
  --supermat data/raw/supermat/SuperMat-2.0.csv \
  --kaggle-supercon data/raw/kaggle-supercon/raw.tsv \
  --supercon2 data/raw/supercon2/supercon2_v22.12.03.csv \
  --structure-root data/raw/3dsc/MP/cifs \
  --output data/processed/evidence.jsonl \
  --report data/processed/quality-report.json \
  --review-output data/processed/review-queue.jsonl
```

The JSONL output retains source identifiers and unknown values. The report
summarizes coverage and exact plus conservatively canonicalized composition
duplicates; it does not claim that records with the same composition are
scientifically identical. The review queue lists duplicate candidates and
formulas that need human inspection before any training split is created.

SuperCon2 is automatically extracted literature data. Its pressure and
temperature values are useful for finding papers to review, but they are not
validated experimental claims. In particular, an extracted temperature can be
magnetic or structural rather than superconducting; verify the original paper
before using a row as a candidate or training label.

After that run, generate five deterministic review passes:

```bash
PYTHONPATH=src python -m rt_superconductor.triage \
  --input data/processed/evidence.jsonl \
  --output data/processed/triage-runs.json \
  --limit 25
```

The five passes prioritize highest reported `Tc`, highest `Tc` with structures,
cross-source evidence, disagreement that may have high information value, and
the intersection of structure plus cross-source evidence. They are review
priorities, not newly discovered materials or model predictions.

This repository is a software foundation for research collaboration. A computed
score or predicted transition temperature is a hypothesis until it is supported
by an appropriate physical calculation and reproducible measurement.

The first AI baseline is documented in [modeling.md](docs/modeling.md) and can
be run after installing `.[ml]` with `PYTHONPATH=src python scripts/train_baseline.py`. It is a
composition-only benchmark with grouped evaluation and ensemble uncertainty;
it is a starting point for the active-learning engine, not a pressure-aware
superconductivity predictor.

The first active-learning spine is available in `rt_superconductor.search`.
It keeps proposed candidates separate from observations, applies the pressure
ceiling explicitly, ranks a batch using uncertainty and feasibility, and emits
stable composition-screen tasks for later ML, DFT, quantum, or laboratory
backends. See the [active-learning guide](docs/active-learning.md) and
[founding charter](docs/founding-charter.md) for the staged AI and simulation
plan.

For a no-terminal Materials Project run, open the repository's **Actions** tab,
choose **Low-budget materials screen**, select **Run workflow**, and download
the resulting `low-budget-screen` artifact. The workflow reads the
`Materials_Project_API` GitHub secret as a runtime-only `MP_API_KEY`; it never
writes the key to the repository or to an artifact.
