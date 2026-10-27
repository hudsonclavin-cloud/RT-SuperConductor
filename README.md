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

See [the architecture](docs/architecture.md) and [the roadmap](docs/roadmap.md)
for the working design.

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

This repository is a software foundation for research collaboration. A computed
score or predicted transition temperature is a hypothesis until it is supported
by an appropriate physical calculation and reproducible measurement.
