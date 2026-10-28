# RT-SuperConductor founding charter

## Mission

RT-SuperConductor will be a hybrid AI and physics discovery engine for finding
materials that could support a reproducible room-temperature superconducting
state under useful pressure conditions.

The system will search a large materials space, simulate progressively more
expensive virtual experiments, learn from every result, and recommend the next
small set of physical experiments with the highest expected value.

## Scientific target

The target is a material that demonstrates superconductivity between
273.15 K and 298.15 K (0–25 °C) at either:

1. approximately atmospheric pressure; or
2. a pressure at or below a provisional 0.1 GPa (100 MPa) engineering ceiling.

The 100 MPa value is a computational screening boundary. A qualified engineer
must define the safe pressure, vessel design, seals, fatigue limits, and
operating procedure for any experiment.

A successful scientific claim requires a reproducible synthesis, a defined
resistance measurement showing a superconducting transition, magnetic evidence,
calibrated temperature and pressure at the sample, and independent replication.

## What the system will do

The research loop has six layers:

1. **Evidence layer** — ingest literature, structures, calculations, and
   laboratory results with provenance and explicit unknown values.
2. **Learning layer** — train transparent baseline models, estimate uncertainty,
   and later add generative or foundation models after leakage-resistant tests.
3. **Candidate layer** — propose compositions, substitutions, structures, and
   material families subject to chemistry, stability, synthesis, and pressure
   constraints.
4. **Simulation layer** — run cheap composition screening for thousands of
   candidates, structure and stability screening for hundreds, and detailed
   electronic, phonon, magnetic, or electron–phonon calculations for a small
   shortlist.
5. **Decision layer** — use active learning to choose the next virtual or
   physical experiment based on expected target performance, uncertainty,
   novelty, feasibility, and information value.
6. **Feedback layer** — return calculations, successful measurements, failures,
   and inconclusive results to the evidence store without converting unknowns
   into negative labels.

## Classical and quantum computing strategy

The first working system will run on ordinary CPUs and established scientific
software. Quantum computing is a planned extension for selected hard problems,
such as correlated electronic states, magnetic Hamiltonians, and ground-state
energy estimation. The software will expose explicit simulation-stage and
backend contracts so a quantum backend can replace or augment a classical
backend without changing the evidence or decision layers.

Quantum hardware is therefore an accelerator and research direction, not a
dependency for the first useful results.

## Guardrails

- A prediction is a hypothesis until an appropriate calculation and measurement
  support it.
- Automatically extracted literature values remain linked to their source paper
  and require review; they can describe magnetic or structural transitions.
- Missing pressure, temperature, synthesis, or method data stays unknown.
- A high predicted transition temperature cannot compensate for an unknown or
  infeasible pressure requirement.
- Exact and near-duplicate records remain together during evaluation.
- Experimental partners and qualified engineers set the final measurement and
  pressure limits.

## First implementation milestone

The first code milestone is a model-agnostic active-learning spine:

- normalized evidence records;
- candidate hypotheses with uncertainty and feasibility fields;
- deterministic acquisition scoring and batch selection;
- stable simulation tasks with provenance and backend contracts; and
- tests that keep candidate proposals separate from observations.

Later milestones will add a composition baseline, structure enrichment through
Materials Project, reproducible Quantum ESPRESSO/Wannier90/EPW workflows,
generative candidate proposals, and quantum-computing adapters.

## Definition of success

RT-SuperConductor succeeds as software when it can produce a reproducible,
expert-reviewable shortlist whose next experiments are better justified than a
simple critical-temperature leaderboard. It succeeds scientifically only when
the full evidence standard above is met by an independently replicated material.
