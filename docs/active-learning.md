# Active-learning spine

The runnable cycle is:

```bash
python scripts/run_active_learning.py
```

It downloads the pinned UCI and 3DSC inputs, trains family-held-out models,
and emits `data/processed/candidates/top_k.csv`. The row-level fields include
prediction, ensemble uncertainty, chemical family, MP hull energy when
available, and a human-readable reasons string. The file is a review queue;
it is not an experimental result.

The first search layer is dependency-free and intentionally model-agnostic. A
future ML model supplies `predicted_tc_k` and `tc_uncertainty_k`; the selection
logic then applies the target and feasibility constraints.

```python
from rt_superconductor.search import CandidateHypothesis, make_simulation_tasks, propose_batch
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA

candidates = [
    CandidateHypothesis(
        candidate_id="family-a:001",
        composition="Example2X",
        predicted_tc_k=285.0,
        tc_uncertainty_k=35.0,
        pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
        structural_plausibility=0.7,
        synthesis_feasibility=0.5,
        novelty=0.8,
        provenance=("model:baseline-v1",),
    )
]

proposals = propose_batch(candidates, batch_size=10)
tasks = make_simulation_tasks(proposals, input_revision="dataset-manifest-sha256")
```

Known pressures above the 0.1 GPa engineering screen are excluded. Unknown
pressures are excluded by default and can be included only with
`allow_unknown_pressure=True`, which adds an explicit review reason. This keeps
the target constraint visible while still allowing an information-gathering
campaign.

The production acquisition path additionally caps uncertainty at 100 K and
multiplies its reward by structural/stability plausibility. A missing MP API
key therefore produces an explicit degraded shortlist with unknown hull energy;
it does not turn high uncertainty into a stability claim.

The emitted `SimulationTask` values can later be sent to a composition model,
structure service, Quantum ESPRESSO workflow, quantum backend, or laboratory
integration. Results remain `SimulationResult` objects until a reviewer decides
whether they belong in the evidence store.

`scripts/run_three_generation_campaign.py` exercises this spine in three
explicit generations over the pending SuperCon2 review queue: triage, pressure
and target acquisition, then stable task emission. Its output remains a
review artifact; it does not promote extracted literature into training data
or claim that a candidate is superconducting.
