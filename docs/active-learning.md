# Active-learning spine

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

The emitted `SimulationTask` values can later be sent to a composition model,
structure service, Quantum ESPRESSO workflow, quantum backend, or laboratory
integration. Results remain `SimulationResult` objects until a reviewer decides
whether they belong in the evidence store.

`scripts/build_review_campaign.py` is a **three-stage paper-review
triage**, not three learning generations. It inventories the pending queue,
prioritizes source-verification risks, and emits paper-review tasks. It reports
`learning_cycles_completed=0` and blocks model inference until a validated
candidate model and compatible feature matrix exist. Extracted Tc is retained
only as `extracted_tc_kelvin`; it is never copied into a prediction field and
the workflow emits no hand-assigned “prediction uncertainty.”

A genuine learning cycle requires new reviewed or calculated evidence to return
to the evidence store, followed by validation, a new model fit, evaluation, and
candidate inference. The present repository does not yet complete that cycle:
the pressure-aware model is blocked by too little confirmed, family-diverse
training evidence, and calculation outputs are still external/pending. Do not
interpret review triage, ranking based on extracted Tc, or task creation as AI
training or discovery.

The old `scripts/run_three_generation_campaign.py` path remains only as a
compatibility entry point. It prints a warning and invokes this review-only
workflow, whose report explicitly states that zero learning cycles ran.
