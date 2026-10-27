# End goal and action plan

## End goal

The project will count as successful only when an independently reproducible
material demonstrates superconductivity with all of the following:

- a transition in the range 273.15–298.15 K (0–25 °C);
- zero resistance or a resistance drop with a defined measurement protocol;
- magnetic evidence of superconductivity, such as a reproducible diamagnetic
  shielding or Meissner response;
- a documented synthesis route that produces the claimed phase repeatedly;
- pressure and temperature measured at the sample, with calibration and
  uncertainty reported; and
- independent replication on a separately prepared sample.

Pressure is a separate requirement. We will report two target tiers:

1. **Ambient target:** at or below 0.000101325 GPa (1 atm).
2. **Engineering-pressure target:** at or below 0.1 GPa (100 MPa, 1,000 bar),
   as a provisional upper bound for a purpose-built metal pressure vessel.

The 100 MPa ceiling is a screening definition, not a construction instruction.
The achievable pressure depends on vessel geometry, material, seals, temperature,
fatigue life, and safety certification. A qualified pressure-vessel engineer must
set the limit before any experiment. Results above 100 MPa remain scientifically
interesting but do not satisfy the engineering target.

## Gates

### Gate 0: trustworthy evidence

Add pressure, pressure uncertainty, measurement method, synthesis conditions,
sample identity, and publication date to the evidence model. Obtain a source that
contains those fields; the current UCI and 3DSC files do not provide enough
pressure information for the end goal.

### Gate 1: honest computational benchmark

Train a transparent baseline for one selected material family. Keep exact and
canonical duplicates together, hold out entire families, and report chronological
performance separately. A prediction cannot pass this gate by average score
alone; its failure modes and uncertainty must be documented.

### Gate 2: physics screening

Use the baseline to select a small set, then apply structure and stability
filters. Reproduce a published calculation before making new claims. For the
electron–phonon lane, record Quantum ESPRESSO, Wannier90, and EPW versions,
pseudopotentials, convergence settings, and pressure assumptions.

### Gate 3: synthesis and characterization

Work with a laboratory partner whose equipment can make the candidate and measure
the target pressure. The experimental record must distinguish a superconducting
transition from structural, contact, magnetic, or pressure artifacts.

### Gate 4: replication

Repeat the synthesis and measurement independently, including thermal cycling and
pressure release/reapplication where relevant. Only then promote a candidate to a
claim about the target window.

## Immediate actions

1. Extend the normalized schema with pressure, pressure uncertainty, method,
   synthesis, sample, and date fields.
2. Add a pressure-aware source such as NIMS SuperCon, preserving its original
   records and license terms.
3. Build a family-held-out baseline and a chronological split.
4. Select an experimental collaborator and record their chemistry, structure,
   temperature, and pressure limits as hard candidate constraints.
5. Reproduce one published calculation in that accessible material family.
6. Generate a shortlist only after those checks, then send a small set to the
   laboratory for synthesis and measurement.

The current triage runs are therefore useful for choosing what to inspect next,
but they are not evidence that any candidate meets either target tier.
