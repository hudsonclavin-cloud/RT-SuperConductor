# Structure-aware phase status

The structure workflow currently retrieves reference CIFs and prepares a
deduplicated registry, CIF checks, derived geometry fields, and a blocked by
default external calculation manifest.

## Evidence categories

- **Literature observations:** retained in harmonized records with review and
  provenance labels.
- **Reference structures:** CIFs from Materials Project; these do not prove the
  measured sample has that phase.
- **Machine-learning predictions:** only usable after a sufficiently diverse
  empirical training set exists. The current reviewed labels are too sparse for
  reliable family extrapolation.
- **First-principles calculations:** none are completed by the GitHub workflow.
- **Experimental evidence:** only reported literature measurements; no new lab
  results are represented.
- **Open uncertainties:** polymorph matching, defects, synthesis conditions,
  convergence, and pressure constraints remain to be established.

The registry script reports high-value geometry features (cell volume,
density, minimum distance, symmetry, occupancy, formula, and a cutoff-based
cutoff coordination estimate) and labels dimensionality, prototype similarity,
and formation energy as uncomputed when no trusted calculation/source exists.
The coordination estimate is geometry-only and cutoff-dependent. DFT jobs remain
blocked until pseudopotential files, hashes, functional, settings, and an
external compute target are supplied.

The current structure baseline from the downloaded artifact was 26 records, 21
retrieved CIF files, and 13 unique MP IDs. Deduplication should reduce repeated
record-level CIFs to one structure entry while preserving each literature
record in the lineage table. No Bader, ELF, COHP, Wannier, or EPW values have
been produced. See `docs/structure-aware-phase-checklist.md` for acceptance
criteria and `docs/external-compute.md` for the external run interface.
