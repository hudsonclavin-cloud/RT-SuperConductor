# Reviewing canonical duplicates

Run the pipeline with `--review-output` to create
`data/processed/review-queue.jsonl`. Each line is one review unit. A
`duplicate_composition` unit contains the canonical formula and every source
record that currently maps to it.

Treat a duplicate formula as a prompt for inspection, not permission to merge.
Compare the source reference, reported transition temperature, pressure,
measurement method, sample identity, structure identifier, and synthesis details
when those fields become available.

Record decisions in a separate local file such as
`data/review/duplicate-decisions.csv` with these columns:

```text
canonical_composition,record_id,decision,reason,reviewer,reviewed_at
```

Useful decisions are `same_material`, `different_sample`, `different_structure`,
`conflicting_measurement`, and `needs_source_check`. Do not delete or rewrite
the raw records after deciding. The decision file becomes a versioned curation
layer once its licensing and privacy implications have been checked.

The current queue is intentionally conservative. Formula parsing supports
explicit elemental formulas and rejects grouping, hydrate, charge, and variable
notation for later review rather than guessing.
