# Reviewed training data

This directory is reserved for human-reviewed labels that are safe to use in
model training. Keep each submitted file accompanied by its source DOI or
record identifier, reviewer, review date, and any exclusions.

Automatically extracted SuperCon2 rows do not belong here until a reviewer has
confirmed that the reported transition is superconducting, identified the
measurement method, and verified the pressure and sample context.

The continuous-training workflow currently retrains the composition baseline
from the pinned UCI snapshot. A future trainer will merge reviewed files from
this directory only after their schema and provenance pass validation.
