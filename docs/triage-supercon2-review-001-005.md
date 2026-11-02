# First SuperCon2 review-queue triage

This is a citation-level triage of the first five rows produced by
`build_supercon2_review_queue.py` from the pinned SuperCon2 export. It is not
paper-level verification: the export includes extracted labels and citation
metadata, while the workflow environment did not provide the full texts of
these five papers. Their records remain `pending_paper_review`; none of these
notes authorizes a training label.

| Queue ID | Extracted claim | Citation metadata check | Triage |
|---|---|---|---|
| `supercon2-review-001` | MnP, 290 K, ambient | The listed title is “Pressure dependence of the magnetic ground states in MnP” ([DOI](https://doi.org/10.1103/PhysRevB.93.100405)). A 290 K superconducting transition is an extraordinary claim and the title points to magnetic-state measurements, not direct evidence for that Tc. | Do not promote. Verify the exact passage and transition type; likely false extraction or relation-linking error. |
| `supercon2-review-002` | BaFe₂As₂, 150 K, 0.7 MPa | The listed title concerns resistivity anisotropy in underdoped Co-doped Ba(Fe₁₋ₓCoₓ)₂As₂ in its antiferromagnetic orthorhombic phase ([DOI](https://doi.org/10.1103/PhysRevLett.110.207001)). The queue formula omits Co substitution and the extracted Tc is a high-priority outlier. | Do not promote. Resolve composition/doping and inspect the source table or passage; pressure and Tc may be attached to unrelated context. |
| `supercon2-review-003` | CuO₂, 138 K, ambient | The listed title is the broad “Resistance of high-temperature cuprate superconductors” ([DOI](https://doi.org/10.1088/1361-6668/15/7/073020)); `CuO 2` alone is not enough to identify a specific bulk phase or sample. | Do not promote. Resolve the exact compound and determine whether 138 K is a measured transition or contextual value. |
| `supercon2-review-004` | GeH₄, 138 K, ambient | The listed title is “Superconductivity of novel tin hydrides (SnₙHₘ) under pressure” ([DOI](https://doi.org/10.1038/srep22873)). The element in the extracted formula conflicts with the title, and the title explicitly indicates pressure. | Do not promote. Treat as a likely formula/paper-link error; verify against the original record before any correction. |
| `supercon2-review-005` | Hg₀.₈Tl₀.₂Ba₂Ca₂Cu₃O₈.₃₃, 138 K, ambient | The listed title is “Recent advances in high-throughput superconductivity research” ([DOI](https://doi.org/10.1088/1361-6668/ab51b1)), which reads as a review rather than a primary measurement report. | Do not promote. Locate the cited primary experiment and its method, sample, and pressure; otherwise leave unresolved. |

## Result and next verification action

All five are high-risk review leads, not verified positive records. The queue
generator currently ranks by extracted Tc, so these implausible outliers appear
first; high Tc alone is not a meaningful priority score. The next improvement
should rank review work using evidence quality, citation type, measurement
method, sample identity, pressure clarity, duplicate-paper links, and physical
outlier flags. A reviewer with article access should check the original source
passage or table, record page/table/figure and quoted context, then mark a row
confirmed, corrected, excluded, or unresolved. Until then, preserve the raw
extraction and keep the records out of training.
