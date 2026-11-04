# Human guide: checking superconductivity labels

This 30-row sample is a triage exercise, not a set of confirmed facts. It was selected from the 3DSC table with fixed seed **20261104** before using model residuals: 10 Ferrites, 10 Other, and 10 Cuprates. Each group includes at least two rows whose database Tc is zero and at least three fractional-formula rows. The sample is deliberately mixed so reviewers see positive, zero, and doped labels. A zero in a database is not proof that a material was experimentally tested and found nonsuperconducting.

Open [label-audit-template.csv](label-audit-template.csv) in a spreadsheet. Each line is one source-database record. Keep the supplied values intact; add reviewer findings only in the blank human columns.

1. Start with `record_id`, `formula`, `family`, and `database_tc_kelvin`. These say what the dataset row reports, not what a paper necessarily measured. `synthetic_doping_flag=True` means the 3DSC structure table marks a computationally modified structure; it does not mean a lab synthesized it.
2. Open each entry in `doi_available`. A 3DSC `doi_2` value may identify a data or structure record, and SuperCon2/SuperMat are extracted or harmonized literature records. Treat each DOI as a lead. Follow it to the publisher or an open paper, check the title and material, then put the DOI you actually checked in `source_doi_checked`. If several DOIs are listed, record the one that directly supports this composition and measurement; mention disagreements in `reviewer_notes`.
3. In the paper, look for the experimental methods and results passage tied to the exact sample. Check the full composition and doping, phase, sample form, and how Tc was defined: resistive onset, midpoint, zero resistance, magnetic onset, shielding, or a heat-capacity anomaly. Put those details in `exact_sample`, `phase`, and `tc_definition`.
4. Record measurement pressure and synthesis pressure separately. A material synthesized under pressure and measured after decompression is not a high-pressure superconductivity measurement. If the paper does not say, leave the value blank and say “not reported” in `reviewer_notes`; do not assume ambient pressure.
5. `mp_id` and `cif_parent_formula` refer to a computational structure record. A matching formula does not prove that the CIF represents the paper's sample, oxygen content, phase, or vacancy ordering. Set `cif_matches_sample_yes_no` only when the paper and structure support the same sample closely enough to justify it; otherwise leave it blank and explain the gap.
6. After checking the source, choose `keep` if the label and evidence agree, `fix` if a supported value or provenance needs correction, or `quarantine` if the row is ambiguous or mismatched. Put the reason in `reviewer_notes`. Never turn missing pressure or missing evidence into a negative measurement.

## Worked evidence trail

This example uses a SuperCon2 row to show how a source can be checked; it is separate from the 30-row 3DSC sample. The source export row has ID `61ba95dc5ad933a7cac50c65`, formula text `La[O 1-x F x ]FeAs`, database Tc `26 K`, and DOI `10.1088/0953-2048/21/10/105024`. Its `materialId` is not identified as a Materials Project ID, and the row supplies neither a CIF parent nor a 3DSC synthetic-doping flag, so those cells remain blank.

The [IOP publisher page](https://iopscience.iop.org/article/10.1088/0953-2048/21/10/105024) resolves to Gao et al., “Preparation of LaFeAsO0.9F0.1 wires by the powder-in-tube method” (2008). The abstract says: “The transition temperature of the LaFeAsO 0.9 F 0.1 wire is around 25 K.” It also says X-ray diffraction found LaFeAsO0.9F0.1 to be the main phase in the wires. That makes the paper a relevant lead for the record's composition and provides a reported wire transition near 25 K. The database's extracted 26 K and the paper's “around 25 K” are not identical; inspect the full paper for the transition criterion and exact sample before deciding whether the difference is a correction or an acceptable rounded/reporting difference. This guide does **not** assign a human verdict.

| record_id | formula | family | database_tc_kelvin | mp_id | cif_parent_formula | synthetic_doping_flag | doi_available | source_doi_checked | exact_sample | phase | tc_definition | measurement_pressure_gpa | synthesis_pressure_gpa | cif_matches_sample_yes_no | reviewer_notes | verdict_keep_fix_quarantine |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 61ba95dc5ad933a7cac50c65 | La[O 1-x F x ]FeAs | Alloys (source label) | 26 |  |  |  | 10.1088/0953-2048/21/10/105024 |  |  |  |  |  |  |  |  |  |

The publisher page is the source used for this example; the human-audit columns above stay empty for the reviewer. The page's abstract is evidence to inspect, not a substitute for checking the full paper or declaring that this 26 K record is correct.

Here is how I would carry that source evidence into a separate worked row. The example is not added to the 30-record audit CSV, and it does not assign a verdict:

| field | source-supported example entry |
|---|---|
| `source_doi_checked` | `10.1088/0953-2048/21/10/105024` (publisher page opened) |
| `exact_sample` | The abstract describes a LaFeAsO0.9F0.1 wire. |
| `phase` | X-ray diffraction says LaFeAsO0.9F0.1 is the main phase; this does not prove every grain or the CIF is identical. |
| `tc_definition` | Abstract says “around 25 K”; it does not specify onset, midpoint, or zero resistance. Check the results/methods before entering a criterion. |
| `measurement_pressure_gpa` | The abstract does not state it. Keep blank until the paper is checked. |
| `synthesis_pressure_gpa` | The abstract does not state it. Keep blank until the paper is checked. |
| `cif_matches_sample_yes_no` | No CIF is supplied for this SuperCon2 example, so leave blank. |
| `reviewer_notes` | Database label is 26 K; the abstract reports “around 25 K”. The full paper is needed to decide whether these refer to the same transition criterion. |
| `verdict_keep_fix_quarantine` | Blank. The abstract alone does not justify a final audit verdict. |

To regenerate the sample after a fresh checkout, first create a full 3DSC harmonized table and structure index with `scripts/ingest_3dsc.py`, directing its outputs to a working directory. The checked-in `data/training/harmonized_supercon.csv` can contain only the small reviewed set; the audit builder will stop if it does not include the three target families. Then pass those generated paths explicitly, for example:

```bash
python scripts/ingest_3dsc.py \
  --output /tmp/rt-supercon-audit/harmonized.csv \
  --structure-index /tmp/rt-supercon-audit/structure-index.csv \
  --report /tmp/rt-supercon-audit/ingestion.json \
  --features /tmp/rt-supercon-audit/features.csv
python scripts/build_label_audit.py \
  --input /tmp/rt-supercon-audit/harmonized.csv \
  --structure-index /tmp/rt-supercon-audit/structure-index.csv
```
