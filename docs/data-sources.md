# Data sources for pressure-aware research

## Immediate source: NIMS SuperCon2

[SuperCon2](https://mdr.nims.go.jp/datasets/2169a31f-ca73-4391-aceb-7acf15066f78)
is the strongest current source for the pressure-aware milestone. The supplied
export contains 40,324 records with extracted critical temperatures, applied
pressure text, measurement methods, synthesis-related fields, formulas, DOIs,
and publication metadata. The source is described as CC BY 4.0.

SuperCon2 is automatically extracted literature data. It can contain values
from unrelated magnetic or structural transitions, malformed pressure strings,
and duplicated passages from one paper. Scalar values are normalized
conservatively; ranges and ambiguous values remain unknown with their original
text in caveats. Every shortlist row must be checked against its cited paper.

The archive's observed main CSV has 23 columns, even though one supplied data
dictionary lists 28. Treat the observed export as versioned input and reconcile
the difference before training a model.

## Secondary pressure source: SuperMat 2.0

[SuperMat](https://github.com/lfoppiano/SuperMat) is the best next import for
the pressure-aware milestone. Its `SuperMat-2.0.csv` file has `material`,
`tcValue`, `pressure`, `me_method`, and source-paper filename fields. The source
repository states that the dataset is CC BY 4.0 and cites DOI
10.1080/27660400.2021.1918396.

Its pressure and transition values are free-text literature annotations. They
must be parsed conservatively and checked against the cited paper. It contains
no crystal structures, and many rows have no material annotation.

The weekly evidence loop reads the CSV from the public `lfoppiano/SuperMat`
repository and records the checked-out source revision and input checksum in
its artifact. The current upstream CSV is small (about 1,300 annotated rows),
not a massive training corpus. It has no DOI column; `filename` points toward
the paper record. The workflow creates a paper-review queue and never treats
these annotations as verified labels. The repository states CC BY 4.0 for the
dataset; retain attribution to the SuperMat authors and its cited publication.

## Sources already imported

- UCI Superconductivity Data: useful composition features and `Tc`, but no
  row-level pressure or measurement method.
- 3DSC_MP: `Tc` plus Materials Project structures, but no pressure field.
- Kaggle's [SUPERCON Dataset snapshot](https://www.kaggle.com/datasets/chaozhuang/supercon-dataset):
  a raw tab-separated SuperCon literature table with 26,321 rows, sample IDs,
  common names, structure labels, and journal citations. It does not provide
  row-level pressure or measurement-method fields, so it is useful for
  provenance and benchmark coverage rather than pressure screening. Its
  featurized file is retained separately and is not treated as new evidence.

These remain valuable for structure and baseline modeling, but neither can label
the ambient-versus-engineering-pressure target by itself.

## Supplementary sources to investigate

Recent search-result leads on mechanisms and tuning are catalogued in
[`recent-literature-leads.md`](recent-literature-leads.md). They are explicitly
unverified and are not part of the training corpus.

- NIMS SuperCon legacy catalog: a broad superconducting-material catalog that
  may provide additional records beyond the SuperCon2 extraction. Obtain an
  authorized export and inspect its current fields before assuming pressure
  coverage.
- Materials Project: computed structures and properties for enriching candidates;
  it is not an experimental pressure/Tc evidence source. API access requires a
  user key and its terms must be followed.
- Paper-specific high-pressure hydride datasets: useful for physics validation,
  but each dataset must be imported with its paper, pressure range, calculation
  method, and license rather than merged as if it were a universal catalog.

## Import order

1. Normalize SuperCon2 pressure, methods, synthesis text, and publication
   metadata; keep automatic-extraction caveats attached to each row.
2. Normalize SuperMat pressure and measurement fields and link each row to its
   source-paper filename.
3. Add the Kaggle SuperCon raw table for literature and citation coverage while
   keeping pressure unknown.
4. Resolve material names and formulas conservatively; preserve unresolved rows.
5. Link records to 3DSC or Materials Project structures where the match is
   reviewable.

## Newly identified sources and uploads

- **Original NIMS SuperCon datasheet** ([dataset](https://mdr.nims.go.jp/concern/datasets/m900p111d?locale=en), [user guide](https://doi.org/10.48505/nims.4488)) is a high-priority addition distinct from SuperCon2. The supplied source audit reports fields for separate Tc criteria, transition width, susceptibility, Meissner fraction, heat-capacity jump, oxygen content, preparation, and sample structure. Inspect the release's full schema before import; preserve `tcn` as a censored “not observed down to” result, not `Tc=0`, and do not confuse maximum applied pressure with pressure at measurement.
- **PhononDB** ([maintainer repository](https://github.com/atztogo/phonondb), [NIMS collection](https://mdr.nims.go.jp/collections/8g84ms862?locale=en)) is a computed harmonic-phonon source. It can contribute stability and vibrational information only for defensible structure/method matches. Harmonic phonons are not electron-phonon coupling, an experimental superconductivity label, or a Tc result. Verify each record's license and settings.
- **Materials Cloud superconductivity calculations** have two identifiers in the supplied materials: record `3kbt5-r3n56` in the source audit and DOI `10.24435/materialscloud:qv-bq` in the screenshot. Their identity and contents are unresolved; the screenshot reports 10.7 TiB while the audit describes a 48.1 GiB archive. Do not bulk-download either. First compare landing-page metadata, version, README, file manifest, schema, and license. These calculations are computational evidence, not observed Tc.
- The two newly uploaded `3DSC_MP` archives are byte-for-byte identical to one another and to the existing `data/raw/3dsc/3DSC_MP.tar.gz` (`sha256=df2575280064e9c477dac4d73280416f5c8bf063b5bf6509e8e3fc9f1dcb1650`). They add no new records. The existing archive has 41,234 CSV rows and 10,904 CIF files; the rows are structure-to-record links and must not be counted as 41,234 independent superconductors.

The current pinned importer run sees 5,773 CSV rows and links 5,773 CIFs, so
archive version and row-count definitions need reconciliation before those
counts are combined.

## What more data is worth, measured 2026-11-27

A learning curve on the ranking learner (`docs/learning-curve.md`) found
about +0.03 AUC for each doubling of the labelled training rows, with no
sign yet of levelling off. That makes one large, new source worth far more
than many small ones. In order of expected value:

1. **NEMAD superconductor database** (Itani, Zhang, Itani and Zang, *J.
   Magn. Magn. Mater.* 652, 174189, 2026; https://nemad.org).
   - 78,203 records over 19,058 compositions, extracted from papers by
     language models.
   - The site describes its data as CC BY 4.0.
   - A visitor who is not signed in sees 10 results per search, so getting
     the bulk file means asking the authors or signing in.
   - Every row needs the same checks as SuperCon2.
2. **Measured nulls from full-text reading** (library text-mining access) and
   from authors of make-and-measure papers. They are small in number, but
   they fill the blind table and the negatives the curve cannot measure.
3. **Larger calculated electron-phonon sets**, such as the 20,000+ metals in
   Gao et al., *Nat. Commun.* 2025 (10.1038/s41467-025-63702-w), against
   the 8,203 Alexandria calculations already here.
   - They help a physics-based screen.
   - A formula-only model of them did not help predict experiments
     (`docs/physics-proxy2.md`).

## NEMAD's magnetic datasets, checked 2026-11-27

The owner supplied the paper (Itani, Zhang and Zang, "The northeast
materials database for magnetic materials", *Nat. Commun.* 16, 9415, 2025,
10.1038/s41467-025-64458-z) and its repository
(github.com/sumanitani/NEMAD-MagneticML, also Zenodo 10.5281/zenodo.17042814).

**What it holds.** No superconductor data. Three tables of compositions:

| Table | Compositions usable here | Where the label comes from |
| --- | --- | --- |
| Ferromagnets, with Curie temperature | 16,337 | extracted from papers by language models |
| Antiferromagnets, with Néel temperature | 6,791 | the same |
| Non-magnetic | 11,136 | calculated (Materials Project), not measured |

**Could ferromagnets serve as non-superconductors?** They would be
*assumed* negatives: nobody cooled them to check. Measured here
(`python scripts/magnetic_negatives_check.py`, results in
`docs/magnetic-crosstab.json` and `docs/magnetic-ferromagnet-scores.json`):

- [Certain] Nearly all are new to us. Of 13,160 ferromagnets with a Curie
  temperature of 50 K or more, 12,721 are neither held nor a doping step
  from a held composition.
- [Certain] The assumption has real exceptions. Of the 122 ferromagnets we
  hold with a label we may read, 37 (30%) are labelled superconductors.
  - That sample is biased upward: a ferromagnet is in a superconductor
    table because someone looked for superconductivity in it.
  - Some are real coexistence (RuSr2GdCu2O8, ErNi2B2C). Some are a
    pressure effect our table does not mark (iron, CrAs).
  - Antiferromagnets are worse: 117 of 281 (42%). They are not usable as
    negatives.
- [Certain] Cycle 6 already discounts most ferromagnets. On 3,000 it has
  never seen, its mean probability is 0.39. For scale, on the development
  rows it gives known superconductors 0.79 and known nulls 0.53.
  - 30% of the ferromagnets still score 0.5 or more, and 4% score 0.8 or
    more.
- [Likely] The ones it scores highest are mostly errors in the magnetic
  table, not in the model. The top of the list holds Ba0.05Pb0.95 and
  Pb0.95Sr0.05 with "Curie temperatures" of 263 K and 103 K. Lead with 5%
  of a second metal is a superconductor.
  - So adding every ferromagnet as a negative would put wrong labels
    exactly where the model is being asked to decide.

**Reading.** Not a source of superconductor rows. A possible source of
assumed negatives, but only with a filter (for example Curie temperature of
300 K or more, and a magnetic element making up a real share of the
formula) and only as a planned learning cycle judged on the development
rows. Nothing was added to any training table here.

## The NEMAD superconductor database, downloaded and checked 2026-11-27

With the owner's API key, the whole superconductor database was downloaded
and checked against this project's data and, for a random sample, against
the source papers (`docs/nemad-superconductor.md`).
- About 2,900 of its compositions are new to us and pass a filter that
  proved reliable in the paper check: NEMAD's own experimental flag, at
  normal or unstated pressure.
- Records without that flag were never right in the sample.
