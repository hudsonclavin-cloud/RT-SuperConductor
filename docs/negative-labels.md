# Negative labels: what "tested non-superconductor" really means, and where to get better ones

**Main idea:** our negatives were database zeros, not verified null results. Every zero in 3DSC came from the same 2018 SuperCon export, 83 of them were wide-gap insulators nobody would have measured, and our formula matching blurred doping boundaries such as Fe1.01Se (8 K) versus Fe1.03Se (none). Each negative now carries an evidence tier, SuperCon-MTG is a second source, and a template exists for recording negatives you verify in papers. The experiment below measures whether any of this changes the classifier.

## What we found in the existing negatives

| Finding | Evidence |
| --- | --- |
| 3DSC's 1,778 zeros are SuperCon's zeros | `origin_sc = Supercon` on every zero row; 3DSC adds structures, not measurements |
| So "2 reports" was one report | 1,035 of 1,104 negatives had exactly two sources: the Stanev export and 3DSC's copy of it |
| 83 negatives are wide-gap insulators | BN (4.5 eV), BaTiO3, CaTiO3, MoS2, WS2, CaSe: database zeros, not experiments |
| Doping boundaries get blurred | FeSe matched Fe1.03Se (0) and Fe1.01Se (8.25 K); La2CuO4 has both 0 and 32 K reports depending on oxygen; NdBa2Cu3O6.1 (90 K) vs O6.3 (0) |
| Zero can mean "very low" | Rh superconducts at 0.3 mK and Pt at 0.004 K; one database rounds both to 0 |
| 81 compositions have both zero and positive reports | Flagged as `known_mixed_reports`; they are labelled by their maximum Tc |

## Evidence tiers (`negative_evidence_tier` in `data/candidates/mp-candidates.csv.gz`)

| Tier | Meaning | Used in training | Stable snapshot |
| --- | --- | --- | --- |
| A_referenced_null | A person checked the paper: no transition down to a stated temperature (`data/training/negative_evidence.csv`) | yes | 0 so far |
| B_corroborated_zero | Tc = 0 in at least two independent lineages (SuperCon, Hosono, SuperMat, audit), exact match | yes | 13 |
| C_database_zero | Tc = 0 in one lineage, exact match, not a wide-gap insulator | yes | 948 |
| H_unsuccessful_search | Tc = 0 only in Hosono's search list, which mixes measured nulls with synthesis failures | no | 109 |
| D_suspect | Near match only, or band gap > 1 eV in MP | no | 123 |

Lineages matter more than source counts: the Stanev export, 3DSC and SuperCon-MTG's `S-` rows are all SuperCon, so agreeing with each other proves little.

## Sources

| Source | What it holds | Reachable from the cloud session | Status here |
| --- | --- | --- | --- |
| **SuperCon-MTG** (Adiga et al. 2025; [GitHub](https://github.com/adigasuhas/Accelerating-Search-for-Superconductors-using-Machine-Learning), [arXiv 2505.11964](https://arxiv.org/abs/2505.11964)) | 13,731 cleaned SuperCon rows, 1,584 with Tc = 0; `H-` rows are Hosono's list | yes (GitHub) | **Vendored**, CC BY 4.0: `data/training/external/supercon-mtg/` |
| **Hosono et al.** unsuccessful-materials tables ([MRS Bulletin](https://www.cambridge.org/core/journals/mrs-bulletin/article/in-pursuit-of-novel-superconductors-four-years-and-1000-materials/827A52A99F9C1ADD5A5C92FA6346B073); open-access STAM paper [PMC5099821](https://pmc.ncbi.nlm.nih.gov/articles/PMC5099821/); [arXiv 1505.02240](https://arxiv.org/abs/1505.02240)) | About 700 unsuccessful materials with notes on whether each was made and measured | no | Only via MTG's `H-` rows (453 zeros), without the notes that separate measured nulls from synthesis failures |
| **SuperBand** ([GitHub](https://github.com/ljcj007/SuperBand), [arXiv 2409.09419](https://arxiv.org/abs/2409.09419)) | DOI per row; the paper's 1,112 non-superconductors are in a training set on [scidb.cn](https://www.scidb.cn/en/file?id=c2487ff51a6230ba2ff55f0b5a8bfe8f), not on GitHub | GitHub yes, scidb no | GitHub copy has only 2 zeros (Rh, Au0.978In0.022); not used |
| **NIMS MDR SuperCon** ([collection](https://mdr.nims.go.jp/collections/4c428a0c-d209-4990-ad1f-656d05d1cfe2?locale=en), [datasheet ver. 240322](https://mdr.nims.go.jp/pid/650c4826-f0ca-42e8-8dd9-94025a5307ce)) | The official releases with journal references per entry | no | The 2024 release's references would upgrade C-tier zeros toward A; needs a download from a normal network |
| SuperCon-MTG cleaning logs (`Data_Cleaning_Log.xlsx` in the same repo) | Journal references for rows removed during cleaning | yes | Only 32 zero-Tc rows carry a reference; not worth importing |

## Recording a verified negative

Add a row to `data/training/negative_evidence.csv` only after reading the paper. The builder turns a row with `superconductivity_observed = no` and a DOI into tier A for that exact formula.

| Column | What to put |
| --- | --- |
| formula | exact stoichiometry as measured |
| phase_or_sample | polymorph, sample form, growth method if given |
| publication_doi | the measurement paper, not a review |
| page_table_or_figure | where the null result is stated |
| superconductivity_observed | `no` (or `yes` with `tc_k`, which becomes a positive report) |
| lowest_measured_temperature_k | the floor of the measurement: "no transition down to 1.8 K" is the result, not "never superconducts" |
| pressure_gpa, applied_field_t | conditions, blank if ambient and zero field |
| measurement_method | resistivity, susceptibility, specific heat |
| synthesis_conditions | if the paper gives them |
| evidence_note | one line in your own words |
| reviewer, date | who checked it, when |

## Which negatives to look for first

In order of value to the current classifier (`docs/mp-sc-classifier-top200.csv`):

1. **Metallic compounds next to the top candidates.** The list is dominated by Pb, Hg, Bi and Tl intermetallics and by Nb–Pd–Rh, Ir–Ru, Ta–Nb–Mo and V–Ga–Ge alloys. A measured null in any of those systems is the most direct test of a plausible false positive.
2. **Non-superconducting members of doping series** (FeSe excess Fe, La2CuO4 oxygen content, Y1−xTbxNi2B2C). These define decision boundaries and are already partly visible in the `known_mixed_reports` rows.
3. **Different polymorphs of one composition**, to test whether structure features separate measured outcomes.
4. **Hosono's measured nulls**, once the paper's notes identify which `H-` rows were actually synthesized.

Search phrases that work: `"no superconductivity was observed" "down to"`, `"no superconducting transition" resistivity`, `"absence of superconductivity" "specific heat"`, `"<formula>" superconductivity susceptibility`.

## Does better negative evidence change the model?

`python scripts/eval_negative_tiers.py` trains with four negative sets and scores every one on the same frozen test rows (results in `docs/sc-negative-tiers.json`).

Negatives per condition in the stable snapshot: current 1,056; audited 951; audited plus new 961; plus Hosono 1,070. Positives are the same 1,194 throughout. Test rows: 2,250 (all) and 2,145 (audited subset, tier D rows removed from testing too).

| Training negatives | Gradient boosting, all test rows | Gradient boosting, audited test rows | Random forest, all | Random forest, audited |
| --- | --- | --- | --- | --- |
| current (every legacy zero) | 0.802 | 0.804 | 0.800 | 0.803 |
| audited (tier D removed) | 0.789 (−0.014, 0 of 3) | **0.822 (+0.018, 3 of 3)** | 0.806 (+0.005, 3 of 3) | 0.819 (+0.016, 3 of 3) |
| audited plus new (tiers A–C) | 0.793 (−0.009, 0 of 3) | **0.826 (+0.022, 3 of 3)** | 0.807 (+0.006, 3 of 3) | 0.820 (+0.017, 3 of 3) |
| plus Hosono (tier H) | 0.782 (−0.021, 0 of 3) | 0.820 (+0.016, 3 of 3) | 0.803 (+0.003, 2 of 3) | 0.816 (+0.013, 3 of 3) |

Paired differences are against "current" on the same shuffles.

**What this means**

- **The suspect negatives were inflating the score.** With wide-gap insulators in both training and testing, the model collects easy wins by learning "insulators are not superconductors". Take them out of training and the model loses those wins on the insulator test rows, but on the rows that matter (metals, which is what the classifier is actually asked about) it improves by about 0.02 AUC on every shuffle. The old 0.817 headline was partly that easy trick.
- **Hosono's list should stay out until it is audited.** Its rows cost the model on every measure. That matches the caveat that the list mixes measured nulls with compositions that were never made.
- **SuperCon-MTG mostly corroborates what we had.** It adds about 10 usable negatives to the stable snapshot; nearly all of its genuinely new zeros are Hosono's.
- **The production classifier now trains on tiers A–C only** (`negative_training_use = use`) and reports its held-out quality on the same basis.

The biggest remaining lever is the one no download provides: verified nulls near the top candidates, recorded in `data/training/negative_evidence.csv` from the papers themselves.
