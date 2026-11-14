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

Negatives per condition in the stable snapshot: current 1,056; audited 951; audited plus new 961; plus Hosono 1,070. Positives are the same 1,192 throughout. Test rows: 2,248 (all) and 2,143 (audited subset, tier D rows removed from testing too).

| Training negatives | Boosting, all test rows | Boosting, metallic test rows | Random forest, all | Random forest, audited |
| --- | --- | --- | --- | --- |
| current (every legacy zero) | 0.799 | 0.804 | 0.792 | 0.799 |
| audited (tier D removed) | 0.785 (−0.014, 0 of 3) | 0.816 (+0.012, 3 of 3) | 0.801 (+0.009, 3 of 3) | 0.817 (+0.017, 3 of 3) |
| audited plus new (tiers A–C) | 0.788 (−0.011, 0 of 3) | **0.819 (+0.016, 3 of 3)** | 0.801 (+0.009, 3 of 3) | **0.818 (+0.018, 3 of 3)** |
| plus Hosono (tier H) | 0.787 (−0.012, 1 of 3) | 0.820 (+0.017, 3 of 3) | 0.798 (+0.005, 2 of 3) | 0.812 (+0.013, 3 of 3) |

Paired differences are against "current" on the same shuffles. "Audited" test rows are the ones without tier D labels: the metals and plausible compounds the classifier is actually asked about.

**What this means**

- **The suspect negatives were flattering the score.** The wide-gap insulators are easy to classify, so they inflate any score computed on them. On the rows that matter, both models improve by 0.012 to 0.018 AUC on every shuffle once the suspect negatives are out of training. On all rows, the forest improves (+0.009) and boosting drops slightly (−0.011 to −0.014), which fits boosting having exploited the "insulators are not superconductors" shortcut harder than the forest.
- **Hosono's list shows no clear effect either way.** Boosting is unchanged (+0.001 on the audited rows) and the forest is slightly worse (−0.006). It stays excluded for a reason that doesn't depend on this test: the source mixes measured non-superconductors with compositions that could not be synthesized.
- **SuperCon-MTG mostly corroborates what we had.** It adds about 10 usable negatives to the stable snapshot; nearly all of its genuinely new zeros are Hosono's.
- **The production classifier trains on tiers A–C only** (`negative_training_use = use`) and reports its held-out quality on the same basis.

## Quarantined positives

Negatives were not the only labels in question. The 3DSC rows on structure `mp-510625` (CuCrO₂ at 132 K and two iron-doped versions at 117.5 K and 98 K) are quarantined by `rt_superconductor.label_quarantine` pending a paper check, and a Tc above 77 K without copper triggers the same rule. The raw SuperCon export carries the same three values, so `build_mp_candidates.py` now drops a quarantined label from every source, not just 3DSC's copy (9 reports in the stable build). Before this, CuCrO₂ was a positive training example for the classifier through the SuperCon route even though Codex's pipeline had excluded it. A reviewer decision of `retain` in `data/training/label_audit_queue.csv` releases a label. Removing the three rows moved the classifier's held-out AUC from 0.818 to 0.829.

The biggest remaining lever is the one no download provides: verified nulls near the top candidates, recorded in `data/training/negative_evidence.csv` from the papers themselves.

## Wrong negatives found in November 2026 (for the owner's review)

Six compositions that the network's table (`data/processed/scratchnet/compositions.csv.gz`)
labels as non-superconductors are reported superconductors. Each rests on two
SuperCon zeros, which per the table above is one lineage. Nothing has been
relabelled: these are listed for a person to check against the papers.

| Formula | Where it sits | What the literature reports | Source | Confidence |
| --- | --- | --- | --- | --- |
| YRu3B2 | development, fold 3 | Bulk superconductor, 0.63 to 0.81 K by criterion, three independent groups; the 1980 null stopped at 1.2 K | arXiv:2512.16945, 2512.09314 (Phys. Scr. 101, 055912), 2512.08514 | high |
| LuRu3B2 | development, fold 2 | Bulk superconductor, 0.95 K (specific heat), about 90% superconducting fraction | arXiv:2512.16945 | high (one group) |
| ScS | development, fold 1 | Bulk superconductor, 5.1 K, rocksalt; first reported in 1974, Tc depends on Sc vacancies | Phys. Rev. B 106, L020504 (arXiv:2203.16458); Moodenbaugh et al., Mat. Res. Bull. 9, 1671 (1974) | high |
| W3Re2C | locked test set | Bulk superconductor, 6.2 K, beta-Mn type, about 100% shielding | arXiv:2509.15078; J. Am. Chem. Soc. 148, 2429 (2026) | high |
| WB | development, fold 3 | 4.3 K (tetragonal) and 2.8 K (orthorhombic), one group, bulk nature not checked; SuperCon also holds an older zero | Kayhan et al., Solid State Sci. 14, 1656 (2012) | medium |
| NbS | locked test set | Orthorhombic NbS superconducts at 6.0 K (2023); stoichiometric NbS of unstated structure was measured as not superconducting in 1954; NbS0.9 superconducts at 3.8 K | Ruan et al., Phys. Rev. B 108, 174517 (2023) | medium: depends on the crystal form |

Checked and **not** found to be wrong:

| Formula | Finding |
| --- | --- |
| VRu | Equiatomic VRu is not reported to superconduct; V-Ru superconducts only on the vanadium-rich side (about 42 to 47.5 at.% Ru). Label likely right (low confidence: the 1971 paper was seen only as an abstract). |
| W2B5 | No phase of this composition is supported. The phase once called W2B5 is WB2 (W2B4); its 5.4 K report was not reproduced in single crystals, which show no superconductivity down to 1.8 K below 50 GPa. |
| V3Pb | No measurement found; the compound probably does not form. |
| Mo3Re2C | No measurement found (its tungsten twin W3Re2C superconducts). |

How they were found: YRu3B2 and LuRu3B2 came up while verifying published
machine-learning picks (`data/benchmarks/prospective-cases/`). The next seven
came from Supplementary Table F.23 of arXiv:2604.23758, which lists literature
Tc values extracted by a language model; Mo3Re2C was added because the model
rated it highly. Each was then looked up by one reader and re-checked by a
second, sceptical reader against the original papers (full text where it
could be reached, otherwise abstracts and citing papers).

What it means:

- **The locked score was computed against at least two wrong answers**
  (W3Re2C and NbS, out of about 1,950 locked rows). The locked set has been
  used once and is not rescored. [Likely] The effect on its AUC is small.
- **Old nulls stop too high.** Two of the six were missed because an early
  measurement stopped at 1.2 K, and one depends on which crystal form was
  measured. A zero with no stated lowest temperature is weak evidence.
- **To act on this**, read each paper and record it in
  `data/training/negative_evidence.csv` with `superconductivity_observed = yes`
  and `tc_k`. `build_mp_candidates.py` turns such a row into a positive report.
  ~~`scripts/build_scratchnet_data.py` does not read that file, so the network's
  table would need the same change made there before a rebuild.~~
  **Correction (14 November 2026):** it does. `scripts/build_scratchnet_data.py`
  takes its labels from `build_mp_candidates.known_index`, which reads this
  file, so one recorded row changes both tables after a rebuild
  (`tests/test_scratchnet_data.py` checks this with YRu3B2). The network's
  table checksum in `docs/scratchnet-data.json` will change with it.
