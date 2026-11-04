# Existing-but-untested Materials Project shortlist

This screening result combines the Materials Project (MP) with five superconductivity exports. “Untested” means there was no exact reduced-formula match in those exports; it does not establish global literature absence. Model predictions are hypotheses, not superconductivity evidence. All shortlist rows have `dft_verification=pending`.

The MP query returned **30,358** entries within the stability and size bounds (energy above hull ≤0.025 eV/atom; 1–20 sites). Of these, **19,726** passed the summary metallicity prefilter. The database check excluded **2,292** exact formula matches and left **17,434** unmatched formulas; **465** also overlap a source’s unresolved element system and are explicitly flagged for review. All **17,434** structures fetched for unmatched formulas were ordered. Their **252,806** task references were checked; **13,535** materials had a metallic task result under GGA+U, r2SCAN, or r2SCAN+U, while **3,899** had no qualifying method result. The resulting method-qualified pool contains **13,535** materials.

| Filter/result | Rows |
|---|---:|
| MP summaries returned within query bounds | 30,358 |
| Energy above hull ≤0.025 eV/atom | 30,358 |
| Site count ≤20 | 30,358 |
| Summary metallicity prefilter | 19,726 |
| Exact formula matches excluded | 2,292 |
| Unmatched reduced formulas before ordering check | 17,434 |
| Ordered structures | 17,434 |
| GGA+U/r2SCAN method-qualified metals | 13,535 |
| No qualifying metallic method result | 3,899 |
| Assigned to training families | 13,535 |
| Scored after embargoed family gate | 1,326 |
| Shortlisted (top 50 per eligible family) | 150 |
| Top shortlist rows with chemistry-policy flags | 12 |
| Unresolved element-system overlaps retained for review | 465 |

Families are assigned to the nearest training record using 80% total-variation composition distance and 20% crystallographic symmetry distance. We score only families with shared-MP-ID-embargoed within-family Spearman strictly above 0.5.

| Family | Training rows | Embargoed Spearman | Assigned pool | Shortlist rows | Policy flags |
|---|---:|---:|---:|---:|---:|
| Cuprates | 879 | 0.540 | 278 | 50 | 1 |
| Ferrites | 517 | 0.578 | 383 | 50 | 11 |
| Oxides | 320 | 0.513 | 665 | 50 | 0 |

These families failed the quality gate and were not scored:

| Family | Training rows | Embargoed Spearman | Decision |
|---|---:|---:|---|
| Heavy-fermion | 418 | 0.094 | Excluded: Spearman must be > 0.5 |
| Other | 3,529 | 0.470 | Excluded: Spearman must be > 0.5 |

Ranking uses family-specific random-forest predicted Tc minus tree-spread standard deviation. Tree spread is an uncalibrated uncertainty proxy, and a random forest cannot extrapolate beyond its family’s training target range. Chemistry-policy flags are review prompts, not synthesis-safety approvals. MP stability and electronic labels are computational and still need scientific review and experimental validation.

Formula-source exports and SHA-256 checksums:

| Source | Rows | Distinct parsed reduced formulas | SHA-256 |
|---|---:|---:|---|
| [3dsc](https://raw.githubusercontent.com/aimat-lab/3DSC/8d69ca9c94b83d387378549225d3b7c3af85ca42/superconductors_3D/data/final/MP/3DSC_MP.csv) | 5,773 | 12,968 | `353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24` |
| [stanev](https://raw.githubusercontent.com/aimat-lab/3DSC/8d69ca9c94b83d387378549225d3b7c3af85ca42/superconductors_3D/data/source/SuperCon/raw/Supercon_data_by_2018_Stanev.csv) | 16,414 | 16,060 | `b34a955cc59f7a1556bc92c4b9db559bfd634c03227b7e95337a2e5d69227dd6` |
| [supercon2](https://mdr.nims.go.jp/filesets/b737b44a-b07a-4853-9378-8ba63f644e79/download) | 40,324 | 3,418 | `13a18935213c570bb06d1f7ba4aa209abe535b37eaae3c8dafa382b4bb8efc52` |
| [supermat](https://github.com/lfoppiano/SuperMat/blob/main/data/csv/SuperMat-2.0.csv) | 1,305 | 152 | `80e84ab67649a562a838c5a58fa8bb8025fe1e3225ad2eb664d9bcf3be8ed3ce` |
| [uci](https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip) | 21,263 | 15,295 | `b68ae6b55ea8581eff8b1ffba073a899db7e2d2f7f3b781bb0802f643f51e5f7` |

SuperCon2 and SuperMat include many variable or markup formulas that could not be reduced to fixed compositions. Those unresolved rows contribute element-system review flags, not exact matches. Full filter counts, source hashes, family-gate details, limitations, and the output checksum are in [untested-pool-report.json](untested-pool-report.json); the ranked records are in [untested-pool-shortlist.csv](untested-pool-shortlist.csv).

Run: [GitHub Actions #41905030957](https://github.com/hudsonclavin-cloud/RT-SuperConductor/actions/runs/41905030957) (head `a77c9f7f1b13`, completed 2026-11-04T20:46:11Z).
