# SuperCon-MTG (vendored copy)

| Item | Value |
| --- | --- |
| Source | https://github.com/adigasuhas/Accelerating-Search-for-Superconductors-using-Machine-Learning, file `SuperCon-MTG/SuperCon_MTG.csv` |
| Commit | `6bee6b21910b466df40b32e7623e7824c0ee6509` (2025-12-31) |
| License | CC BY 4.0 for the dataset (`LICENSE_CC`, copied here); code in that repository is MIT and is not copied |
| Paper | S. Adiga et al., "Accelerating Search for Superconductors using Machine Learning", Comput. Mater. Sci. (2025), DOI 10.1016/j.commatsci.2025.114453; arXiv 2505.11964 |
| SHA-256 of `SuperCon_MTG.csv` | `9743f0b000ee59b4ef1efb9b9ff2c1e967d45abebe9310da69f5d273c398423e` |

13,731 rows: `Material-ID`, `Chemical_Formula`, `Temp_critical` (K). The ID prefix records the lineage of each row:

| Prefix | Rows | Tc = 0 | Origin (per the paper and README) |
| --- | --- | --- | --- |
| `S-` | 13,192 | 1,110 | NIMS SuperCon, MDR datasheet ver. 240322, after the authors' cleaning (duplicates merged, repeated compositions averaged) |
| `H-` | 493 | 453 | Hosono et al. systematic superconductor search; includes materials that were made and showed no transition **and** compositions that could not be synthesized |
| `SM-` | 46 | 21 | SuperMat |

Why it is here: it is the only accessible source that adds negative (Tc = 0) reports beyond the 2018 SuperCon export already used, and its `H-` rows are the only machine-readable copy of Hosono's unsuccessful-materials list reachable from this project. The authors average repeated compositions and may record very-low-temperature bounds as 0, so a 0 here is a database value, not a verified measurement. `scripts/build_mp_candidates.py` records which lineage each zero comes from and never treats a lone `H-` zero as a measured null result.
