# Materials Project: all 30,358 stable materials, saved and scored

**Main idea:** every stable Materials Project material with at most 20 atoms per cell is now saved in the repository and scored. Of the two scoring models, the one trained on real superconductor labels works. The physics model does not yet beat chance on real outcomes.

## What is saved

| File | What it holds |
| --- | --- |
| `data/candidates/mp-stable-summary.csv.gz` + `.manifest.json` | All 30,358 materials from the MP summary API (energy above hull ≤ 0.025 eV/atom, ≤ 20 sites, not deprecated): stability, symmetry, metallicity and band gap, magnetism, density, ICSD links. API version, query and SHA-256 are in the manifest. Snapshot taken 2026-11-06. |
| `data/candidates/mp-candidates.csv.gz` | The same rows plus an element-rule chemical family and a `known_status` from SuperCon (Stanev 2018) and 3DSC |
| `data/candidates/mp-sc-classifier-scores.csv.gz` | Every material's classifier probability and physics-model Tc |
| `docs/mp-sc-classifier-top200.csv`, `docs/mp-sc-classifier.json` | The ranked EPW-eligible list and the model report |
| `docs/mp-physics-top200.csv`, `docs/mp-physics-ranking.json` | Physics-model-only ranking, with its out-of-sample check |

Materials Project data is CC BY 4.0. Cite Jain et al., *APL Materials* 1, 011002 (2013).

## What is in it

| Count | Materials |
| --- | --- |
| 30,358 | All stable materials with ≤ 20 sites |
| 19,726 | Metallic in MP |
| 4,472 | Magnetic in MP |
| 16,774 | Have an ICSD entry, so made experimentally at least once |
| 13,992 | Theoretical only, never synthesized |
| 1,140 | Known superconductors (reported Tc > 0) |
| 1,104 | Tested and reported as non-superconducting (3DSC Tc = 0) |
| 28,114 | No match in SuperCon or 3DSC |
| 13,203 | EPW-eligible: untested, Other family, metallic, non-magnetic, not radioactive, ≤ 20 sites |

Known compounds are matched by atomic fractions, not by spelling. "Exact" means every fraction is within 0.0015. "Near" means the same elements with every fraction within 0.02. That is how MgB2 matches the reported B2Mg0.98 (38.4 K): SuperCon has no exact "MgB2" row. Materials Project has also renamed its IDs (`mp-1229062`-style to `mp-aaaaabza`-style), so these files use the new IDs.

## The two models, tested on real outcomes

| Model | Trained on | Out-of-sample test | Result |
| --- | --- | --- | --- |
| λ/ωlog physics model (`score_pool_physics.py`) | 1,057 JARVIS DFT calculations | Separate 1,062 known superconductors from 988 tested non-superconductors, excluding training formulas | **AUC 0.53**, about chance; Spearman with measured Tc 0.16 |
| Superconductor classifier (`train_mp_sc_classifier.py`) | 2,244 labelled MP materials: formula plus MP properties | 5-fold GroupKFold by element system | **AUC 0.80**; 90% of its top tenth are superconductors vs 51% overall |

The physics model looked reasonable against other DFT values (Tc Spearman 0.51), but it does not carry over to measured outcomes. Use the classifier to choose what to check, and EPW to check it.

Limits of the classifier:

- It predicts whether a material superconducts at all, not how high its Tc is.
- Negative labels are materials someone reported as tested and non-superconducting. That is a biased sample, so the model partly learns what people chose to test.
- "Untested" means no SuperCon or 3DSC match. A literature search is still needed before calling anything new.

## Current top candidates

The highest classifier scores among EPW-eligible materials are Pb, Hg, Tl and Bi intermetallics, plus Pt-group germanides: Y(GePt)2, NaHg3, Ca2(TlPb)3, BiPb3Au, Bi2PdPt, K2Hg7, LaGe3Ir. These neighborhoods hold many known low-Tc superconductors, which is a sanity check, but expect low Tc.

Where both models agree (classifier probability high and physics Tc ≥ 10 K), the list is Nb- and Re-based transition-metal alloys: Nb2MoOs, Nb6PdRh, Nb2ReRu, Nb6PdPt, Nb12IrAu3, ReOs, TaNbMo, V6GaGe. Almost all are MP "theoretical" ordered models of what are really disordered alloys. An EPW result on an ordered cell is an approximation to the real alloy.

## How to refresh

```bash
python scripts/fetch_mp_snapshot.py          # needs MP_API_KEY (40 s for all 30,358)
python scripts/build_mp_candidates.py        # needs data/raw from download_training_data.py
python scripts/train_mp_sc_classifier.py     # about 2 minutes
python scripts/score_pool_physics.py --candidates data/candidates/mp-candidates.csv.gz \
  --output docs/mp-physics-top200.csv --summary docs/mp-physics-ranking.json --top 200
```

The Materials Project API rejects Python's default `Python-urllib` user agent with HTTP 403, so the fetcher sends its own.
