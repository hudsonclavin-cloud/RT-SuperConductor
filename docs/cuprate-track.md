# Cuprate track: building what is known about cuprates into the model

**Main idea:** nobody can calculate a cuprate's Tc from first principles, so cuprates have no counterpart of the EPW check. What exists is a set of empirical rules about which cuprates reach a high Tc. This track turns those rules into 20 features and tests them on cuprate families the model has not seen. Result as of 2026-11-11: the features help a little and consistently; the two-part "best Tc × doping share" model helps under one definition of family and not under the stricter one; the textbook doping dome applied directly does worse than no cuprate knowledge at all.

## What was built

| Piece | File |
| --- | --- |
| Feature extractor | `scripts/cuprate_features.py` |
| Feature table, 874 cuprates | `data/training/cuprate-features.csv` (report: `docs/cuprate-features.json`) |
| Corrections to wrongly matched parent structures | `data/training/cuprate-parent-overrides/` |
| Model comparison on held-out families | `scripts/eval_cuprate_models.py` → `docs/cuprate-model.json` |

## The rules, and the feature each became

| Rule from experiments | Source | Features |
| --- | --- | --- |
| Tc follows a dome in hole doping p: Tc/Tc,max = 1 − 82.6 (p − 0.16)² | Presland et al., Physica C 176, 95 (1991) | `hole_doping_p`, `abs_doping`, `doping_distance_from_optimum`, `presland_dome`, `electron_doped` |
| Tc,max rises from one CuO₂ plane to three, then falls | Long-standing empirical trend | `cuo2_layer_count`, `infinite_layer` |
| A more distant apical oxygen goes with a higher Tc,max | Pavarini et al., Phys. Rev. Lett. 87, 047003 (2001) | `apical_o_distance_min_a`, `apical_o_distance_mean_a`, `apical_o_count_mean`, `has_apical_oxygen` |
| Flat, well-ordered planes do better | Eisaki et al., Phys. Rev. B 69, 064512 (2004) | `plane_buckling_rms_a`, `in_plane_cu_o_mean_a`, `in_plane_cu_cu_mean_a`, `fractional_species_count`, `largest_minor_fraction`, `dopant_fraction_total` |

Doping comes from charge balance on the formula with nominal valences (La³⁺, Sr²⁺, O²⁻ and so on). `valence_ambiguous` marks the 405 rows that contain an element whose valence in cuprates is uncertain, such as Tl, Bi or Pb. The geometry comes from the crystal structure: a copper counts as part of a CuO₂ plane when its four nearest oxygens form a flat square and each of them bridges in a nearly straight line to another copper. That second condition separates planes from the Cu-O chains of YBa₂Cu₃O₇. On a textbook YBa₂Cu₃O₇ cell the extractor finds two planes, one apical oxygen at 2.29 Å, and correctly leaves the chain copper out.

## A data problem found on the way

3DSC attaches a Materials Project structure to each superconductor by matching formulas. For the largest cuprate family that match is the wrong crystal form.

| Family | Rows | What 3DSC matched | What the superconductors really are |
| --- | --- | --- | --- |
| La₂₋ₓ(Sr,Ba,Ca,…)ₓCuO₄ | 172 | La₂CuO₄ in the T′ form: no apical oxygen, Cu-O 2.00 Å | The T form: two apical oxygens at 2.46 Å, Cu-O 1.91 Å |
| Sr₁₋ₓLnₓCuO₂ | 17 | SrCuO₂ built from double Cu-O chains, with no CuO₂ plane | The infinite-layer form: CuO₂ planes separated only by Sr |

Materials Project lists the T′ form as the lowest-energy La₂CuO₄, which is why it was picked. `data/training/cuprate-parent-overrides/rules.json` replaces the structure for these 189 rows. The rules look only at the formula, never at Tc: Ce-substituted La₂CuO₄, which really is T′, keeps its matched structure, and SrCuO₂ rows with no substituent are left alone because the formula does not say which form was made. Smaller mismatches of the same kind may remain among the other 159 parent structures.

## The test

Each model is trained on some cuprate families and scored on others, 5 folds, 3 reshuffles. Two definitions of "family" are scored:

- **Parent** (main result): rows that share a parent structure. 161 families; the largest has 189 rows.
- **Strict**: rows linked by a shared parent structure or a shared set of elements, taken transitively. 93 families, but the links chain together and one group holds 394 of the 874 rows, so its folds are uneven. It is the harshest check available.

All models are random forests with the same settings. "Formula" is the project's existing 157 formula features.

### Parent families

| Model | Tc error (K) | Tc error, superconducting rows (K) | Rank correlation | Error in a family's best Tc (K) | Rank correlation of families by best Tc |
| --- | --- | --- | --- | --- | --- |
| Always guess the median | 30.3 | 32.8 | – | 30.0 | – |
| Formula features (current approach) | 23.1 | 27.4 | 0.46 | 24.8 | 0.63 |
| Cuprate features only | 21.6 | 21.6 | 0.53 | 19.6 | 0.59 |
| Formula + cuprate features | 22.8 | 27.0 | 0.55 | 24.6 | 0.67 |
| Two-part: best Tc from structure × learned doping share | **21.4** | 23.6 | 0.51 | **17.0** | 0.61 |
| Two-part, with formula features in the best-Tc model | 23.1 | 28.7 | 0.51 | 23.3 | 0.67 |
| Two-part with the fixed Presland dome | 27.6 | 39.6 | 0.36 | 23.9 | 0.37 |

### Strict families

| Model | Tc error (K) | Tc error, superconducting rows (K) | Rank correlation | Error in a family's best Tc (K) | Rank correlation of families by best Tc |
| --- | --- | --- | --- | --- | --- |
| Always guess the median | 32.4 | 35.5 | – | 39.3 | – |
| Formula features (current approach) | 24.1 | 25.8 | 0.54 | 18.2 | 0.63 |
| Cuprate features only | **21.9** | **21.1** | 0.58 | 19.3 | 0.59 |
| Formula + cuprate features | 23.3 | 25.1 | 0.58 | 17.6 | 0.67 |
| Two-part: best Tc from structure × learned doping share | 23.6 | 28.2 | 0.53 | 20.3 | 0.65 |
| Two-part, with formula features in the best-Tc model | 25.9 | 32.4 | 0.52 | 17.4 | 0.67 |
| Two-part with the fixed Presland dome | 29.5 | 41.8 | 0.27 | 23.0 | 0.57 |

Fold-to-fold spread is 3 to 6 K, so a difference of 1 or 2 K in one table is not much on its own. The paired comparisons in `docs/cuprate-model.json` count how often each model beats the formula model on the same fold.

## What the results say

1. **The cuprate features carry real information.** Twenty features do better than 157 formula features on unseen families under both definitions: 1.5 K lower error with parent families and 2.2 K with strict ones, in 3 of 3 reshuffles each time. The gain is largest where it matters, on rows that superconduct: 5.8 K and 4.8 K.
2. **The two-part model is not established.** With parent families it has the lowest error of all and gets a family's best Tc right to 17 K against 25 K for the formula model (better in 3 of 3 reshuffles, 10 of 15 folds). With strict families that advantage is gone: it is 2 K worse than the formula model on a family's best Tc and on superconducting rows. A method that wins under only one way of splitting has not earned a place in the main pipeline.
3. **The textbook dome, applied directly, hurts.** Multiplying by the fixed Presland dome is worse than using no cuprate knowledge: 39.6 K error on superconducting rows against 27.4 K. The dome is not wrong; the doping estimate is too crude. A formula-level average puts YBa₂Cu₃O₇ at p = 0.33, where the dome says Tc = 0, because a third of its copper is in chains. Letting a forest learn the doping share from the same inputs works far better than the fixed formula.
4. **No model got better at ranking families.** For finding better materials the useful question is which unseen family has the highest best-Tc. Every model scores between 0.59 and 0.67 there, and the differences are within the noise.
5. **What the cuprate-only forest leans on:** the apical-oxygen distance first (shuffling it adds 15.7 K of error), plane buckling second (10.3 K), then the doping features (2 to 4 K each). That matches the published picture, with a caution: this ranking is measured on the training rows, and a bond length can also work as a fingerprint of a family.
6. **Fixing the two mismatched parent structures changed the scores by less than the noise.** It is still the right structure to use.

## Limits

- The structures are parent compounds with doping added artificially, so the geometry never reflects the doped material.
- Oxygen contents in the source formulas are often nominal, and they decide the doping estimate.
- 874 rows, of which 576 superconduct, in 161 families. That is small, and the families are very unequal in size.
- These are correlations. A feature that predicts well has not been shown to cause anything.

## Not done

- **A check on Hazel.** The QE environment includes Wannier90, which can calculate the hopping ratio t′/t that Pavarini's rule is really about, for any parent structure. Cuprate cells are small, so this is much cheaper than EPW. It would check the descriptor, not Tc.
- **Plane-only doping.** Estimating how much of the charge sits on the planes, as opposed to chains or reservoir layers, is the most direct way to make the dome usable.
- **Wiring into the ranking.** Nothing here changes what the pipeline ranks yet. The cuprate-only model is a candidate to replace the formula model for the cuprate family; the evidence is a modest, consistent gain.

## Rebuild

```bash
python scripts/cuprate_features.py
```

```bash
python scripts/eval_cuprate_models.py
```

The first needs the 3DSC table and structure files (`python scripts/download_training_data.py --all-cifs`); the second needs only the committed feature table and takes about three minutes.
