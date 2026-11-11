# GitHub Actions: what to run, and with which parameters

**Main idea:** almost everything runs by itself when the files it depends on change. As of 2026-11-11 only two workflows are worth starting by hand, and one option should be left off until a decision is made about the DOS gate.

To start a workflow by hand: open the repository's **Actions** tab, pick the workflow in the left column, press **Run workflow**, leave the branch on `main`, fill in the parameters below and press the green button.

## The short version

| You just did this | Run this | Parameters |
| --- | --- | --- |
| Added rows to `data/training/negative_evidence.csv` or set a decision in `data/training/label_audit_queue.csv` | Nothing. **Superconductor classifier** starts when the change reaches `main`, and on the pull request | none |
| Want to see whether another model now beats gradient boosting (after about 20 or more new labels) | **Superconductor classifier** | `model`: `gradient_boosting`; `run_benchmark`: on; `refresh_mp_snapshot`: off |
| Promoted a Hazel result into `data/training/hazel-epw-records.json` | Nothing. **EPW loop** starts when the change reaches `main` | none (the workflow has no parameters) |
| Want the weekly training run now | **Continuous AI training** | leave every box empty or off |
| Anything else | Nothing | |

## Every workflow

| Workflow | Starts by itself | Parameters | What to know |
| --- | --- | --- | --- |
| **Unit tests** | Every pull request and every push to `main` | none | Never needs a manual run |
| **Superconductor classifier - rebuild labels and retrain** (`sc-classifier.yml`, new) | Push to `main` or a pull request that changes the label files, the snapshot or the classifier scripts | `model` (default `gradient_boosting`), `run_benchmark` (default off), `refresh_mp_snapshot` (default off) | About 4 minutes (3.5 on its first run). Rebuilds labels from all sources, retrains, and compares the result with the committed ranking in the run summary. Results are an artifact; nothing is committed |
| **EPW loop - rescore candidates after new results** (`epw-loop.yml`) | Push to `main` that changes Hazel records, a candidate list or the scoring scripts | none | A few minutes. Scores the first candidate list that has rows (see below) |
| **Continuous AI training** (`continuous-training.yml`) | Sundays 03:17 UTC, and pushes to `main` that touch its inputs | eight, described below | 25 to 60 minutes for the plain run |
| **Physics-target model, graph benchmark, and Materials Project enrichment** | Push to `main` that changes the λ/ωlog trainer or its inputs | none | Up to 4 hours. It will start by itself after the 2026-11-11 trainer change |
| **Promote pressure-aware surrogate** | Push to `main` that changes the Tc model's inputs | none | Nothing to do by hand |
| **Structure and density inputs** | Push to `main` that changes structure inputs | none | Nothing to do by hand |
| **Low-budget materials screen** | Never | `minimum_tc_k` (default `70`), `mp_limit` (default `5`) | An early screen, kept for reference. No reason to run it now |

## Continuous AI training: the eight parameters

Leaving everything empty or off gives the normal training run. Each option below switches the run to a different job.

| Parameter | Default | What it does | Use it now? |
| --- | --- | --- | --- |
| `family` | empty | Adds a within-family shortlist for one chemical family, for example `Ferrites` | Optional. Costs a few minutes |
| `build_untested_pool` | off | Queries Materials Project and rebuilds the untested-pool and EPW-eligible lists | **No.** See the warning below |
| `mp_snapshot_run_id` | empty | Rebuilds the pool offline from an earlier run's saved Materials Project query | No. Same gate, same empty result |
| `deep_ablation` | off | Paired physics-feature ablation over every 3DSC structure | No. Up to 6 hours, and structure features added at most 0.006 AUC in `docs/sc-feature-ablation.md` |
| `deep_mp_enrichment` | off | Fetches DOS, orbital fractions and bulk moduli for reference materials | No. Not needed for the current plan. It reads DOS files the same way as the pool builder (below); it may still work for older reference materials, but that has not been rechecked since the pool builder's failure |
| `deep_mp_limit` | `100` | How many reference materials `deep_mp_enrichment` queries; `0` means all | Only with `deep_mp_enrichment` |
| `deep_mp_ids` | empty | A comma-separated list of Materials Project IDs for `deep_mp_enrichment` | Only with `deep_mp_enrichment` |
| `deep_mp_reference_run` | empty | Reuses the properties a previous deep run fetched | Only with a deep run |

### Warning: `build_untested_pool`

Two reasons to leave it off for now.

1. **It commits to the branch it runs on.** Its last step runs `git commit` and `git push origin HEAD:<branch>`. Started on `main`, it pushes straight to `main` with no pull request.
2. **Its required DOS gate cannot pass today.** The gate needs a density-of-states curve for each candidate. On 2026-11-09 it asked for 149 curves and received none, so the lists came out empty (`docs/untested-pool-summary.md`). The files it asks for are not in Materials Project's public bucket: a direct check on 2026-11-11 found the bucket's DOS files named by old-style task numbers (for example `dos/mp-1000000.json.gz`, last changed in 2023), and none of the five example keys from the failed run exists under either its new or old name. Only 231 of 23,882 candidates had any DOS task reference to begin with. No candidate was ever compared with the 0.1 states/eV/atom threshold, so the empty list says nothing about the materials.

The decision that is open: keep DOS as a required gate, which gives no candidates until the data is reachable, or make it an optional extra column, as it was in the lists archived on 2026-11-09. Until then the EPW loop uses the classifier's ranking, described next.

## Which list the EPW loop scores

`scripts/score_pool_physics.py` takes the first of these that has rows:

1. `docs/epw-eligible-shortlist.csv`, the strict list (empty since 2026-11-09)
2. `docs/untested-pool-shortlist.csv` (empty since 2026-11-09)
3. `docs/mp-sc-classifier-top200.csv`, the classifier's top 200 EPW-eligible materials

Rows from the classifier's list keep the classifier's order. The λ, ωlog and Tc columns from the physics model are attached as a second opinion and do not change the order, because that model does not separate real superconductors from tested non-superconductors (AUC 0.53). The run summary names the list it used and why.

## One thing to check after the next push

The **Physics-target model** workflow prints a line "Physics-derived Tc MAE". Before 2026-11-11 the trainer fitted λ directly and that number was about 5.9 K. It now fits log λ, and the same held-out test on the committed table gives 2.71 K (guessing the median gives 2.99 K). If the next run still prints a number near 5.9 K, the change did not take effect and should be looked at.
