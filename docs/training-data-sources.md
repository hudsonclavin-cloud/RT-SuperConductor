# Where more training data can come from: a verified survey, 2026-11-27

**Main idea:**
- **No source found here can lift the ranking to 0.9 on its own.** That
  includes the free and the paid ones.
- **The scores of 0.9 and above in published papers come mostly from random
  splits and easy negatives.** Under honest, held-out testing the best
  comparable result is about ours (0.78).
- **What can move the number:** cleaner and newer measured negatives,
  crystal structures, and a better-posed target. The survey also turned up
  ready-made lists of high-Tc candidates at ambient pressure, for Hazel.

**How it was made.** A research swarm, run at the owner's request:
- six finders, each searching a different kind of source;
- independent sceptical verifiers, three sources each;
- a completeness critic.

It found 79 candidate sources and 48 benchmark results. Records:
`docs/data-swarm/survey-2026-11-27.json`.

**What was and was not verified.**
- The first verification pass was capped at 36 sources (48 checks with the
  critic's additions). That left 27 unchecked, which I did not catch until the
  run ended.
- A second round is verifying those 27. Its results will be added below.

## What "0.9" means: results other groups report

| Study | Split | Inputs | Result |
| --- | --- | --- | --- |
| Konno et al., Phys. Rev. B (2021) | **Time held out** (trained on older entries) | Composition | **AUC 0.78** for Tc > 0; **0.94 for Tc > 10 K** |
| This project, ranking of record | **Group held out** | Composition | AUC 0.79 (non-cuprates) |
| Xiong et al. | Random vs. forward-in-time CV | Composition | R² 0.92 falls to 0.82 |
| Pogue et al. | Random vs. cluster held out | Composition | "Degrades significantly" |
| Trezza and Chiavazzo | Random | Composition | AUC 0.99, but against random Materials Project entries, which are easy negatives |
| ElementsClaw (Li et al., ICML 2026) | Random, with augmented structures | Structure | AUC 0.996 |
| S2SNet (Liu et al.) | Random | Structure | AUC 0.91; Magpie random forest 0.785 |
| Jung, Jung and Cole | Random | Composition | AUC 0.986 |

- [Likely] **Ours is already at the honest frontier for composition-only
  models.** The published 0.99s come from random splits, where near-identical
  compositions sit on both sides, or from easy negatives.
- [Likely] **The target matters as much as the model.** The same Konno model
  scores 0.78 for "superconducts at all" and 0.94 for "Tc above 10 K".
  - Our nulls mostly mean "not superconducting above about 1–2 K"
    (`docs/label-error-rate.md`). So the current task asks the model to
    separate a 3 K superconductor from a 1 K null, which composition alone
    may not decide.
  - For the room-temperature goal, "Tc above 10 K (or 20 K)" is the better
    question, and it can be scored honestly.
- **A trap to avoid.** Adding easy negatives, such as measured insulators,
  would raise the AUC without making the ranking any better at the metals it
  is used on.

## The four levers, and what the survey found

### 1. Cleaner labels, and newer measured negatives

| Source | What it adds | Access | Verdict |
| --- | --- | --- | --- |
| **OpenAlex full-text search** for phrases like "no superconductivity … down to" | 51,257 works since 2015 match; about 800 for 12 exact phrases. Most full texts are on arXiv | Open; workable here | **The best open route to hard negatives from 2015–2026, which nothing we hold covers.** A pilot of 200 papers first |
| MPDS / PAULING FILE | 27,734 superconductivity entries, including 3,046 "lowest temperature of investigation" (measured nulls), tied to specific phases | Commercial (ASM International) | Possibly the largest curated source we lack. **Check whether the NCSU library subscribes** |
| SuperCon-MTG cleaning logs (two spreadsheets) | Reasons recorded by hand for its zero-Tc rows; 152 of our nulls could be upgraded to "a person read the paper" | Open on GitHub; licence names only the CSV | Small, but cleans existing labels. Ask about the licence |
| NBS Technical Note 983 (Roberts 1978) | About 60 measured nulls not in NIMS by reference number | Free scanned PDF | An afternoon of transcription |
| ElementsClaw verified negatives | Up to 385 | Its host is blocked here | Low |

**Checked here, against a warning from the survey.** A verifier warned
that most of our nulls might be blank Tc values written as 0. They are not:
- 1,691 of the 1,957 non-cuprate nulls (86%) have a NIMS measured floor;
- only 22 are bare zeros (`docs/label-error-rate.md`).

The warning came from an older pipeline file.

### 2. More labelled compositions

| Source | What it adds | Access | Verdict |
| --- | --- | --- | --- |
| **AutoSuperCon** (Manium, Geisler, Hirschfeld, Coldea; arXiv:2611.15627) | 6,793 superconductors (5,843 bulk, ambient pressure), extracted from arXiv preprints up to March 2026; checked against NIMS (R² 0.96) | **Not released**: no data statement in v1 | **The biggest new positive set**, and it covers 2021–2026, where SuperCon2 stops. Perhaps 1,000–2,000 new compositions [Speculative]. **Email the authors** |
| SCDB (ISSP Hefei, arXiv:2611.29445) | 1,371 compositions, audited LLM extraction, no negatives | Registration | Mostly duplicates |
| NEMAD | — | Held | Already in use |

- At +0.03 AUC per doubling of training data, AutoSuperCon's likely new
  rows are worth about +0.005 to +0.01 at home.
- Their real value is as a new blind test set (2021–2026) that no model
  here has seen.

### 3. Crystal structures

| Source | What it adds | Access | Verdict |
| --- | --- | --- | --- |
| **Crystallography Open Database (COD)** | Open experimental CIFs | **Reachable from here** | Automated matching of the 3,345 compounds in `docs/perplexity/structures.csv` |
| Materials Project | Calculated structures, many of them derived from ICSD | Held, and the GitHub secret exists | Part of the same matching |
| MC3D (Materials Cloud) | 32,013 PBEsol-relaxed structures of experimentally known stoichiometric compounds, made with Quantum ESPRESSO | Its host is blocked here | Useful for Hazel inputs; needs a network allowance |
| MateriaViva-Open | 55 structure databases deduplicated into one index | Open | A cross-check |
| SuperBand, SuperCon3D, 3DSC_ICSD variants | SuperCon matched to structures | Open | Mostly duplicates of 3DSC |

### 4. Physics inputs: calculated electron-phonon coupling

| Source | What it adds | Access | Verdict |
| --- | --- | --- | --- |
| **BETE-NET** (Hennig group, npj Comput. Mater. 2025) | 806 Eliashberg spectral functions, 276 compositions we lack | Open, Apache-2.0, 89 MB | Its recipe (Quantum ESPRESSO, PBEsol, ONCV) is close to Hazel's, and it shares 340 compositions with Alexandria. **A method cross-check for Hazel** |
| Gao et al. 2025, 20,000+ metals | About 12,000 beyond our Alexandria set | Being verified in round 2 | — |
| BEE-NET (2026) | The public part duplicates Alexandria; the 741 new predictions are not released | — | Low |
| Geisler et al. 2026 | About 7,300 compounds with Eliashberg Tc and critical fields | Not released | Low |

[Certain] Calculated sets did not carry over to experiments in this
project's own test (`docs/physics-proxy2.md`). Their use is screening and
checking Hazel, not training the ranking.

## For the room-temperature goal

| Source | What it holds | Verdict |
| --- | --- | --- |
| **HTSC-2025** (Renmin, arXiv:2506.03837) | 140 predicted high-Tc superconductors at ambient pressure, with CIFs: the X2YH6, MXH3 and M3XH8 hydrides and B-C cage compounds. The highest is 160 K; none is confirmed by experiment | A ready list of templates and calibration targets for Hazel. No licence, so keep it under `data/raw/` |
| **Tokuyama et al.** (Kyoto, arXiv:2512.20…) | 2,277 hydride records (CC BY 4.0), including 148 at ambient pressure with calculated Tc of 40–120 K (the Mg2XH6 and perovskite-hydride families) | Candidates for Hazel, not ranking labels. 93% are above 1 GPa |

- Round 2 is checking the experimental tests of these predictions:
  - Mg2IrH6, and the Mg-Ir-H compounds actually made;
  - Mg4Pt3H6 and Mg2RhH6;
  - delithiated LiBC and SrB3C3;
  - GNoME's ambient-pressure hydrides.
- [Likely] The gap between prediction and experiment is the key fact here:
  the predicted 100 K+ hydrides at ambient pressure have so far been hard or
  impossible to make.

## What is not worth doing

- **Measured band gaps as negatives** (about 2,400 insulators). They are
  easy rows, far from the metals the ranking is used on, so they would
  inflate the AUC (see the trap above).
- **Hydride data under pressure in the ranking's training set.** It would
  push the model toward compounds that exist only under pressure.
- **Chasing random-split benchmark numbers.**

## Owner actions

1. **Email the AutoSuperCon authors** (corresponding author on
   arXiv:2611.15627) for the database and its licence. It is the biggest
   source of new measured superconductors, covering 2021–2026.
2. **Ask the NCSU library** whether it subscribes to ASM International's
   MPDS (PAULING FILE).
3. **Network allowances, if wanted:** `materialscloud.org` (MC3D structures,
   Materials Cloud archives) and `developer.damo-academy.com` (ElementsClaw
   labels). I can propose the change when it is needed.

## What I can do here, in order of value

1. **Structure matching** of the 3,345 compounds against COD and the
   Materials Project. No credits are needed.
2. **A pilot of full-text null mining**: OpenAlex finds the papers, arXiv
   provides the text, and the readers extract floors with verbatim quotes.
3. **A planned test of the "Tc above 10 K" target** on the existing rows, to
   see whether it is the better-posed question for the room-temperature
   goal.
4. **BETE-NET as a Hazel method cross-check.**
5. After round 2: a short list of ambient-pressure high-Tc candidates with
   published structures, for the Hazel queue after the benchmark.

## Round 2: the 27 sources round 1 left unverified (2026-11-27)

All 27 exist. None raises the ranking's AUC much. Nine matter for the
room-temperature goal. Records: `docs/data-swarm/survey-2026-11-27-round2.json`.

**Two corrections to round 1:**
- **The 20,000-metal calculated set (Gao et al.) is reachable from here.**
  - It is the Alexandria phonon and electron-phonon release of 2025.08.11:
    93 files, 5.06 GB, CC BY 4.0.
  - Each entry has λ, ωlog and Tc by McMillan, Allen-Dynes and Eliashberg.
  - The download needs the owner's approval because of its size.
- **BEE-NET's public files are about 12 GB of model weights**, not 60 MB. Its
  data duplicates what we hold.

**The central fact for the room-temperature goal** [Likely, from Gao et al.
and the hydride papers]:
- The calculated high-Tc compounds at ambient pressure are thermodynamically
  unstable.
- The ones that are stable have modest Tc.
- Where synthesis has been tried, it has failed or needed pressure.

| Compound | Calculated Tc at ambient (method) | Stability | Tried in a lab? |
| --- | --- | --- | --- |
| Mg2IrH6 | 59–175 K (Allen-Dynes to anisotropic Eliashberg) | 60 meV/atom above the hull | Yes, and failed: only insulating Mg2IrH5 and Mg2IrH7 formed |
| Mg2RhH6 | 45–59 K | 30 meV/atom above | **Made at 30–74 GPa: measured 18–29 K**. Lost below about 28 GPa |
| Li0.5BC (delithiated LiBC) | about 65 K | metastable | **Made: no superconductivity**. A calculated high Tc that failed |
| Mg4Pt3H6 | 0.9 K | on the hull | **Made: 2.9 K at ambient after a pressure quench** |
| LiZrH6Ru (GNoME) | 17–32 K | **on the hull** | Not reported |
| SrAuH3 | 132 K (Allen-Dynes) | stable above about 7 GPa only | No |
| KInH3 | 73 K | 77 meV/atom above; dynamical stability disputed | No |
| Li2AgH6 | 110–132 K | 319 meV/atom above; unstable in molecular-dynamics tests | No |
| CaB8C, KPbB6C6 (B-C cages) | 77–88 K | metastable, or unstable at 1 atm | No |
| SrB3C3 | — | — | **Made: about 20 K at 40 GPa** |

## A room-temperature track for Hazel, proposed after the benchmark

The current queue tests the pipeline on ordinary compounds. A second track
would point it at the chemistries that theory says could reach high Tc.

**Calibration first.** The method's error on hydrides and boron-carbon
frameworks needs to be known before any high-Tc number from it can be
trusted, so the first runs are cases with a measured answer:
1. **Mg2RhH6 at about 50 GPa:** measured 29 K, calculated about 60 K. 9 atoms.
2. **Ordered Li0.5BC:** calculated about 65 K, measured nothing. The
   false-positive check.
3. **SrB3C3 at 40 GPa:** measured about 20 K. A covalent framework, 14 atoms.

**Then the candidates,** cheapest and most checkable first:
1. **SrAuH3:** 5 atoms; CIF in HTSC-2025.
2. **KInH3:** 5 atoms; a dense phonon grid settles its disputed stability.
3. **LiZrH6Ru:** 9 atoms. The one candidate on the hull, so the most likely
   to be makeable.

**Limits** [Certain about the method, Speculative about the size of the
effect]:
- **Hydrogen needs anharmonic phonons.** The Hazel chain is harmonic, and
  published anharmonic corrections change hydride Tc by tens of percent.
- **Pressure runs need a new input.** They need pressure-relaxed cells,
  which the current packages do not include.
- **Expect upper bounds.** The calculated Tc from this chain should be read
  as an upper bound, as for the benchmark.
- **Results stay private.** Candidate results stay under `data/raw/hazel/`
  (AGENTS.md).
