# Physics-feature study completion report

Branch: codex/physics-features-deep. All six phases and the Phase 3.5 amendment were executed, with blocked scientific cases retained. No room-temperature superconductor was verified.

| Phase | Measured outcome |
|---|---|
| 1 — data and leakage audit | 5,773 reference CIFs processed; 0 parsing failures; target/matching metadata excluded from features. |
| 2 — paired ablations | Formula baseline cross-family MAE 10.4523 K versus mean predictor 16.1169 K; full tables and importance diagnostics in modeling.md. |
| 3 — structure models | Frozen/fine-tuned/scratch graph representations cross-family MAE 10.5140 / 11.4483 / 12.2198 K. CHGNet fine-tuning 4.546 CPU-hours; scratch ALIGNN 1.334. |
| 3.5 — physical checks | DOS/orbital data for 5,243 rows, bulk moduli 2,757; MP enrichment missing/quarantined for 14 legacy-reference rows; angles 1,361, buckling 231. All scalar groups cross-family MAE 10.4464 K. |
| 4 — constrained search | 36,902 substitutions considered; 4,338 accepted; 60 hypotheses retained (50 review-tier, 10 low-confidence). 28 ML relaxations/phonon calculations: 20 imaginary-mode signals and 8 without signals; 32 disorder blocks. Al false positives limit interpretation. |
| 5 — uncertainty/extrapolation | Empirical RF OOF above-training-max count 0; physics-derived 14/1,057, with only 1 corresponding DFT target actually above its fold maximum. Independent calibration helped some tasks and worsened others. |
| 6 — CI/reporting | Full integration run 41775076764 succeeded with 8 variants ×5 scopes on the 5,773-row cohort; nullable pressure and artifact reuse verified. 147 unit tests passed. CI baseline/all-group cross-family MAE 10.4351/10.4474 K; this is separately versioned from local results. |

## Calculations and unresolved cases

The project completed 28 ML-potential relaxations and Gamma/coarse-mesh phonon screens. These are not DFT calculations. New DFT/EPW/Bader/ELF/COHP/Wannier calculations completed: **0**. External compute access, approved pseudopotentials, convergence studies and expert method review remain necessary. The 32 disordered candidate rows require actual occupational ordering decisions; they were not rounded into invented structures. Four legacy MVC references returned unrelated DOS chemistry, and three had no DOS; all 14 associated rows remain unresolved for MP enrichment.

The Al control falsely triggered the ML instability threshold across tested settings. Consequently the 20 candidate flags are diagnostic signals and the 8 non-flags are not validated stability verdicts. All 60 hypotheses retain first-principles verification pending.

## Workflows and evidence

- Full integration: https://github.com/hudsonclavin-cloud/RT-SuperConductor/actions/runs/41775076764 . Source tables, version hashes and all CI metrics: deep-ci-integration.json.
- Primary MP collection: run41739776208, 2,708 references, 0 endpoint failures; 5,243 DOS fingerprints and orbital rows.
- Legacy-reference audit: run41744148063; returned numerical values were quarantined, not used.
- Earlier retrieval diagnostics: 41737358261, 41737943085, 41738416962, 41739223278. They exposed metadata, storage-key and NumPy compatibility issues; their failures remain documented.

## Model limitations

The final CI result has different library versions and within-family tied-group fold assignments from the local studies (cross-family folds are unchanged); compare variants within each run, not local graph models against the CI scalar baseline. CI baseline/all-group cross-family MAE is 10.4351/10.4474 K, mean 16.1169 K. Local paired values are 10.4523/10.4464 K. Neither study establishes an overall physical-feature advantage.

The empirical labels have incomplete pressure/sample provenance; the study is pressure-agnostic. New missing pressure stays unknown, and high-pressure synthesis is distinct from measurement pressure. Zero-Tc database conventions are not universal experimentally verified negatives. Many crystal/DFT descriptors refer to parent proxies rather than the measured doped sample. Graph training used disclosed short pilots and pretrained structures may overlap input structures. Feature importances are not causal mechanisms; extra descriptors did not materially improve global MAE. The EPC model applies to its stated conventional electron–phonon assumptions and simplified Tc expression, not a universal pairing mechanism. Its nonlinear extrapolations were mostly errors.

## Single most valuable next action

Have a condensed-matter mentor approve and run one converged conventional DFPT benchmark with matched structure, functional, pseudopotentials and Tc convention. The Ta JARVIS reference and explicit acceptance/failure decisions are in next-experiments.md. This supplies qualified physical feedback before enlarging the candidate search.

## Changed files

The following inventory includes scientific inputs and reports as well as code; raw datasets, model weights and large caches were not committed. Existing protected files were preserved.

- .github/workflows/continuous-training.yml
- README.md
- data/training/README.md
- docs/deep-al-control.json
- docs/deep-cache-verification.json
- docs/deep-candidate-inputs/0238530304f4bd3a818ac5a035c1abb826d1cc8858c7e622c61db402a1a9fb42.cif
- docs/deep-candidate-inputs/0461d546682dea44bec30c1257ce5d447b30a6f79b5acd83423acec8a08691db.cif
- docs/deep-candidate-inputs/06bf19bf3169b903fe13aabbdefca8bcc98db1de8bfe17dcc27125e405db72ff.cif
- docs/deep-candidate-inputs/0805f26c2553988bf4ff7c378bb345b6c34a9f1626e6ed99813cc9e7e11952db.cif
- docs/deep-candidate-inputs/08557315de8cf2382f11830c7a61c3e9a73a24e232f238fed4a479ca11129771.cif
- docs/deep-candidate-inputs/0d97d4c9c32edc3aa6401c040b5dbbe80ae9c72360873e995b7eec2d0fa1ee60.cif
- docs/deep-candidate-inputs/13495affbd37fa6e73f6891c27e9d1575e19157e7594fecb89e5aaa3fe9fda30.cif
- docs/deep-candidate-inputs/193bd79c2480c32528423b311a5917b01c1e6f442a9c22384b4ca2fe7e3fef68.cif
- docs/deep-candidate-inputs/23790761cb8a5e11a7f82bfa921362de1eed89ff656316818159136faeb87be8.cif
- docs/deep-candidate-inputs/282d59144451c0c9806f6c96b15b82513bd81469572548eb0d7cd6eeda00cbe1.cif
- docs/deep-candidate-inputs/34bbf7f6b647182a4aae8bb6e1b4cc29b8e752426f17a7cad703de96c36e3d61.cif
- docs/deep-candidate-inputs/37e8fa86f5b42cdd37c6a84ab194ebdc1763af019d63c67e0d74026c2282424e.cif
- docs/deep-candidate-inputs/3b1495cc2497ab56a969e6757b0fe6c883a57a65ec751e98b6ee43a4fbe92c57.cif
- docs/deep-candidate-inputs/3c67fcfa273ce567b6bf65843c3e47c2151a460e89540ec6d0fbaab0189f2fb7.cif
- docs/deep-candidate-inputs/3f56715e518c2a52da4a21a1c6c0dd35a28d0f8c54f8423de360ac078449e809.cif
- docs/deep-candidate-inputs/3fd4a34bbb36ec8ab3312b38b1bc26bd5d3986bcf8efdd098e2a21044d073e28.cif
- docs/deep-candidate-inputs/455f3c0119cec84a1b9f3206010cfc54b8995a6914c8acd3327ed4c096e88c05.cif
- docs/deep-candidate-inputs/45e60d1a2fdae6c504f4e581bb843aae08888cf39ec40bf21976a0604d0402d7.cif
- docs/deep-candidate-inputs/464c02d73fdcb43bd88c70fdadb25753000bbbbc5b3c9ff4ec605e53f89134ac.cif
- docs/deep-candidate-inputs/48ee62c4df1dcd6d6cf3531c422240bdef2267aabb4c844f3787a58b0789949d.cif
- docs/deep-candidate-inputs/4a54e4e5e6124156034ad7f70a4f538a39a2757721693ffd4d279e6b5a43f3cc.cif
- docs/deep-candidate-inputs/4da02eb487e6835e69e6e4a488fa1ca1366dbdbc7c161a4432afe4447693a424.cif
- docs/deep-candidate-inputs/51bc80e5ef74e0a94b7aaaeb7c0cc0c175875c30d0519d4302c5ae31504593c2.cif
- docs/deep-candidate-inputs/520186d7257a78eb073ba299c5a4f932b9e572b91a3c31af25525af4e419dc7a.cif
- docs/deep-candidate-inputs/549e6e747aee845ca8a08e2c47279f8dfcc26c691768c685b2b15c440ad85f68.cif
- docs/deep-candidate-inputs/5552f7e6b5f7b47277433296a91985fd579f6a520c513c07f47f41b0f9df1ea2.cif
- docs/deep-candidate-inputs/571ad25a94b1e268000f39837a19587286cf5152d80ba58fac4e34a0d5c4c039.cif
- docs/deep-candidate-inputs/57b3330259d86cd2c748a99e9c39f52414a08a27fd0b1a20eb19da92e671e38c.cif
- docs/deep-candidate-inputs/5aaa8e1ae2e89eb8026e666429f98ef4e8f99bc90c593a9b8a088578828f4be5.cif
- docs/deep-candidate-inputs/5b5cf9910a1aa24d38b3d8b164c7085c013623e4dc82f8d61b2e91f2c10f08e6.cif
- docs/deep-candidate-inputs/5d9ea6df361ff18f8b915c1fb685782e43af746e981427eee8fac48d93ff5002.cif
- docs/deep-candidate-inputs/6231e3d3f79ff598f58b941f2b226c30c550f25a4aca6c6884297de2dad57c46.cif
- docs/deep-candidate-inputs/7067f848a7dfc45ac039108170fc4b81f7ee41f96804893211d7a151c6d1d76e.cif
- docs/deep-candidate-inputs/74f101aa8e3baa5f89e60d14c449034567a9c96879d0a6e84611b8dd12cab5cc.cif
- docs/deep-candidate-inputs/75f0457c3029c268c9f96a33726091a190853147f87c90eb2228919df5309eac.cif
- docs/deep-candidate-inputs/7797bd0cfd9986986a0ad6849c7d216589042033247dcc4449c7cccf52ca9c99.cif
- docs/deep-candidate-inputs/79b9bfb2a27059a6d4af8ea5728fffc6a6471e84f7795f54f5b8b5c3e3cd238c.cif
- docs/deep-candidate-inputs/7e2366e4b4e0c2192e9caae1a76a5b4f9e97b47649f2bcf75cdd1328ef65cd34.cif
- docs/deep-candidate-inputs/7f5746a3c92c6c91263d5245a9aba3527d177e5ff35ec9172fb33d94e2145b0e.cif
- docs/deep-candidate-inputs/8542d70049b1d1e72841043e5719239e28b0729a747782390ea8fd82b1434dcf.cif
- docs/deep-candidate-inputs/8670c34151c69d50b7dbb2c2fa43faf256b0244b38fd3798f52e091d43b42170.cif
- docs/deep-candidate-inputs/88922d687534eb8530ba385c1086e6f1e23f453219d0756e501e1e79f681abc9.cif
- docs/deep-candidate-inputs/8ac7a00a5d5c6614073a98de480b5bc5b829c875fe45ef7328a7325693664e72.cif
- docs/deep-candidate-inputs/8daff2bdee415f63ed5e58a29fb6ffae69d5235709bc8c30aed568b8d1367e71.cif
- docs/deep-candidate-inputs/91e5aee50489159128908e0780ca99d86dc184e222105b18e06f2b6ab8980885.cif
- docs/deep-candidate-inputs/92865ae0362e002954328ccbfcf38c0190d60a3914a5df8408469b6ff8037598.cif
- docs/deep-candidate-inputs/9a65c417e15f8a8e8a4aa0ba9cb40b3553f9f3fa1ed4cdca4ad3303e5331f71d.cif
- docs/deep-candidate-inputs/9baf4bbdcb5c985b380ce7b783519bc6a5effee5f4bae378d0f4f40a1e076a5a.cif
- docs/deep-candidate-inputs/README.md
- docs/deep-candidate-inputs/a5020c5eaea282eaf16820da6cfdc2a36b86fd566769c44dfee18fedff29effa.cif
- docs/deep-candidate-inputs/a5b48dd2a3b415397a018802e60a6baadc87102404db4dff3fa266d4d1a36a09.cif
- docs/deep-candidate-inputs/a5d1a83e73fa8462c8c1575a62c58aa215b30aeb68408acb930a4451059c99d3.cif
- docs/deep-candidate-inputs/aa464d9991532305e493cda5f06c376f301ea3383e3df0a42794d7e12360f05a.cif
- docs/deep-candidate-inputs/ab172172e2c75ca06411851e4c570308ec9f9f4c8a75f0bc80f921b70b426714.cif
- docs/deep-candidate-inputs/ad78e78634f0f090bec8d608173a854302f4c4f9f6ab01efacfed41c552b0d27.cif
- docs/deep-candidate-inputs/ae01a56133c2eb60b64aafc164134c72e316c4b6d50351afb7273b60839e7260.cif
- docs/deep-candidate-inputs/ae0e949f803bb18f4a25d36445767d466144179ec3cf93e823c40929b303f750.cif
- docs/deep-candidate-inputs/al-control.cif
- docs/deep-candidate-inputs/b3f18705b821f5268937dbafdf7217cbe2e8d600aa2bdf42026d7673920ddd7d.cif
- docs/deep-candidate-inputs/b6c4aa346db25815dbc8de91748eeb4afd308279cab6bc8e5fab057982020e34.cif
- docs/deep-candidate-inputs/b7bea413867a5f9c721246a5c9d32f4208a20a4fcecf8efce150e6db62d7d785.cif
- docs/deep-candidate-inputs/b95ec28e68ff09384339352f61ddb8dd44bcb328832fb9757ba70d53148e5d9b.cif
- docs/deep-candidate-inputs/bd336fdaba2e54cacaba137bba66b8e88db571d4c7cdab9406517033157fb7c0.cif
- docs/deep-candidate-inputs/bd81dafa0a48af6011eee404b1211ed809ae4d6955932755c3b092caa0bb2b4a.cif
- docs/deep-candidate-inputs/bdf2b944953d49ad3374b658bb3da17d3bd3bf8cf556b21173b53e3f753a9612.cif
- docs/deep-candidate-inputs/c4fcf0cf2aaac61aed92e4f01ebb517fbc20374a58ff1837f5a3fd3613683e5b.cif
- docs/deep-candidate-inputs/c6307eaf81d4aaae46b1b74598382f829957062019dd0a9c4cd7746424b3f6c5.cif
- docs/deep-candidate-inputs/cb3a1788df6ebfe480d135544b75e070110933f60cb36458b3b6286c537e673c.cif
- docs/deep-candidate-inputs/cd0afeef659f114bb7f6ad75cc14532366bf7524778d2fefb1dbc105e8cde6ed.cif
- docs/deep-candidate-inputs/ce9a23c4a2b763507aeb20304aa6c0db0a23087d4439c9b6858634f035d46e3a.cif
- docs/deep-candidate-inputs/cfb1029629fb824eb1b5212ab34e6ee28e9c30e00a8bb87b33ed02cc6130f84c.cif
- docs/deep-candidate-inputs/d02c393e04866505b23111f2a80e5374edbab451280f7739181dcb46fa662c0d.cif
- docs/deep-candidate-inputs/d3c6ff89aa005456b4d6354d5e32fbb99f883c4bab418fc7e3bfd5d75ea6e1fa.cif
- docs/deep-candidate-inputs/d86827a190325f578a9846ea4ca1b5305f2654fd222a3e7b93297944714db739.cif
- docs/deep-candidate-inputs/dc764ad4f69577090b5683c6ebadb98c4cd1a022292a1a1f58918e2857c97455.cif
- docs/deep-candidate-inputs/ddf8faef05c1140ac27f8cbcae8e3a884e18dc9ad64f6d4c9409e692a260a5ca.cif
- docs/deep-candidate-inputs/de0231e20253c67d060a477591625c4e490164787d7427bac389240ebb1acb54.cif
- docs/deep-candidate-inputs/ded81237db988a6ed08ec9f12352d2044e77bc8650724d18eaf4fbd359633a5f.cif
- docs/deep-candidate-inputs/dfdeb271164995a50aa97fa39b2a360d2c619aac5517601a911da6efba9eabbd.cif
- docs/deep-candidate-inputs/e0a756c960f2d61dfbc2acc3d7f4ac5cf9024564cf76da28f4060094ef625156.cif
- docs/deep-candidate-inputs/e146e989f6d2fa434f88cfcb26a7c25b85098df103dc0af38e9da9601cb3ec7f.cif
- docs/deep-candidate-inputs/e1df6b308b7d1c9ebe9596fe808a78cdca260eed8222c3f9221e1f25edee8823.cif
- docs/deep-candidate-inputs/e47998d4ddffd3119fd1b00c3258747ca3ba3c6944dc2274c1e592adeed0f6fa.cif
- docs/deep-candidate-inputs/ec6e37201e75c5578592b5ab5ec0062012ff9517bb4166c98efde6c9c9c87ae0.cif
- docs/deep-candidate-inputs/ed3871bf1d479fd15441c0513bd65e7173eb04361d45194b10177cd699a34b1f.cif
- docs/deep-candidate-inputs/f0a9b2e66a11e25a58bdcff36e5ba57a9f4c4f6807f95b5e9cbb743394f9f166.cif
- docs/deep-candidate-inputs/f165274b7bb59eeaa291d5a87ddd796e02b1d179f8db546353e052b091065f79.cif
- docs/deep-candidate-inputs/f2315432d69bf469471b7c01bd9c6282aa22b1125b1f872815c87e87a82d318c.cif
- docs/deep-candidate-inputs/f9566607b3224db0373e0359b302af6168045efc803a2a4336f1468b136a5527.cif
- docs/deep-candidate-inputs/fd0f711380dbfe3999a9ceff81d1f49b3c2030770cf3261dfc48c254272c2538.cif
- docs/deep-chgnet-jarvis.json
- docs/deep-ci-integration.json
- docs/deep-combined-encoder.json
- docs/deep-dynamic-stability.json
- docs/deep-mp-diagnostic.json
- docs/deep-mp-full.json
- docs/deep-mp-smoke.json
- docs/deep-mp-spectrum-smoke.json
- docs/deep-mp-storage-diagnostic.json
- docs/deep-phase1-audit.json
- docs/deep-phase2-ablation.json
- docs/deep-phase3-attempts.json
- docs/deep-phase3-complete.json
- docs/deep-phase3-frozen-report.json
- docs/deep-phase35-ablation.json
- docs/deep-phase35-controls.json
- docs/deep-phase35-feature-coverage.json
- docs/deep-phase35-geometry.json
- docs/deep-phase35-mechanisms.json
- docs/deep-phase35-merge.json
- docs/deep-phase4-campaign.json
- docs/deep-reference-identity-audit.json
- docs/deep-scratch-alignn.json
- docs/deep-shortlists/ferrites.csv
- docs/deep-shortlists/other.csv
- docs/deep-shortlists/physics.csv
- docs/deep-uncertainty.json
- docs/faculty_research_summary.md
- docs/modeling.md
- docs/next-experiments.md
- docs/physics-features-deep-checklist.md
- docs/physics-features-deep-completion.md
- docs/physics-features-deep-one-page.md
- docs/uncertainty-calibration-physics.png
- docs/uncertainty-calibration-within-families.png
- docs/uncertainty-calibration.png
- pyproject.toml
- scripts/chemistry_policy.py
- scripts/deep_ablation.py
- scripts/deep_encoders.py
- scripts/deep_mechanisms.py
- scripts/deep_physics_checks.py
- scripts/deep_physics_features.py
- scripts/deep_uncertainty.py
- scripts/enrich_deep_mp.py
- scripts/enrich_physics_mp.py
- scripts/ingest_3dsc.py
- scripts/merge_physics_enrichment.py
- scripts/promotion_gate.py
- scripts/rank_candidates.py
- scripts/run_physics_search.py
- scripts/train_physics_targets.py
- src/rt_superconductor/harmonized.py
- tests/test_active_learning.py
- tests/test_deep_physics.py
- tests/test_deep_uncertainty.py
- tests/test_phase6_reporting.py
- tests/test_phonon_screen.py
- tests/test_physics_target_models.py
