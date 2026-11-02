# Faculty summary and candidate literature source check

Checked 2026-11-02. “Not found” means that the exact claim or composition was not located in the primary sources searched; it is not evidence that the material is impossible to make, unstable, or nonsuperconducting. These proposed compounds are 1111 iron oxypnictides (iron pnictides), not ferrites.

## Summary of the five factual claims

| Claim | Verdict | Primary source URL |
|---|---|---|
| Boeri, Dolgov and Golubov paper metadata, electron–phonon coupling and conclusion | VERIFIED | [APS, PRL 101, 026403 (2008)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.026403) |
| 3DSC paper title, first author, journal and year | VERIFIED | [Nature Scientific Data paper](https://www.nature.com/articles/s41597-023-02721-y) |
| UCI superconductivity dataset DOI 10.24432/C53P47 | VERIFIED | [UCI dataset record](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |
| SmFeAsO1−xFx has a reported Tc near 55 K; ambient-pressure qualifier | VERIFIED WITH QUALIFICATION | [Original report, arXiv:0804.2053](https://arxiv.org/abs/0804.2053) |
| VCA and explicit supercells are both used to model F substitution in LaFeAsO1−xFx | VERIFIED | [Yndurain, arXiv:1009.4909](https://arxiv.org/abs/1009.4909) |

## Part 1 — source verification

### 1. Boeri, Dolgov and Golubov (2008)

**VERIFIED.** The APS record gives L. Boeri, O. V. Dolgov and A. A. Golubov; title “Is LaFeAsO1−xFx an Electron-Phonon Superconductor?”; *Physical Review Letters* **101**, article 026403 (2008); DOI [10.1103/PhysRevLett.101.026403](https://doi.org/10.1103/PhysRevLett.101.026403).

Exact sentences from the APS abstract:

> “For pure LaFeAsO, the calculated electron-phonon coupling constant λ=0.21 and logarithmic-averaged frequency ωln=206 K give a maximum Tc of 0.8 K, using the standard Migdal-Eliashberg theory.”

> “Our results indicate that electron-phonon coupling is not sufficient to explain superconductivity in the whole family of Fe-As-based superconductors, probably due to the importance of strong-correlation effects.”

Source: [APS article page](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.026403). The numerical calculation is for **pure LaFeAsO**, while the paper title concerns F substitution. The conclusion is that conventional electron–phonon coupling alone is insufficient for the family, not that phonons have no role in any iron pnictide.

### 2. 3DSC bibliographic identity

**VERIFIED.** The publisher citation identifies the paper as “3DSC - a dataset of superconductors including crystal structures,” first author Timo Sommer, in *Scientific Data* 10, 816 (2023).

Exact publisher citation:

> “Sommer, T., Willa, R., Schmalian, J. & Friederich, P. 3DSC - a dataset of superconductors including crystal structures. Scientific Data 10, 816 (2023).”

Source: [Nature Scientific Data](https://www.nature.com/articles/s41597-023-02721-y), DOI [10.1038/s41597-023-02721-y](https://doi.org/10.1038/s41597-023-02721-y).

### 3. UCI dataset DOI

**VERIFIED.** Resolving [https://doi.org/10.24432/C53P47](https://doi.org/10.24432/C53P47) leads to the UCI Machine Learning Repository record for “Superconductivty Data.” The record’s citation text is:

> “Hamidieh, K. (2018). Superconductivty Data [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P47.”

Source: [UCI record](https://archive.ics.uci.edu/dataset/464/superconductivty+data). Its page describes 21,263 records and 81 features; it is a useful tabular benchmark, not a structure-resolved experimental evidence database.

### 4. SmFeAsO1−xFx near 55 K and pressure

**VERIFIED WITH QUALIFICATION.** The original report gives a 55.0 K resistive onset and 54.6 K Meissner transition. Its abstract states:

> “Here we report the superconductivity in the iron-based oxyarsenide Sm[O1-xFx]FeAs, with the onset resistivity transition temperature at 55.0 K and Meissner transition at 54.6 K.”

The article also says:

> “Here we report our new results on the samarium-arsenide Sm[O1-xFx]FeAs that synthesized under high pressure, with a resistivity onset Tc of 55.0 K”

Source: [arXiv:0804.2053](https://arxiv.org/abs/0804.2053), published as *Chinese Physics Letters* 25, 2215 (2008), DOI [10.1088/0256-307X/25/6/080](https://doi.org/10.1088/0256-307X/25/6/080). The reported sample was synthesized at about 6 GPa. The report does not clearly specify a pressure-cell Tc measurement, but the 55 K result should not be described as proof of synthesis at ambient pressure. The record supports “reported Tc about 55 K”; the stronger combined claim “made and measured at ambient pressure” is not established by this source.

### 5. Virtual-crystal approximation versus supercell

**VERIFIED.** Yndurain’s paper explicitly compares the two approaches for F-substituted LaFeAsO. Its abstract says:

> “We perform calculations using the virtual crystal approximation as well as supercell calculations with F substitutional impurity atoms. The results validate the virtual crystal approximation for the electronic structure near the Fermi level.”

Source: [arXiv:1009.4909](https://arxiv.org/abs/1009.4909), “Electron-phonon interaction in Fe-based superconductors: Coupling of magnetic moments with phonons in LaFeAsO1−xFx,” DOI [10.1209/0295-5075/94/37001](https://doi.org/10.1209/0295-5075/94/37001). This supports VCA as a computational approximation for the studied electronic structure; it does not establish that VCA captures every local-disorder, vacancy-ordering, or phonon effect.

## Part 2 — literature check of proposed compositions

The nearest-compound column reports evidence for a related composition, not a prediction for the proposed formula. “Exact composition made?” refers to a publication reporting that nominal composition; it does not imply phase purity or independent replication. Where a paper reports a nominal batch but no Tc for that batch, that distinction is explicit.

| Proposed composition | Exact composition reported made? | Closest located real compound and reported Tc | DOI / source | What the literature supports about feasibility |
|---|---|---|---|---|
| Sm0.8Th0.2FeAsO | **Nominal batch yes; Tc at x=0.20 not reported in the source inspected.** The series includes x=0.20. | Sm0.85Th0.15FeAsO: about 42 K; Sm0.7Th0.3FeAsO: 51.5 K. | [10.1103/PhysRevB.82.064517](https://doi.org/10.1103/PhysRevB.82.064517); [arXiv:1006.0126](https://arxiv.org/abs/1006.0126) | Th substitution in SmFeAsO is experimentally feasible and superconducting at neighboring concentrations. Do not assign those Tc values to x=0.20 or call the x=0.20 sample nonsuperconducting without its measurement.
| Tb0.89Th0.11FeAsO | **No exact x=0.11 report located.** | Tb0.9Th0.1FeAsO: onset Tc about 45 K; Tb0.8Th0.2FeAsO: about 52 K. | [10.1103/PhysRevB.78.132506](https://doi.org/10.1103/PhysRevB.78.132506); [arXiv:0806.1675](https://arxiv.org/abs/0806.1675) | Th substitution is demonstrated and superconducting in the Tb series. Exact x=0.11 remains unverified, not known impossible or unstable.
| Nd0.89Th0.11FeAsO | **No exact Th-substituted composition located.** | NdFeAsO0.85F0.15: Tc=43 K. | [10.1103/PhysRevB.79.094515](https://doi.org/10.1103/PhysRevB.79.094515) | This is an F-doped analogue, not evidence for Th solubility in NdFeAsO. No located source establishes the proposed Th composition as unstable, impossible, or nonsuperconducting.
| La0.8Th0.2FeAsO | **Yes.** | Exact composition La0.8Th0.2OFeAs: Tc about 30.3 K. | [10.1088/0953-8984/21/17/175705](https://doi.org/10.1088/0953-8984/21/17/175705); [arXiv:0902.3379](https://arxiv.org/abs/0902.3379) | Exact nominal composition was synthesized and reported superconducting. The paper reports an ambient-pressure material; this is not close to the project’s room-temperature target.
| La0.89Th0.11FeAsO | **No exact x=0.11 report located.** | La0.8Th0.2FeAsO: Tc about 30.3 K. | [10.1088/0953-8984/21/17/175705](https://doi.org/10.1088/0953-8984/21/17/175705); [arXiv:0902.3379](https://arxiv.org/abs/0902.3379) | Neighboring La–Th chemistry is synthesizable and superconducting, but this does not establish x=0.11. No evidence found that the exact composition is impossible or intrinsically nonsuperconducting.
| Ce0.8Th0.2FeAsO | **No exact Th-substituted composition located.** | CeFeAsO0.84F0.16: Tc about 41 K. | [10.1103/PhysRevLett.101.217002](https://doi.org/10.1103/PhysRevLett.101.217002) | F-doped CeFeAsO demonstrates superconductivity in the host family, not Th substitution. Exact composition feasibility and superconductivity remain unknown.
| Pr0.8Th0.2FeAsO | **No exact Th-substituted composition located.** | PrFeAsO1−xFx has optimum near x≈0.15 with Tc about 47 K. | [10.1103/PhysRevB.80.144517](https://doi.org/10.1103/PhysRevB.80.144517); [arXiv:0907.1308](https://arxiv.org/abs/0907.1308) | F-doped PrFeAsO is superconducting. This does not establish Th solubility; no evidence located that the proposed composition is impossible or nonsuperconducting.
| Eu0.8Th0.2FeAsO | **No exact Th-substituted composition located.** | EuFeAsO0.85F0.15: Tc about 11 K. | [10.1088/2053-1591/3/7/076001](https://doi.org/10.1088/2053-1591/3/7/076001); [arXiv:1506.02875](https://arxiv.org/abs/1506.02875) | The F-doped Eu analogue is reported superconducting, with a much lower Tc than other rare-earth members. Exact Eu–Th composition stability/solubility was not established in sources found.
| ThFeAsO | **No exact oxide report located.** | Related but chemically distinct ThFeAsN: superconductivity at 30 K. | [10.1021/jacs.6b00236](https://doi.org/10.1021/jacs.6b00236); [arXiv:1603.01501](https://arxiv.org/abs/1603.01501) | The nitride does not validate the oxide: replacing N with O changes chemistry and charge balance. No source located establishes ThFeAsO as synthesized, impossible, unstable, or superconducting.
| PrFeAsO0.84F0.16 | **Exact nominal x=0.16 not located in the phase-series report.** | PrFeAsO1−xFx: optimum x≈0.15, Tc about 47 K; reported measured compositions include values near x=0.14. | [10.1103/PhysRevB.80.144517](https://doi.org/10.1103/PhysRevB.80.144517); [arXiv:0907.1308](https://arxiv.org/abs/0907.1308) | The F-doped family is demonstrated superconducting. x=0.16 is plausible interpolation but is not an independently verified exact synthesis/Tc result from the source located.
| TbFeAsO0.89F0.11 | **No exact nominal x=0.11 report located; x=0.10 and 0.20 batches are reported.** | TbFeAsO0.9F0.1: susceptibility onset about 45.5 K and resistive midpoint about 43.8 K. | [10.1039/B808474B](https://doi.org/10.1039/B808474B); [arXiv:0806.0926](https://arxiv.org/abs/0806.0926) | F substitution is demonstrated and superconducting. The cited samples were made under high pressure (about 10 GPa); synthesis pressure should be tracked separately from Tc measurement pressure. Exact x=0.11 remains unverified.
| SmFeAsO0.84F0.16 | **Exact nominal x=0.16 not established by the original 43 K paper; its figure includes nominal x=0.15.** | SmFeAsO0.85F0.15: transition as high as 43 K. | [10.1038/nature07045](https://doi.org/10.1038/nature07045) | The nearby F-doped Sm phase is directly reported superconducting. The 55 K Sm report is a different paper with high-pressure synthesis; neither result should be silently mapped onto exact x=0.16.

### Pressure and Tc interpretation

For the candidate records, preserve at least three distinct fields: synthesis pressure, pressure during Tc measurement, and the pressure value/condition actually reported. High-pressure synthesis does not by itself mean high-pressure superconductivity; likewise, omission of a measurement pressure is not affirmative proof that pressure was ambient. The 55 K Sm report explicitly describes high-pressure synthesis. The Tb/F study also uses high-pressure synthesis. These facts matter to a project targeting operation near 1 atm or below a practical pressure ceiling.

### Search scope and limits

Primary publisher pages, arXiv records/manuscripts, and DOI-resolving records were opened directly. Google Scholar, arXiv, and publisher-site searches were used to locate reports for the exact formula strings and substitution families. Search-result snippets were not treated as evidence. Exact-composition “not found” outcomes should be revisited with a domain expert and broader citation chaining before experimental decisions; historical papers may use different nominal formula notation or report measured rather than nominal stoichiometry. Search coverage is not an exhaustive proof of absence.

