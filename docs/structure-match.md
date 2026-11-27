# Crystal structures for compounds the models see only as formulas

**Main idea:** the 3,345 ordered non-cuprate compounds in the table with no
3DSC structure were matched automatically to the Crystallography Open
Database (COD) and to the Materials Project data held here. Independent
checks of random samples measure how often a match is right.
- **Direct matches are reliable.**
- **Matches through a parent compound** are right about two times in three.
- **Matches through a pure element** are not reliable enough to use.

Built 2026-11-27 at the owner's request, after Perplexity Computer ran out
of credits.
- **Script:** `scripts/structure_match.py`.
- **Per-compound table:** `docs/structure-match/summary.csv`.
- **Counts:** `docs/structure-match/summary.json`.
- **Licences:** COD is CC0; the Materials Project is CC BY 4.0.

## Method

1. **COD:** one query per element set (1,378 sets, then 402 more for
   parents), cached in `data/raw/cod/`.
2. **Direct match:** an entry with the same elements, every atomic fraction
   within 0.02.
3. **Parent match**, for compounds with no direct match:
   - one element of at most 0.20 atomic fraction is merged into another
     element of the same chemical class (rare earth with rare earth, and so
     on), as a substituent on the host's site: (Dy,Ho)Ni2B2C → DyNi2B2C;
   - the merged composition is then matched the same way.
4. **Polymorphs.** Among the matching COD entries, the polymorph of the
   Materials Project ground state is preferred. That ground state is its
   lowest-energy entry citing ICSD records. This rule was added after
   round 1 of the checks.
5. **Confidence:**
   - **high:** a direct match;
   - **medium:** a parent that is a compound;
   - **low:** a parent that is a pure element, a host at 85% or more, or an
     ambiguous parent, meaning the series' other end member has a different
     structure.

## Counts (revised rules)

| Status | Compounds |
| --- | --- |
| Experimental structure, direct (COD 265, Materials Project via ICSD 22) | 287 |
| Experimental parent (COD 802, Materials Project 187) | 989 |
| Calculated structure only | 19 |
| Nothing found | 2,050 |

- **By confidence:** high 287, medium 613, low 376.
- **Why so many have nothing:** COD is thin on intermetallics, which sit
  mostly in the commercial ICSD and Pearson databases. 393 of the 3,345 are
  also dilute alloys with no single ordered structure.

## How often a match is right: round 1 (50 random matches, before the revised rules)

Independent checkers read the literature for each match. Records:
`docs/structure-match/verification-round1.json`.

| Confidence | Right, or the same structure type | 95% interval |
| --- | --- | --- |
| High | 15 of 15 | 0.80–1.00 |
| Medium | 17 of 25 | 0.48–0.83 |
| Low | 4 of 10 | 0.17–0.69 |

**What went wrong, and what was changed:**
1. **Hypothetical structures in COD.** fcc Cr, Ta, Nb and Zr come from one
   1993 theory paper. COD does not mark them.
   - Changed: the Materials Project ground state now picks the polymorph.
   - In round 1's sample this fixed Cr–Re, Ta–Hf and Nb–Hf, which now get
     bcc.
2. **A high-pressure or minority polymorph picked.** Mo3OS3 got a 13 GPa form
   of Mo3S4 instead of the Chevrel phase. The same change covers this,
   where the Materials Project has the ground state.
3. **The wrong end member.** (Ir,Rh)Se2 got orthorhombic IrSe2, but the
   series is pyrite like RhSe2. (W,Nb)C got hexagonal WC, but the quenched
   solid solution is cubic.
   - Changed: such parents are flagged **ambiguous** and graded low.
   - The flag is noisy. In round 1's sample it demoted 6 right matches, and
     it misses series whose other end member is not in the local Materials
     Project file, a 30,000-entry subset.

These rules were shaped by round 1's 50 matches, so their accuracy is
measured on a fresh sample (round 2, below).

## Use

- **As model inputs:** high and medium only, 900 compounds, with low and
  none treated as missing.
- **As Hazel inputs** (structures for calculations): high only, after a look
  at the specific entry.
