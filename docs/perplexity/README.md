# Perplexity Computer tasks: crystal structures and an independent label reader

**Main idea:** two research jobs for Perplexity Computer, each with an input
file, a prompt to paste and a fixed output format, so the results can be
imported and checked here.

| Job | Input | What it gives the project |
| --- | --- | --- |
| **A. Crystal structures** | `structures.csv`: 3,345 compounds in 34 batches of 100 | Structures for compounds the models now see only as formulas. That is one of the four levers on the AUC ceiling |
| **B. Label reader** | `label-reader-01.csv`: 69 compounds | An independent first reading of the label audit's strongest leads. That is reader one of the two the correction rule requires |

**Rules that keep the results usable:**
- **The inputs are blind.** They carry no label, Tc, model score or reason
  for being listed, so the reader cannot be steered.
- **No locked rows.** Locked compositions are excluded, and
  `scripts/perplexity_batches.py` refuses to write a file that contains one.
- **What happens to results.** Send each result file back (upload it in chat,
  or put it in `data/raw/perplexity/`). Then:
  - **Structures** are checked against COD and the Materials Project before
    they are used. They enter training only through a planned learning cycle.
  - **Label readings** count as one reader. A second independent reader and a
    sceptical verifier still follow before any label changes.
- **One batch per task.** Run job A one batch at a time: batch 1, then 2,
  and so on. Upload `structures.csv` once and say which batch.

Regenerate the inputs with `python scripts/perplexity_batches.py`.

---

## Job A: paste this, with `structures.csv` attached

```text
ROLE: You are a careful crystallography research assistant. Accuracy matters more than coverage.

TASK: The attached file structures.csv lists inorganic compounds (columns: batch, id, formula,
composition_key). Work on the rows where batch = 1. For each compound, find its EXPERIMENTALLY
determined crystal structure and report it in the CSV format below.

The formula column lists elements alphabetically (for example B6C3DyHo2Ni6). When you search,
also try the conventional ordering and name (for example (Dy,Ho)Ni2B2C, a borocarbide). The
composition_key gives atomic fractions; a reported compound matches if every element is within
0.02 atomic fraction.

WHERE TO LOOK, in this order:
1. Crystallography Open Database (crystallography.net): open CIFs.
2. The primary literature: search "<compound> crystal structure", "<compound> X-ray diffraction",
   "<compound> Rietveld", "<structure type> <elements>".
3. Materials Project (next-gen.materialsproject.org): note the mp-id. Say whether MP lists an
   ICSD source for it; MP structures are DFT-relaxed, so label them calculated unless MP cites
   an experimental (ICSD) source.
4. SpringerMaterials / Pearson's Crystal Data / ICSD if you can access them: give identifiers
   and parameters, not CIF text.
5. OQMD or AFLOW only as a last resort, labelled calculated.

RULES:
1. Evidence or nothing. Every row must cite a DOI or URL you actually opened. If you cannot find
   a structure, write found = no. Never infer a structure from a similar compound, a structural
   family, or memory.
2. Polymorphs: one row per reported polymorph, with when it is stable in phase_note (for example
   "high-temperature phase above 900 K", "metastable, quenched", "thin film only").
3. Doped or off-stoichiometric compounds: if only the parent compound's structure is reported,
   give the parent's structure with found = parent only, and name the parent in notes.
4. Experimental beats calculated. structure_source is experimental only for single-crystal XRD,
   powder XRD or neutron diffraction from a real sample.
5. Do not report superconductivity or Tc. This is a structure task only.
6. Keep each id exactly as given. Do not merge, split or reorder ids except for polymorph rows.
7. If you run out of time, return what you have and list the ids you did not reach.

OUTPUT: a CSV file named structures-results-batch-01.csv with exactly these columns:
id, formula, found (yes / no / parent only), match (exact / within 0.02 / parent),
phase_note, structure_type (prototype, e.g. Cu3Au, AlB2, ThCr2Si2), pearson_symbol,
space_group_symbol, space_group_number, a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg, z,
structure_source (experimental / calculated), method (e.g. powder XRD, single-crystal XRD,
neutron, DFT), measurement_temperature_K, icsd_code, cod_id, mp_id, source_doi, source_url,
confidence (high / medium / low), notes

ALSO:
- If COD has CIFs for any of these, put them in a zip named structures-cifs-batch-01.zip, with
  file names <id>_<cod_id>.cif.
- Before you return, re-open the sources for 5 random rows that have found = yes, and confirm the
  space group and lattice parameters match what you wrote. Report that check and any fixes.
- End with a summary: rows done, found yes / no / parent only, and any ids not reached.
```

For the next batch, change the three "1"s: `batch = 2`,
`structures-results-batch-02.csv` and `structures-cifs-batch-02.zip`.

---

## Job B: paste this, with `label-reader-01.csv` attached

```text
ROLE: You are an independent, sceptical literature reader for a superconductivity dataset.
You are deliberately not told what any database says about these compounds. Do not try to
guess it; read the primary papers.

TASK: The attached file label-reader-01.csv lists 69 compounds (columns: id, composition_key,
formula). For each, determine from the PRIMARY literature whether BULK superconductivity was
observed AT AMBIENT PRESSURE in a sample of this composition.

WHAT COUNTS:
- Bulk: a polycrystal, single crystal or bulk powder. Not a thin film, interface, surface,
  filamentary or trace (minority-phase) signal. Say which.
- Ambient pressure: no applied pressure. A high-pressure result goes in its own row with
  pressure_gpa filled.
- Composition: the measured sample is within 0.15 atomic percent of the listed composition on
  every element, or the paper measured a single-phase series that brackets it within 2 atomic
  percent. The formula column lists elements alphabetically; the composition_key gives atomic
  fractions.
- A "no" counts only down to the lowest temperature measured. Record that temperature.

EVIDENCE:
- Primary sources only: the paper that made and measured the sample. Databases (SuperCon,
  NEMAD, Materials Project) and reviews are leads to the primary paper, never evidence on their
  own.
- Quote at most two sentences of the passage, table or figure caption that settles it.
- If papers disagree, give one row per paper.
- If you could read only the abstract, say so in access and lower the confidence.
- If you find nothing, write superconductivity_observed = unclear and say what you searched.

OUTPUT: a CSV file named label-reader-results-01.csv with exactly these columns:
id, formula, composition_as_reported, composition_match (within 0.15 at.% / bracketing series
within 2 at.% / no), phase_or_sample, sample_form (polycrystal / single crystal / powder /
thin film / other), superconductivity_observed (yes / no / unclear), tc_k,
lowest_measured_temperature_k, pressure_gpa, applied_field_t, measurement_method (resistivity /
magnetization / specific heat / other), synthesis_conditions, publication_doi,
page_table_or_figure, evidence_quote, access (full text / abstract only), confidence
(high / medium / low), notes

End with a summary: yes / no / unclear counts, and any ids you could not finish.
```

---

## What comes back, and what happens next

| Result | Here | Then |
| --- | --- | --- |
| `structures-results-batch-NN.csv` (and CIFs) | Each row checked against COD and the Materials Project; parameters compared; anything that does not match is set aside | Structure features for a planned learning cycle, judged on the development rows as usual |
| `label-reader-results-01.csv` | Rows that disagree with the table go to a second independent reader and a sceptical verifier | Confirmed corrections go into `data/training/negative_evidence.csv`; checked rows go into `data/training/label-audit-reviewed.csv` |

---

## Without Perplexity Computer (added 2026-11-27)

Perplexity Computer ran out of credits after very little work. The same jobs
work in smaller pieces with **ChatGPT Pro deep research** (about 20
compounds per run) or **plain Perplexity** (about 5 per question). Every
label-reader file has a `batch` column of 20 rows:

| File | Rows | Batches of 20 | Purpose |
| --- | --- | --- | --- |
| `label-reader-02.csv` | 100 | 5 | **Do this first.** A random sample to measure the label error rate (`docs/label-error-rate.md`) |
| `label-reader-01.csv` | 69 | 4 | The label audit's strongest leads |

**Crystal structures (job A) are better done here than by a chat tool.**
- Most of the job is matching compositions against the Crystallography Open
  Database and the Materials Project. That is mechanical, and both can be
  reached from this project's machine.
- A chat tool is only needed for the rows that matching cannot settle, such
  as which polymorph was measured.

### ChatGPT Pro deep research: attach the CSV and paste this

Change the batch number and file name for each run.

```text
You are an independent, sceptical literature reader for a superconductivity dataset. You are
deliberately not told what any database says about these compounds; do not try to guess it.

Attached: label-reader-02.csv (columns: batch, id, composition_key, formula). Work ONLY on the
rows with batch = 1 (20 compounds). The formula lists elements alphabetically; the
composition_key gives atomic fractions. Search under the conventional name too (for example
CeGe0.03Pt3Si0.97 is Ge-doped CePt3Si).

For each compound, find from the PRIMARY literature (the paper that made and measured the
sample) whether BULK superconductivity was observed AT AMBIENT PRESSURE.
- Bulk means polycrystal, single crystal or bulk powder: not a thin film, interface, surface,
  filamentary or trace signal.
- The sample must be within 0.15 atomic percent of the composition on every element, or the
  paper must measure a single-phase series that brackets it within 2 atomic percent.
- A "no" counts only down to the lowest temperature measured; record that temperature.
- Databases (SuperCon, NEMAD, Materials Project) and reviews are leads to the primary paper,
  never evidence on their own.
- Old papers (Physical Review, J. Less-Common Metals, 1950s-1980s) matter here; try hard to
  reach them, including through citing papers that quote their results.
- If papers disagree, give one row per paper. If you find nothing, say unclear and say what you
  searched.

Return ONE CSV code block with exactly these columns:
id, formula, composition_as_reported, composition_match (within 0.15 at.% / bracketing series
within 2 at.% / no), sample_form (polycrystal / single crystal / powder / thin film / other),
superconductivity_observed (yes / no / unclear), tc_k, lowest_measured_temperature_k,
pressure_gpa, measurement_method, publication_doi, page_table_or_figure, evidence_quote (at
most two sentences), access (full text / abstract only), confidence (high / medium / low), notes

After the CSV, give counts of yes / no / unclear.
```

### Plain Perplexity: paste 5 compounds at a time

```text
For each compound below, find the primary paper that made and measured it, and say whether
bulk superconductivity was observed at ambient pressure. Use primary papers only, not
databases or reviews. A "no" needs the lowest temperature measured. If you cannot find it,
say unclear.

R001  Ba0.9ORb0.1Sb2Ti2   (atomic fractions Ba 0.150, O 0.167, Rb 0.017, Sb 0.333, Ti 0.333)
R002  CeGe0.03Pt3Si0.97   (Ce 0.200, Ge 0.006, Pt 0.600, Si 0.194)
R003  B1.95C0.051Mg       (B 0.650, C 0.017, Mg 0.333)
R004  Al3GeNb14           (Al 0.166, Ge 0.055, Nb 0.779)
R005  ...

Answer as a table: id | formula | superconducting in bulk at ambient pressure (yes / no /
unclear) | Tc (K) | lowest temperature measured (K) | sample form | DOI | one-sentence quote
```

Copy the id, formula and composition_key of the next five rows from the CSV
each time.

### Sending results back

- Upload each CSV, or paste each table into chat.
- Rows are matched by id. Nothing is applied to the training table until
  the correction rule has run.
