# Pseudopotential source record

The user pointed to [`dalcorso/pslibrary`](https://github.com/dalcorso/pslibrary),
a GPL-2.0 Quantum ESPRESSO pseudopotential generator library. We inspected and
pinned repository revision `835dd6477c90fa9d59524914de50901d4be73e36`.

That GitHub repository contains `ld1.x` input/job files and scripts for
generating UPF files. It does **not** contain ready-to-use `.UPF` files, and
therefore is not itself a complete pseudopotential directory to give QE. The
catalog lists the library's suggested PBE family by element for elements in
the current 13-structure registry. `$fct` and `*` are intentionally patterns:
the actual functional-specific filename and PAW/ultrasoft variant must be
selected and checked against the library's generated output and test jobs.

Do not treat the catalog as an approved calculation setup. Before production,
the compute operator should select a consistent PAW or ultrasoft family,
obtain/generate the UPF files using a compatible Quantum ESPRESSO `ld1.x`,
record each UPF SHA-256 and source/license, read the recommended cutoff from the
UPF, and converge cutoffs and k meshes for each material. Relativistic choices
also need review for heavy elements and spin-orbit-sensitive questions.

The source checkout used for inspection was temporary; the project records its
source URL, revision, license, and recommendations here rather than copying
generator code into the application repository.
