# Family-diverse literature search

This note records public leads found while looking for empirical training
records outside the current Y-cuprate set. The rows are in
`data/training/review_queue_family_leads.csv` and come from the public NIMS
SuperCon2 export. SuperCon2 is an automatic literature extraction, so these
are review leads rather than confirmed labels.

The search also checked the public GitHub repository for SuperMat 2.0. Its
dataset is available at
`https://github.com/lfoppiano/SuperMat/tree/master/data/csv` and provides
paper-linked (T_c), pressure, and measurement-method annotations. It is useful
for finding additional records, but its free-text values still require paper
verification.

The new queue deliberately includes several kinds of lead:

- iron-based records near ambient pressure and under pressure;
- an MgB₂ thin-film record near 39 K;
- hydride records with predicted transition temperatures above 200 K;
- calculated records and experimental records kept in separate evidence types;
- extraction errors such as a formula/title mismatch and a missing formula.

The hydride entries require pressures of 120–460 GPa in the extracted rows.
They are useful for mechanism and model checks, but they do not meet the
project's ambient or 0.1 GPa engineering target. The iron-based and MgB₂ leads
are more useful for broadening the empirical family coverage, subject to source
review.

For each pending lead, verify the DOI paper and then record the exact formula,
(T_c), pressure, sample form, method, and whether the value is measured or
calculated. Only then should a row move into the harmonized training table.
