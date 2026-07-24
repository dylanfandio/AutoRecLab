# Case-study evidence

Curated artifacts for the searches discussed in the paper. Full workspaces are
not committed; `../audit/file_manifest.csv` and `../audit_preregistered/file_manifest.csv`
are the SHA-256 inventories. Each folder holds the entered prompt, resolved and
source configuration, cost log, requirement checklists, run summary, console log,
and the representative generated code, native results, custom result table, and
validation report(s) that demonstrate the cited behaviour.

## Exploratory phase (P0/P1/P2, several models)

| Run | Paper reference | Behaviour |
| --- | --- | --- |
| V100_P0_MINI_I05_R02 | Results; Table 2 | 90-entry MovieLens; split/threshold deviation |
| V100_P0_GPT54_I05_R01 | Results | Second 90-entry MovieLens (per-seed native files) |
| V100_P0_MINI_I05_R03 | Table 2 | Recall relabelled as Precision |
| V100_P0_NANO_I05_R05 | Table 2 | Mock popularity under three algorithm labels |
| V100_P2_GPT54_I05_R02 | Reviewer alignment | 81.8% reviewer vs 2/270 coverage |
| V100_P0_MINI_I05_R04 | Table 2 | KeyError: 'rating' schema/loader mismatch |
| V100_P0_NANO_I05_R02 | Table 2 | OmniRec/LensKit API mismatch |
| SMOKE_V100_P0_I01 | Table 2 | Excluded smoke run; ~3.64 GiB Amazon allocation |

## Preregistered phase (P2 vs P3, GPT-5.4 Mini, five runs each)

Ten runs backing the P2-vs-P3 comparison and the validator-subversion finding.
`validation_report*.json` in the P3 runs include the report that rewrote its
denominator from 270 to 30 (R03) and the corrupt report (R04).

| Run set | Runs | Behaviour |
| --- | --- | --- |
| V100_P2_MINI_I05_R01..R05 | 5 | Detailed prompt; native metrics, partial coverage, 0 valid |
| V100_P3_MINI_I05_R01..R05 | 5 | Contract prompt; no native metrics, custom tables, subverted validators, 0 valid |
