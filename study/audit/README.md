# Retained-run artifact audit

Generated: 2026-07-21

## Scope and method

This audit covers every file retained under `study_runs/` and `study_logs/`. All
2,810 files (47.47 GiB) were opened and SHA-256 hashed. Text, CSV, and JSON
artifacts were parsed; the very large prediction JSON files were inspected with
streaming logic. Pickles and other binary formats were not deserialized because
deserializing untrusted pickle data can execute code. Their signatures and
readability were checked instead. PNG and PDF signatures were also validated.

The historical 36 runs described in the draft are **not** part of this retained
artifact set. The principal follow-up sample is the 19 nonzero-cost directories
whose names match `V100_P[012]_(NANO|MINI|GPT54)_I05_Rnn`. Pilot, smoke,
encoding-failure snapshots, and one empty run directory are kept separate.

## Main result

The retained experiments are sufficient for an artifact-backed study of
AutoRecLab's reliability and failure modes. They are not sufficient to claim a
completed three-dataset recommender benchmark or to estimate seed sensitivity.

Across the 19 planned follow-up searches:

- 2,832 model API calls cost USD 23.57.
- The cost-log spans sum to approximately 15.2 hours.
- The searches generated 188 production code nodes.
- 78/188 nodes ran without a recorded exception (41.5%); 110/188 crashed.
- 24 of the 78 non-crashing nodes contained mock or fallback behavior.
- 12/19 searches produced at least one requested metric from a native OmniRec
  `results.json` artifact.
- 0/19 searches completed the requested task satisfactorily.

The requested full matrix contains 270 metric entries: 3 datasets x 3
algorithms x 5 seeds x 3 cutoffs x 2 metrics. Two searches reached all 90
MovieLens entries, but both used incompatible preprocessing/splitting choices
and then failed at the next dataset. These are genuine partial executions, not
valid completions.

## Important validity findings

1. `MakeImplicit(3)` retained 82,520 of 100,000 MovieLens ratings. This is a
   `rating >= 3` interpretation, while the task requested `rating > 3` (55,375
   ratings before subsequent filtering).
2. Several runs attempted a zero-sized validation set, which the installed
   stack rejects. Other runs silently used non-requested train/validation/test
   proportions to satisfy the framework.
3. A 135-row result that appears complete uses a mock popularity recommender for
   all three algorithm labels. Another run copied Recall values and labeled them
   Precision. Zero-valued fallback tables also occur.
4. Intermediate artifacts can be materially better than the node chosen during
   finalization. Therefore the final summary alone does not reliably describe
   the best evidence produced by a search.
5. Reviewer checklists vary in size (11 to 23 requirements). Best reviewer score
   has only a weak descriptive association with objective native-metric coverage
   (Pearson r = 0.337). One 81.8% reviewer score covered only 2/270 requested
   metric entries.
6. Recurring failure families include invalid split parameters, input-schema
   mismatches, OmniRec API/version mismatches, loader/download failures,
   evaluation-schema errors, path problems, and an Amazon prediction allocation
   failure of approximately 3.64 GiB.

## Interpretation boundaries

- Code nodes are nested attempts within 19 searches; they are not 188
  independent experiments.
- The model/prompt design is unbalanced: P0 covers several models, P1 is mainly
  Nano, and P2 is GPT-5.4. Model superiority and causal prompt effects cannot be
  inferred cleanly.
- Cost and elapsed time are descriptive because runs differ in failure point and
  generated work.
- Downloaded archives and canonicalized dataset copies are execution caches, not
  additional experimental datasets. Some nodes used those built-in loaders
  rather than the exact supplied inputs.
- Exact duplicate artifacts are common: 168 content groups contain 673 copies,
  representing about 2.05 GiB of redundant bytes. Copies and deterministic
  reruns must not be counted as independent evidence.
- The missing historical 36-run artifacts should be reported as an explicit
  reproducibility limitation and analyzed separately from this retained sample.

## Recommended paper framing

Primary question: How reliably does an autonomous recommender-systems lab turn a
complex natural-language specification into executable and semantically valid
experiments?

Useful supporting questions are where failures arise, whether automated reviewer
scores track objective completion, and how cost relates to valid coverage. A
failure-stage funnel, run-by-run coverage heatmap, reviewer-score-versus-coverage
scatterplot, and failure taxonomy table can be constructed from the audit files.

## Audit artifacts

- `file_manifest.csv`: every retained file, size, type, and SHA-256
- `run_summary.csv`: directory-level inventory and recorded costs
- `csv_inventory.csv`, `json_inventory.csv`: structured-file inspection
- `code_inventory.csv`, `generated_code_assessment.csv`: generated-code evidence
- `native_metrics.csv`, `native_result_files.csv`: normalized native metrics
- `text_marker_inventory.csv`: text/log marker counts
- `duplicate_groups.csv`: exact content duplicates
- `audit_totals.json`: global totals

The reproducible audit utilities are `study/audit_runs.py` and
`study/analyze_audit.py`.
