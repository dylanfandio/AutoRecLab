# Experiment Summary

## User Request

Run a reproducible recommender-systems experiment on the local AutoRecLab environment using the installed recommender stack, without installing LensKit 0.14.4. Evaluate ALS for implicit feedback, ItemKNN, and MostPopular on MovieLens100K, Amazon Video Games, and Last.FM using local files only, 5-core filtering, per-user 80/20 train/test holdout, seeds 7/13/29/42/87, and report nDCG and Precision at 1, 5, and 10 with reproducibility details.

## What Was Run

The provided experiment code attempted to:

- load local dataset files from `../../../study/data/`
- map raw columns for:
  - `u.data` as MovieLens100K
  - `VideoGames.csv` as Amazon Video Games
  - `UserTaggedArtiststimestamps.dat` as Last.FM
- threshold MovieLens100K and Amazon Video Games to `rating > 3`
- treat Last.FM as implicit interactions
- apply 5-core pruning
- perform a custom user-based 80/20 split
- evaluate:
  - ALS via `LensKit.ImplicitMFScorer`
  - ItemKNN via `LensKit.ItemKNNScorer`
  - MostPopular via `LensKit.PopScorer`
- compute nDCG and Precision at k = 1, 5, 10

The run did not complete successfully. It failed during preprocessing before any evaluation results were produced.

## Key Results

| Item | Result |
|---|---|
| Datasets attempted | MovieLens100K, Amazon Video Games, Last.FM |
| Algorithms attempted | ALS, ItemKNN, MostPopular |
| Seeds intended | 7, 13, 29, 42, 87 |
| Metrics intended | nDCG@1/5/10, Precision@1/5/10 |
| Outcome | Failed before producing results |
| Reason for failure | `RecSysDataSet` object lacked `_lineage` during `Pipe(CorePruning(5)).process(ds)` |
| Rows produced | 0 |
| Coverage | 0/270 |
| Validation checks passed | 0 |
| Validation checks failed | Not available from a completed validation report |

### Package and implementation details observed in the code/output

| Category | Detail |
|---|---|
| Python | `sys.version` was recorded in code, but the exact version value is not shown in the output |
| numpy | version recorded in code, exact value not shown in output |
| pandas | version recorded in code, exact value not shown in output |
| matplotlib | version recorded in code, exact value not shown in output |
| omnirec | version recorded in code, exact value not shown in output |
| LensKit API used in code | `LensKit.ImplicitMFScorer`, `LensKit.ItemKNNScorer`, `LensKit.PopScorer` |
| Actual LensKit version | Not shown in the output |
| Algorithm class names | Not recorded in a successful result table because the run crashed before evaluation |

### Preprocessing and counts

No valid preprocessing counts were completed in the output. The only counts visible in the log are error traces from `CorePruning` showing `-1` interactions due to the failed/unknown data variant handling:

| Dataset | Raw interactions | After threshold | After 5-core | Train | Test |
|---|---:|---:|---:|---:|---:|
| MovieLens100K | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | N/A | N/A | N/A | N/A | N/A |
| Last.FM | N/A | N/A | N/A | N/A | N/A |

## Limitations

- The experiment did not finish, so no valid recommendation metrics were produced.
- The output contains no successful result rows, so the requested long-format table cannot be populated.
- The output does not include the exact detected package versions or the exact fully qualified algorithm class names in a completed report.
- The run failed before the required deterministic self-validation could be completed.
- Because of the crash, I cannot verify whether the intended 80/20 split, 5-core filtering, or metric calculations were actually executed correctly.

## Conclusion

This experiment was attempted but did not succeed. It crashed during preprocessing with:

`AttributeError: 'RecSysDataSet' object has no attribute '_lineage'`

As a result, there are no valid metrics, no completed dataset/algorithm/seed results, and no satisfiable 270-row validation report to summarize.