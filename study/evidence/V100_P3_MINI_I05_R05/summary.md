# Experiment Summary

## User Request

The user requested a reproducible recommender-systems experiment using the installed AutoRecLab/OmniRec stack, without requiring LensKit 0.14.4, evaluating ALS for implicit feedback, ItemKNN, and MostPopular on three local datasets with fixed seeds, 80/20 per-user holdout, 5-core filtering, and ranking metrics nDCG and Precision at k = 1, 5, and 10.

## What Was Run

The provided experiment code attempted to:

- Detect installed package versions.
- Load local files from `../../../study/data/`:
  - MovieLens100K: `u.data`
  - Amazon Video Games: `VideoGames.csv`
  - Last.FM: `UserTaggedArtiststimestamps.dat`
- Map each raw file into a common user-item-rating-timestamp schema.
- Convert MovieLens100K and Amazon Video Games to implicit feedback by keeping ratings `> 3`.
- Treat Last.FM tagging events as implicit interactions.
- Apply iterative 5-core filtering.
- For each dataset, algorithm, and seed in `{7, 13, 29, 42, 87}`:
  - perform a per-user 80/20 train/test split,
  - train one of three algorithms:
    - `ALS` → `omnirec.runner.algos.LensKit.ImplicitMFScorer`
    - `ItemKNN` → `omnirec.runner.algos.LensKit.ItemKNNScorer`
    - `MostPopular` → `omnirec.runner.algos.LensKit.PopScorer`
  - evaluate `NDCG([1, 5, 10])` and `Precision([1, 5, 10])`.
- Save incremental results and validate completeness and metric sanity.

The run failed before completion.

## Key Results

| Item | Result |
|---|---|
| Installed `omnirec` | `1.0.0` |
| Installed `lenskit` | `2025.6.2` |
| Installed `recbole` | `not_installed` |
| Installed `numpy` | `1.26.4` |
| Installed `pandas` | `2.3.3` |
| Installed `scipy` | `1.16.2` |
| Installed `matplotlib` | `3.10.7` |
| Algorithm class for ALS | `omnirec.runner.algos.LensKit.ImplicitMFScorer` |
| Algorithm class for ItemKNN | `omnirec.runner.algos.LensKit.ItemKNNScorer` |
| Algorithm class for MostPopular | `omnirec.runner.algos.LensKit.PopScorer` |
| Experiment outcome | Crashed before any evaluation results were produced |
| Cause of failure | `zipfile.BadZipFile: File is not a zip file` when calling `RecSysDataSet.load(tmp_path)` |
| Final result rows produced | `0` |
| Coverage | `0/270` |
| Validation checks passed | `0` reported as passed; experiment did not reach final validation output |

Because the crash happened during `dataframe_to_dataset(train_df)`, no metric table was successfully produced, and no nDCG/Precision values are available from the output.

## Limitations

- The run did not complete, so there are no valid metric results to report.
- The output does not include the requested per-dataset preprocessing counts, per-seed train/test counts, or summary statistics, because the experiment terminated early.
- The provided code attempted to write a CSV file to `RecSysDataSet.load`, but the loader expected a zip-based dataset file, causing the `BadZipFile` error.
- Although the code defines the intended preprocessing and evaluation logic, the output does not confirm that any dataset was successfully evaluated.
- I cannot infer missing metric values or counts; doing so would be speculative.

## Conclusion

The experiment was **not successful**. It detected and reported the installed package versions and the intended LensKit-backed algorithm classes, but it crashed before producing any recommendation metrics. Therefore, the required 270-row results table, seed-wise summaries, and reproducibility report are unavailable from the provided run output.