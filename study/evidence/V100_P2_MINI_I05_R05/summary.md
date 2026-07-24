# Experiment Summary

## User Request

Run a reproducible recommender-systems experiment using the already installed AutoRecLab v1.0.0 software, without requiring LensKit 0.14.4. The request specified three algorithms (ALS for implicit feedback, ItemKNN, MostPopular), three local datasets, 5-core filtering, user-based 80/20 holdout, seeds 7/13/29/42/87, and evaluation with nDCG and Precision at k = 1, 5, 10. It also asked for incremental persistence, exact package versions, algorithm implementations, preprocessing counts, and train/test counts.

## What Was Run

The provided experiment code attempted to:

- Load local files from `../../../study/data/`
  - MovieLens100K: `u.data`
  - Amazon Video Games: `VideoGames.csv`
  - Last.FM: `UserTaggedArtiststimestamps.dat`
- Inspect and map each raw file schema.
- Convert MovieLens100K and Amazon Video Games ratings greater than 3 to implicit feedback.
- Treat Last.FM tagging events directly as implicit interactions.
- Apply iterative 5-core filtering.
- Evaluate three algorithms via `omnirec.runner.algos.LensKit`:
  - `LensKit.ImplicitMFScorer` for ALS for implicit feedback
  - `LensKit.ItemKNNScorer` for ItemKNN
  - `LensKit.PopScorer` for MostPopular
- Use user holdout split `UserHoldout(0.0, 0.2)` with seeds 7, 13, 29, 42, and 87.
- Compute `NDCG` and `Precision` at cutoffs 1, 5, and 10.
- Save incremental JSONL results and summary CSV files.

The run failed before completing the first dataset-to-model execution because `RecSysDataSet(df)` raised:

> `ValueError: The truth value of a DataFrame is ambiguous.`

This means no evaluation results were produced.

## Key Results

### Compact results table

| Dataset | Algorithm | Seed | Cutoff | nDCG | Precision |
|---|---|---:|---:|---:|---:|
| MovieLens100K | ALS for implicit feedback | 7 | 1 | N/A | N/A |
| MovieLens100K | ALS for implicit feedback | 7 | 5 | N/A | N/A |
| MovieLens100K | ALS for implicit feedback | 7 | 10 | N/A | N/A |
| MovieLens100K | ItemKNN | 7 | 1 | N/A | N/A |
| MovieLens100K | ItemKNN | 7 | 5 | N/A | N/A |
| MovieLens100K | ItemKNN | 7 | 10 | N/A | N/A |
| MovieLens100K | MostPopular | 7 | 1 | N/A | N/A |
| MovieLens100K | MostPopular | 7 | 5 | N/A | N/A |
| MovieLens100K | MostPopular | 7 | 10 | N/A | N/A |
| Amazon Video Games | ALS for implicit feedback | 7 | 1 | N/A | N/A |
| Amazon Video Games | ALS for implicit feedback | 7 | 5 | N/A | N/A |
| Amazon Video Games | ALS for implicit feedback | 7 | 10 | N/A | N/A |
| Amazon Video Games | ItemKNN | 7 | 1 | N/A | N/A |
| Amazon Video Games | ItemKNN | 7 | 5 | N/A | N/A |
| Amazon Video Games | ItemKNN | 7 | 10 | N/A | N/A |
| Amazon Video Games | MostPopular | 7 | 1 | N/A | N/A |
| Amazon Video Games | MostPopular | 7 | 5 | N/A | N/A |
| Amazon Video Games | MostPopular | 7 | 10 | N/A | N/A |
| Last.FM | ALS for implicit feedback | 7 | 1 | N/A | N/A |
| Last.FM | ALS for implicit feedback | 7 | 5 | N/A | N/A |
| Last.FM | ALS for implicit feedback | 7 | 10 | N/A | N/A |
| Last.FM | ItemKNN | 7 | 1 | N/A | N/A |
| Last.FM | ItemKNN | 7 | 5 | N/A | N/A |
| Last.FM | ItemKNN | 7 | 10 | N/A | N/A |
| Last.FM | MostPopular | 7 | 1 | N/A | N/A |
| Last.FM | MostPopular | 7 | 5 | N/A | N/A |
| Last.FM | MostPopular | 7 | 10 | N/A | N/A |

### Why values are N/A

No metrics were computed because the run crashed before any `run_omnirec(...)` evaluation could complete.

### Exact package versions detected in code

The experiment code attempted to record:

- `omnirec`: `metadata.version('omnirec')`
- `numpy`: `np.__version__`
- `pandas`: `pd.__version__`
- `matplotlib`: `plt.matplotlib.__version__`

However, the output did not print these values before the crash, so the exact versions are not available from the provided run output.

### Exact algorithm classes used

The code specifies:

- ALS for implicit feedback → `LensKit.ImplicitMFScorer`
- ItemKNN → `LensKit.ItemKNNScorer`
- MostPopular → `LensKit.PopScorer`

### Preprocessing and split counts

The code was designed to record:

- raw interaction counts
- raw user counts
- raw item counts
- post-threshold counts for MovieLens100K and Amazon Video Games
- post-5-core counts
- train/test interaction counts per dataset

But none of these counts were printed in the captured output, and the crash occurred before the final metadata writeout. Therefore they are not recoverable from the provided output.

## Limitations

- The experiment did not complete successfully.
- No evaluation metrics were produced.
- No summary statistics across seeds were produced.
- The exact detected package versions were not printed in the output.
- The run crashed at `RecSysDataSet(df)` due to a pandas DataFrame truth-value error inside `omnirec`.
- Because of the crash, there is no evidence in the output that any dataset was successfully evaluated, any incremental results were retained, or any metric values were finite.

## Conclusion

The requested experiment was set up according to the provided code, including the specified datasets, preprocessing logic, seeds, cutoffs, and algorithm classes. However, it failed before any recommender evaluation ran, so no valid accuracy results, seed-sensitivity analysis, or reproducible reproduction counts can be reported from this output. The only confirmed factual result is the crash caused by `RecSysDataSet(df)` raising a DataFrame ambiguity error.