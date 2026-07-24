# Experiment Summary

## User Request

Run a LensKit 0.14.4 experiment to measure how sensitive recommender accuracy is to data split random seeds, using three algorithms (ALS, ItemKNN, Pop) on three implicit-feedback datasets (MovieLens100K, Amazon Video Games, Last.FM), with 5-core preprocessing, 5 different random seeds, user-based 80/20 holdout, and evaluation using nDCG@1/5/10 and Precision@1/5/10.

## What Was Run

The provided code attempted to:

- Load three datasets from `../../../study/data/`:
  - `u.data` → MovieLens100K
  - `VideoGames.csv` → Amazon Video Games
  - `UserTaggedArtiststimestamps.dat` → Last.FM
- Standardize each dataset to `user`, `item`, `rating`, `timestamp`
- Apply preprocessing:
  - `CorePruning(5)` to all datasets
  - `MakeImplicit(3)` for MovieLens100K and Amazon Video Games
  - `UserHoldout(validation_size=0.0, test_size=0.2)`
- Evaluate three LensKit algorithms via OmniRec:
  - `LensKit.ImplicitMFScorer` (ALS-style implicit MF)
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Repeat the split/evaluation for seeds:
  - 7, 13, 21, 42, 87
- Measure:
  - `NDCG([1, 5, 10])`
  - `Precision([1, 5, 10])`

## Key Results

The experiment did not complete, so no metric values were produced.

| Dataset | Algorithm | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| MovieLens100K | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A |
| MovieLens100K | Pop | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | Pop | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | ALS | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | Pop | N/A | N/A | N/A | N/A | N/A | N/A |

The only observed output was an error before evaluation began:

- `ImportError: cannot import name 'RawData' from 'omnirec.rsds' (unknown location)`

## Limitations

- No model training or evaluation results were generated because the script crashed immediately on import.
- The requested statistical analysis could not be performed.
- The output does not confirm whether the installed `omnirec` package version matches the code’s expected API.
- Because the run failed at import time, there are no per-seed, per-dataset, or per-metric values to report.

## Conclusion

The experiment setup matches the user’s requested design, but the run failed before execution due to an import error involving `RawData` from `omnirec.rsds`. As a result, there are no accuracy or seed-sensitivity results available from this run.