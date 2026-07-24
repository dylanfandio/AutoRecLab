# Experiment Summary

## User Request

You asked for a LensKit 0.14.4 experiment to measure how random data-split seeds affect recommender accuracy on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The request specified:

- 5-core filtering on all datasets
- Convert ratings > 3 to implicit interactions for MovieLens and Amazon Video Games
- 5 random split seeds
- User-based 80/20 holdout
- Algorithms: ALS, ItemKNN, and Pop
- Metrics: nDCG@1, nDCG@5, nDCG@10, Precision@1, Precision@5, Precision@10
- A short statistical analysis

## What Was Run

The code set up an experiment using `omnirec` with LensKit-backed algorithms:

- `LensKit.ImplicitMFScorer` (ALS)
- `LensKit.ItemKNNScorer`
- `LensKit.PopScorer`

Datasets and preprocessing in the code:

- **MovieLens100K**
  - `MakeImplicit(3)`
  - `CorePruning(5)`
  - `UserHoldout(validation_size=0.0, test_size=0.2)`
- **AmazonVideoGames**
  - `MakeImplicit(3)`
  - `CorePruning(5)`
  - `UserHoldout(validation_size=0.0, test_size=0.2)`
- **LastFM**
  - `CorePruning(5)`
  - `UserHoldout(validation_size=0.0, test_size=0.2)`

Seeds intended: `[11, 22, 33, 44, 55]`

However, the run failed during dataset preprocessing before any model training or evaluation completed.

## Key Results

| Dataset | Algorithm | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MovieLens100K | ALS | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| MovieLens100K | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| MovieLens100K | Pop | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| Amazon Video Games | ALS | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| Amazon Video Games | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| Amazon Video Games | Pop | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| Last.FM | ALS | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| Last.FM | ItemKNN | N/A | N/A | N/A | N/A | N/A | N/A | Not run |
| Last.FM | Pop | N/A | N/A | N/A | N/A | N/A | N/A | Not run |

Observed preprocessing output before failure:

- MovieLens100K was found and canonicalized without re-downloading.
- `MakeImplicit(3)` on MovieLens100K:
  - interactions before: 100000
  - interactions after: 82520
- `CorePruning(5)` on MovieLens100K:
  - interactions before: 82520
  - interactions after: 81697

The run then crashed during `UserHoldout` with this error:

- `InvalidParameterError: The 'test_size' parameter of train_test_split must be a float in the range (0.0, 1.0), an int in the range [1, inf) or None. Got 0.0 instead.`

## Limitations

- No evaluation metrics were produced because the experiment terminated before any training or testing completed.
- The statistical summary could not be computed from the output because there were no final results.
- The failure occurred in preprocessing at the `UserHoldout(validation_size=0.0, test_size=0.2)` step, specifically because `validation_size=0.0` was passed into `train_test_split`, which rejected it in the executed environment.
- Since the run stopped on the first dataset/seed combination, results for all three datasets and all five seeds are unavailable.

## Conclusion

The experiment was set up correctly in terms of the intended datasets, algorithms, metrics, and seed loop, but it did not complete. As a result, there are no measurable findings about seed sensitivity, and no nDCG or Precision values are available from this run.