# Experiment Summary

## User Request

The request was to use LensKit 0.14.4 to compare how different data split random seeds affect recommender accuracy on three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM. The plan was:

- 5-core filtering for all datasets
- Convert ratings > 3 to implicit interactions for MovieLens and Amazon
- Use 5 random split seeds
- For each algorithm/dataset/seed, run a user-based 80/20 holdout split
- Train ALS, ItemKNN, and Pop with standard hyperparameters
- Evaluate nDCG@k and Precision@k for k = 1, 5, 10
- Perform a short statistical analysis

## What Was Run

The code did the following:

- Loaded datasets via `RecSysDataSet.use_dataloader(...)`
- For MovieLens100K and Amazon2014VideoGames, applied `MakeImplicit(3)`
- Applied `CorePruning(5)` to all datasets
- Applied `UserHoldout(validation_size=0.2, test_size=0.2)`
- Repeated the process for seeds `[11, 22, 33, 44, 55]`
- Evaluated three LensKit algorithms:
  - `LensKit.ImplicitMFScorer` (ALS-like implicit MF)
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Measured:
  - `NDCG([1, 5, 10])`
  - `Precision([1, 5, 10])`
- Saved per-seed CSVs and attempted summary/statistical outputs

The run completed only partially and then crashed during the Amazon stage due to a memory allocation error while converting predictions to a pandas DataFrame.

## Key Results

The output contains complete evaluation tables only for **Seed 11** on **MovieLens100K** and partial results for **Amazon2014VideoGames**.

| Dataset | Seed | Algorithm | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MovieLens100K | 11 | ALS / ImplicitMFScorer | 0.2036… | 0.17510… | 0.16154… | 0.2036… | 0.17030… | 0.153128… |
| MovieLens100K | 11 | ItemKNNScorer | 0.24920… | 0.22239… | 0.19746… | 0.24920… | 0.21399… | 0.182608… |
| MovieLens100K | 11 | PopScorer | 0.19512… | 0.15620… | 0.14201… | 0.19512… | 0.14549… | 0.130858… |
| Amazon2014VideoGames | 11 | ALS / ImplicitMFScorer | 0.01658… | 0.01445… | 0.01252… | 0.01658… | 0.013756… | 0.01132… |
| Amazon2014VideoGames | 11 | ItemKNNScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Amazon2014VideoGames | 11 | PopScorer | N/A | N/A | N/A | N/A | N/A | N/A |
| Last.FM | 11 | All algorithms | N/A | N/A | N/A | N/A | N/A | N/A |

Additional factual observations from the output:

- MovieLens100K preprocessing:
  - 100000 interactions before implicit conversion
  - 82520 after `MakeImplicit(3)`
  - 81697 after 5-core pruning
- Amazon2014VideoGames preprocessing:
  - 1,324,753 interactions before implicit conversion
  - 1,094,400 after `MakeImplicit(3)`
  - 177,572 after 5-core pruning
- The run never reached a successful completion for all seeds/datasets.
- The summary/statistical analysis files were not shown as produced in the output.

## Limitations

- The experiment did **not finish** successfully.
- Results are only visible for:
  - MovieLens100K, seed 11, all 3 algorithms
  - Amazon2014VideoGames, seed 11, ALS only
- No output is shown for:
  - MovieLens seeds 22, 33, 44, 55
  - Amazon seeds beyond 11
  - Last.FM for any seed
- The code requested “standard hyperparameters,” but the output does not explicitly list the hyperparameter settings used by LensKit.
- Because the run crashed, the requested seed-sensitivity statistical analysis could not be verified from the provided output.
- The reported tables in the log are truncated with ellipses, so some numeric values are approximate in the transcript.

The crash reason was:

- `numpy.core._exceptions._ArrayMemoryError: Unable to allocate 3.64 GiB ...`
- This occurred while building predictions/evaluation data for Amazon2014VideoGames.

## Conclusion

Based on the available output, the experiment was started correctly and followed the intended preprocessing and evaluation design for the first dataset/seed, but it did not complete the full 3-dataset × 5-seed × 3-algorithm study. The only fully observed comparison is for MovieLens100K at seed 11, where ItemKNN performed best among the three algorithms on both nDCG and Precision at k = 1, 5, and 10. However, the provided output is insufficient to quantify seed sensitivity across all datasets, because the run crashed before completing the planned repetitions and statistical analysis.