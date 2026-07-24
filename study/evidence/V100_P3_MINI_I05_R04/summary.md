# Experiment Summary

## User Request

The request was to run a reproducible recommender-systems experiment on the existing AutoRecLab v1.0.0 environment using local dataset files only, evaluating:

- ALS for implicit feedback
- ItemKNN
- MostPopular

on:

- MovieLens100K (`u.data`)
- Amazon Video Games (`VideoGames.csv`)
- Last.FM (`UserTaggedArtiststimestamps.dat`)

with 5-core filtering, implicit-feedback handling, per-user 80/20 train/test holdout, seeds `[7, 13, 29, 42, 87]`, and metrics nDCG and Precision at `k = 1, 5, 10`.

## What Was Run

The provided experiment code attempted to:

- inspect and map the three raw files,
- threshold MovieLens100K and Amazon Video Games to implicit interactions using `rating > 3`,
- treat Last.FM as implicit interactions,
- apply iterative 5-core filtering,
- create per-user 80/20 train/test splits for each seed,
- evaluate three algorithms via `omnirec` / LensKit-backed runner code,
- save long-format results and summary statistics.

The run did **not** complete successfully. It failed during the first evaluation call with:

- `AttributeError: 'str' object has no attribute '_meta'`

followed by a checkpoint cleanup failure:

- `FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints\\out.log'`

So no completed metric table was produced in the available output.

## Key Results

| Item | Result |
|---|---|
| Experiment status | Failed before producing valid evaluation results |
| Completed metric rows | N/A |
| nDCG / Precision values | N/A |
| Dataset preprocessing counts | N/A |
| Train/test counts per seed | N/A |
| Package versions recorded in output | N/A |
| Algorithm classes used | N/A |

## Limitations

- No successful results table was emitted in the provided output.
- The run crashed before any valid recommendation metrics were reported.
- The output does not include the required reproducibility metadata, package versions, algorithm class names, preprocessing counts, or train/test counts.
- Because the experiment failed, there is no factual basis here to report the requested 270-row long-format result table or summary statistics.
- The output also does not confirm whether the intended dataset mapping, 5-core filtering, or metric checks were actually reached before the crash.

## Conclusion

The experiment was **not successfully completed** in the provided environment/output. The only definitive finding is that the runner failed with an internal dataset/object handling error before evaluation results could be generated. Therefore, the requested metrics, seed-sensitivity analysis, and reproducibility tables are not available from this run.