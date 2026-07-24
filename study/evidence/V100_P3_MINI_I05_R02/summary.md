# Experiment Summary

## User Request

Run a reproducible recommender-systems experiment on the existing AutoRecLab v1.0.0 installation, using the local datasets only, with:

- ALS for implicit feedback
- ItemKNN
- MostPopular

Datasets:
- MovieLens100K: `u.data`
- Amazon Video Games: `VideoGames.csv`
- Last.FM: `UserTaggedArtiststimestamps.dat`

Required setup:
- Do **not** require or install LensKit 0.14.4
- Detect and record the installed package versions
- Inspect and map each raw file according to its actual schema
- For MovieLens100K and Amazon Video Games: keep ratings `> 3` and convert to implicit
- For Last.FM: treat tagging events directly as implicit interactions
- Apply 5-core filtering to all processed datasets
- Use seeds `7, 13, 29, 42, 87`
- Perform a per-user 80/20 train/test holdout with no validation split
- Evaluate nDCG and Precision at `k = 1, 5, 10`
- Persist results incrementally
- Verify all metrics are finite and computed from actual recommendations
- Produce long-format results and summary statistics
- Include deterministic validation reporting

## What Was Run

The provided script attempted to:

- Import OmniRec / LensKit-related components
- Load the three datasets from `../../../study/data/`
- Map dataset schemas
- Threshold ratings and apply 5-core filtering
- Split each dataset by user into train/test for each seed
- Evaluate ALS, ItemKNN, and MostPopular
- Write results and validation reports

However, the run failed before completing dataset loading and before any metrics were produced.

The crash occurred in `load_raw_schema()` while processing `Amazon Video Games`:

- It attempted to infer column names from the CSV header
- The inferred columns were all `None`
- Pandas raised:

  `KeyError: "None of [Index([None, None, None, None], dtype='object')] are in the [columns]"`

So the experiment did **not** reach training, recommendation generation, or metric evaluation.

## Key Results

| Item | Result |
|---|---|
| Experiment status | Failed before completion |
| Completed dataset/algorithm/seed combinations | 0 |
| Metric rows produced | 0 |
| Final coverage | 0/270 |
| Failed validation checks | N/A, because validation never ran |
| Detected package versions | N/A in output |
| Algorithm classes used | N/A in output |
| Raw interaction counts | N/A |
| Thresholded / 5-core counts | N/A |
| Train/test counts | N/A |
| nDCG / Precision values | N/A |

## Limitations

- No experiment results were produced, so the requested table of dataset, algorithm, seed, cutoff, nDCG, and Precision cannot be populated from the available output.
- The script did not complete the required deterministic validation.
- The output does not include detected package versions, algorithm class names, preprocessing counts, or train/test counts.
- The failure happened during schema mapping for `VideoGames.csv`, so the raw-file inspection for at least one dataset was not successfully completed.
- Because the run stopped early, there is no evidence that any recommendations were generated or that any metrics were computed from actual recommendation lists.

## Conclusion

The experiment was **not successful**. It crashed during dataset loading, before any model training or evaluation occurred. As a result, none of the requested reproducible recommender-system results, summary statistics, or validation checks are available from this run.