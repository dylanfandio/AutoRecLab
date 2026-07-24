# Experiment Summary

## User Request

You asked for a reproducible recommender-systems experiment in the installed AutoRecLab v1.0.0 environment, without requiring or installing LensKit 0.14.4, evaluating three algorithms:

- ALS for implicit feedback
- ItemKNN
- MostPopular

on these local files from `../../../study/data/`:

- MovieLens100K: `u.data`
- Amazon Video Games: `VideoGames.csv`
- Last.FM: `UserTaggedArtiststimestamps.dat`

with:

- schema inspection of each raw file
- implicit conversion for MovieLens100K and Amazon ratings greater than 3
- direct implicit use of Last.FM tagging events
- 5-core filtering on all datasets
- user-based 80/20 train/test holdout
- seeds 7, 13, 29, 42, 87
- metrics: nDCG and Precision at k = 1, 5, 10
- incremental persistence
- finite, actual metric values only

## What Was Run

Based on the provided code and output, the completed run was only a prototype subset of the requested experiment:

- Dataset used: `MovieLens100K`
- Raw file path checked: `../../../study/data/u.data`
- Raw schema used:
  - separator: tab
  - columns: `user`, `item`, `rating`, `timestamp`
- Preprocessing actually run:
  - `MakeImplicit(3)`:
    - before: 100000 interactions
    - after: 82520 interactions
  - `CorePruning(5)`:
    - before: 82520 interactions
    - after: 81697 interactions
- Split actually run:
  - manual user-based 80/20 train/test holdout
  - no validation split
- Seed actually run: `7`
- Cutoff actually run: `10`
- Algorithm class actually used:
  - `omnirec.runner.algos.LensKit.PopScorer`
  - reported as `LensKit.PopScorer`
  - labeled in the code as `MostPopular`
- Evaluation actually produced:
  - `NDCG@10`
  - `Precision@10`
- Actual recommendation output was verified indirectly by `prediction_rows = 1068695`, and the code rejected zero-prediction runs.

Detected package versions were requested by the code, but they were not printed in the experiment output, so they are not available from the provided materials.

## Key Results

Only one successful result row is available from the provided output.

| Dataset | Algorithm | Seed | Cutoff | nDCG | Precision |
|---|---|---:|---:|---:|---:|
| MovieLens100K | LensKit.PopScorer (MostPopular) | 7 | 10 | 0.17023035757720048 | 0.1537645811240721 |

Additional reproducibility counts from the run:

| Dataset | Raw Interactions | Processed Interactions | Train Interactions | Test Interactions | Prediction Rows |
|---|---:|---:|---:|---:|---:|
| MovieLens100K | 100000 | 81697 | 65734 | 15963 | 1068695 |

Requested multi-seed summary statistics could not be computed as requested. The output contains only one seed for one dataset–algorithm–cutoff combination, so the reported prototype summary is:

| Dataset | Algorithm | Cutoff | Mean nDCG | Std nDCG | Min nDCG | Max nDCG | Range nDCG | Mean Precision | Std Precision | Min Precision | Max Precision | Range Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens100K | LensKit.PopScorer | 10 | 0.17023 | 0.0 | 0.17023 | 0.17023 | 0.0 | 0.153765 | 0.0 | 0.153765 | 0.153765 | 0.0 |

Metric interpretation, based on the reported values:

- `nDCG@10 = 0.1702` indicates relatively limited ranking quality at the top 10; higher would be better.
- `Precision@10 = 0.1538` means about 15.4% of the top-10 recommended items were relevant on average in this evaluation.

## Limitations

The provided code and output do **not** satisfy the full user request. Specifically:

- Only **one** algorithm was run:
  - `MostPopular` via `LensKit.PopScorer`
- The following requested algorithms were **not** run:
  - ALS for implicit feedback
  - ItemKNN
- Only **one** dataset was run:
  - MovieLens100K
- The following requested datasets were **not** run:
  - Amazon Video Games (`VideoGames.csv`)
  - Last.FM (`UserTaggedArtiststimestamps.dat`)
- Only **one** seed was run:
  - 7
- The following requested seeds were **not** run:
  - 13, 29, 42, 87
- Only **one** cutoff was evaluated:
  - 10
- The following requested cutoffs were **not** run:
  - 1, 5
- No evidence is provided that raw schemas for Amazon Video Games or Last.FM were inspected or mapped.
- The requested incremental persistence across all dataset/algorithm/seed/cutoff combinations was not demonstrated; only the prototype single-run result was saved.
- Exact package versions were not shown in the output, so they cannot be reported factually from the provided materials.
- Because only one seed was completed, no meaningful statistical analysis of seed sensitivity can be made.

## Conclusion

From the provided materials, the experiment that actually completed was a **prototype** run of **MostPopular** (`LensKit.PopScorer`) on **MovieLens100K** only, using:

- implicit conversion with threshold `> 3` as implemented by `MakeImplicit(3)`
- `5`-core pruning
- user-based `80/20` train/test holdout
- seed `7`
- evaluation at `k = 10`

The only reported metrics are:

- `NDCG@10 = 0.17023035757720048`
- `Precision@10 = 0.1537645811240721`

The full requested experiment across three datasets, three algorithms, five seeds, and three cutoffs was **not completed** in the provided run, so the missing results, package-version list, and cross-seed statistical conclusions cannot be reported without guessing.