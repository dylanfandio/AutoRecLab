# Experiment Summary

## User Request

The request was to run a reproducible recommender-systems experiment in the existing AutoRecLab v1.0.0 environment using local data only, without requiring LensKit 0.14.4, and to report results for:

- ALS for implicit feedback
- ItemKNN
- MostPopular

across three datasets:

- MovieLens100K (`u.data`)
- Amazon Video Games (`VideoGames.csv`)
- Last.FM (`UserTaggedArtiststimestamps.dat`)

with:

- ratings > 3 retained and converted to implicit for MovieLens100K and Amazon Video Games
- Last.FM tagging events used directly as implicit interactions
- 5-core filtering
- fixed seeds 7, 13, 29, 42, 87
- user-based 80/20 train/test holdout
- metrics nDCG and Precision at k = 1, 5, 10
- incremental persistence of completed results
- exact package versions, algorithm implementations, preprocessing counts, and train/test counts

## What Was Run

Only one experiment run is shown in the provided materials.

From the code and output, the run:

- loaded **MovieLens100K** from the local `../../../study/data/` directory
- applied:
  - `MakeImplicit(3)`
  - `CorePruning(5)`
- used `UserHoldout(validation_size=0.0, test_size=0.2)`
- evaluated **one algorithm only**:
  - `LensKit.ImplicitMFScorer` via `ExperimentPlan.add_algorithm(LensKit.ImplicitMFScorer, {})`
- measured:
  - `NDCG([10])`
  - `Precision([10])`

The output shows the environment reused an existing LensKit environment, but it does not show a version number for LensKit itself.

## Key Results

| Dataset | Algorithm | Seed | Cutoff | nDCG | Precision |
|---|---|---:|---:|---:|---:|
| MovieLens100K | LensKit.ImplicitMFScorer | 42 | 10 | 0.19813894474495342 | 0.18981972428419935 |
| MovieLens100K | LensKit.ImplicitMFScorer | 42 | 1 | N/A | N/A |
| MovieLens100K | LensKit.ImplicitMFScorer | 42 | 5 | N/A | N/A |

Additional factual counts from the run:

| Item | Value |
|---|---:|
| Raw interactions | 100000 |
| After implicit conversion (`> 3`) | 82520 |
| After 5-core pruning | 81697 |
| Train interactions | 65352 |
| Validation interactions | 0 |
| Test interactions | 16345 |

Detected package versions recorded in the output:

| Package | Version |
|---|---|
| omnirec | unknown |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |

Algorithm implementation recorded in the output:

| Algorithm class |
|---|
| `LensKit.ImplicitMFScorer` |

## Limitations

The provided output is **not sufficient** to satisfy the full requested experiment.

Specifically:

- Only **MovieLens100K** was run; there are no results for:
  - Amazon Video Games
  - Last.FM
- Only **one algorithm** was run; there are no results for:
  - ALS for implicit feedback across all required datasets/seeds
  - ItemKNN
  - MostPopular
- Only **one seed** was run:
  - 42
  - not the requested 7, 13, 29, 42, 87
- Only **NDCG@10** and **Precision@10** were reported.
  - No results are shown for cutoff 1 or 5.
- The code includes a fallback split path, but the output does not indicate whether it was used.
- The environment/version of **LensKit** itself is not reported explicitly.
- The output does not include the actual raw-schema mappings for the Amazon Video Games and Last.FM files, since those datasets were not run here.
- Because only one completed combination is available, no mean/std/min/max/range across five seeds can be computed from the provided output.

## Conclusion

The provided materials document a **single successful MovieLens100K run** using `LensKit.ImplicitMFScorer` with seed 42, implicit conversion threshold 3, 5-core pruning, and a user holdout split. The run produced finite results for **NDCG@10 = 0.19813894474495342** and **Precision@10 = 0.18981972428419935**.

However, the materials do **not** contain the full multi-dataset, multi-algorithm, multi-seed experiment requested, so the remaining requested tables, aggregate statistics, and cross-seed analysis cannot be completed from the provided output alone.