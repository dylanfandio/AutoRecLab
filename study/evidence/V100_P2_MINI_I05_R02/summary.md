# Experiment Summary

## User Request

Run a reproducible recommender-systems experiment in the installed AutoRecLab v1.0.0 environment, using local datasets only, with ALS for implicit feedback, ItemKNN, and MostPopular; apply the specified preprocessing, 5-core filtering, fixed seeds, and report nDCG/Precision at 1, 5, and 10, plus reproducibility details.

## What Was Run

The provided code used:

- `omnirec` version `1.0.0`
- `numpy` `1.26.4`
- `pandas` `2.3.3`
- `matplotlib` `3.10.7`
- `scikit-learn` `1.7.1`

Algorithm classes used:

- ALS for implicit feedback: `LensKit.ImplicitMFScorer`
- ItemKNN: `LensKit.ItemKNNScorer`
- MostPopular: `LensKit.PopScorer`

Important mismatch with the user request:

- The code only processed **MovieLens100K**.
- It did **not** run Amazon Video Games or Last.FM.
- It used `UserHoldout(validation_size=0.15, test_size=0.20)`, which creates a validation split, contrary to the request for only an 80/20 train/test split and no validation split.

Preprocessing and counts reported for MovieLens100K:

- Raw interactions: `100000`
- Raw users: `943`
- Raw items: `1682`
- After implicit conversion with threshold 3: `82520`
- After 5-core pruning: `81697`
- Train interactions: `52379`
- Validation interactions: `0` in the printed summary object
- Test interactions: `16707`

The experiment loop used seeds:

- `7, 13, 29, 42, 87`

Metrics used in the code:

- `NDCG([1, 5, 10])`
- `Recall([1, 5, 10])`

The code then renamed `Recall` to `Precision` in the final table, but the output shown is still the library’s `Recall@k`-style table. So the reported “Precision” values are actually whatever `Recall` returned in the experiment output.

## Key Results

Only MovieLens100K results were produced in the output. The output contains 135 rows, but it does not include the fully expanded table in machine-readable form; it shows the final table with rows for each dataset/algorithm/seed/cutoff and the metrics. The exact table values for every row are not fully recoverable from the truncated display, but the per-algorithm metric patterns are visible.

### Compact results summary

| Dataset | Algorithm | Seed coverage | Cutoffs | nDCG / Precision availability | Notes |
|---|---:|---:|---:|---|---|
| MovieLens100K | ALS (`LensKit.ImplicitMFScorer`) | 5 seeds | 1, 5, 10 | Exact per-row values shown in output table | Seed-sensitive variation visible |
| MovieLens100K | ItemKNN (`LensKit.ItemKNNScorer`) | 5 seeds | 1, 5, 10 | Exact per-row values shown in output table | Metrics were identical across seeds in the summary statistics |
| MovieLens100K | MostPopular (`LensKit.PopScorer`) | 5 seeds | 1, 5, 10 | Exact per-row values shown in output table | Metrics were identical across seeds in the summary statistics |

### Seed-sensitivity analysis from the printed summary

The printed `seed_sensitivity_analysis` shows:

- ALS had nonzero variation across seeds for some metrics:
  - Example visible values:
    - `nDCG@1` mean `0.193001...`, std `0.0` for one grouped subset in the printed analysis due to grouping artifacts, but the raw table shows different seed-specific values for ALS.
  - Because the printed analysis keys are inconsistent with the final table grouping, the summary object is not fully reliable for exact seed sensitivity interpretation.
- ItemKNN and MostPopular show essentially zero variation across seeds in the printed analysis.
- The final table itself indicates:
  - ItemKNN produced the same metric values across all seeds.
  - MostPopular produced the same metric values across all seeds.
  - ALS varied by seed.

### Representative metric values visible in the output

From the displayed evaluation tables for MovieLens100K:

- ALS (`LensKit.ImplicitMFScorer`) had values around:
  - `NDCG@1`: roughly `0.1877` to `0.1962`
  - `NDCG@5`: roughly `0.1733` to `0.1880`
  - `NDCG@10`: roughly `0.1621` to `0.1801`
- ItemKNN (`LensKit.ItemKNNScorer`) was constant across seeds:
  - `NDCG@1`: `0.301166...`
  - `NDCG@5`: `0.253753...`
  - `NDCG@10`: `0.222300...`
- MostPopular (`LensKit.PopScorer`) was constant across seeds:
  - `NDCG@1`: `0.216331...`
  - `NDCG@5`: `0.170169...`
  - `NDCG@10`: `0.149590...`

### Dataset/algorithm/cutoff aggregate statistics

The output includes an aggregate table with mean, std, min, max, and range, but the printed table is truncated in the excerpt and not fully readable here. The visible summary indicates:

- ItemKNN and MostPopular had `std = 0` or effectively `0` across seeds.
- ALS had nonzero spread across seeds in the raw row table, though the printed grouped analysis object inconsistently shows zero for some entries due to grouping/printing issues.

## Limitations

- Only **MovieLens100K** was run; **Amazon Video Games** and **Last.FM** were not processed or evaluated.
- The code created a validation split (`validation_size=0.15`), which conflicts with the request for only an 80/20 train/test holdout and no validation partition.
- The exact schema mapping for the Amazon and Last.FM files was not performed in the provided run.
- The output does not provide a complete, cleanly formatted per-row result table for all 135 rows in a way that can be reconstructed exactly from the excerpt alone.
- The code reports `Recall` from `omnirec` but relabels it as `Precision`; therefore the “Precision” column in the final table is based on the library’s recall metric object, not a separately documented precision computation in the shown code.
- Although the code filtered out non-finite values, the excerpt does not independently show every stored row to verify all 135 were printed completely; however, the run did state that finite values were required and the displayed tables contain numeric values.

## Conclusion

This run partially satisfies the request only for **MovieLens100K**. It successfully used local data, applied implicit conversion and 5-core pruning, evaluated the three requested algorithm classes, and recorded package versions. However, it did **not** process Amazon Video Games or Last.FM, and it used a validation split contrary to the requested 80/20-only holdout. The strongest factual takeaway from the output is that, on MovieLens100K, **ItemKNN outperformed MostPopular**, and **ALS was more seed-sensitive than the other two algorithms**.