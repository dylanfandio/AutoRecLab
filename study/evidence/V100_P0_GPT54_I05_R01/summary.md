# Experiment Summary

## User Request

You asked for a seed-sensitivity experiment using LensKit 0.14.4 on three algorithms—ALS, ItemKNN, and Pop—across three implicit-feedback datasets: MovieLens100K, Amazon Video Games, and Last.FM.

The requested procedure was:

- preprocess each dataset with 5-core filtering
- for MovieLens100K and Amazon Video Games, convert ratings greater than 3 to implicit interactions
- use 5 random split seeds
- perform a user-based 80/20 holdout split
- train with standard hyperparameters
- report nDCG@1, @5, @10 and Precision@1, @5, @10
- include a short statistical analysis of seed effects

## What Was Run

From the provided code and output, the experiment actually ran as follows:

- Framework: OmniRec runner with LensKit algorithms
- Algorithms added:
  - `LensKit.PopScorer`
  - `LensKit.ItemKNNScorer`
  - `LensKit.ImplicitMFScorer`  
    This corresponds to Pop, ItemKNN, and an implicit matrix factorization model used in place of ALS by the experiment code.
- Seeds configured: `7, 13, 29, 42, 87`
- Datasets configured:
  - `MovieLens100K`
  - `AmazonVideoGames`
  - `LastFM`
- Preprocessing:
  - MovieLens100K and AmazonVideoGames: `MakeImplicit(3)`
  - all datasets: `CorePruning(5)`
  - split: `UserHoldout(0.01, 0.2)`

Important implementation detail from the output:
- the split was not a pure 80/20 holdout; it used `UserHoldout(validation_size=0.01, test_size=0.2)`, producing train/validation/test splits.
- For MovieLens100K after preprocessing, the output shows:
  - 100,000 interactions before implicit conversion
  - 82,520 after implicit conversion
  - 81,697 after 5-core pruning
  - split sizes: train 63,624 / val 1,366 / test 16,707

Execution status:
- MovieLens100K completed for all 5 seeds
- AmazonVideoGames failed at the first seed with a download/loading error
- LastFM was not reached because the program crashed after the Amazon failure

## Key Results

Only MovieLens100K produced usable results. The output tables do not expose algorithm names alongside each row in a readable way, so the exact row-to-algorithm mapping is ambiguous from the provided log alone. The run order in code was Pop, ItemKNN, then ImplicitMF, and the metric rows appear in that same order, but the printed table truncates names. To avoid guessing, the table below reports the three result rows as Row 1–3.

### Per-seed results available from the output

| Dataset | Seed | Result row | nDCG@1 | nDCG@5 | nDCG@10 | Precision@1 | Precision@5 | Precision@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MovieLens100K | 7 | Row 1 | 0.22163 | 0.20465 | 0.19097 | 0.22163 | 0.19979 | 0.18261 |
| MovieLens100K | 7 | Row 2 | 0.40827 | 0.32125 | 0.27385 | 0.40827 | 0.29714 | 0.24115 |
| MovieLens100K | 7 | Row 3 | 0.27147 | 0.18792 | 0.17367 | 0.27147 | 0.16904 | 0.15811 |
| MovieLens100K | 13 | Row 1 | 0.20572 | 0.20143 | 0.19087 | 0.20572 | 0.19746 | 0.18431 |
| MovieLens100K | 13 | Row 2 | 0.39448 | 0.30883 | 0.26812 | 0.39448 | 0.28526 | 0.23839 |
| MovieLens100K | 13 | Row 3 | 0.23860 | 0.18375 | 0.17060 | 0.23860 | 0.16967 | 0.15779 |
| MovieLens100K | 29 | Row 1 | 0.23011 | 0.21376 | 0.20072 | 0.23011 | 0.20912 | 0.19268 |
| MovieLens100K | 29 | Row 2 | 0.39024 | 0.30419 | 0.26288 | 0.39024 | 0.28229 | 0.23425 |
| MovieLens100K | 29 | Row 3 | 0.28525 | 0.20013 | 0.18602 | 0.28525 | 0.18049 | 0.16999 |
| MovieLens100K | 42 | Row 1 | 0.21951 | 0.20305 | 0.19163 | 0.21951 | 0.19852 | 0.18420 |
| MovieLens100K | 42 | Row 2 | 0.39660 | 0.30896 | 0.26829 | 0.39660 | 0.28484 | 0.23807 |
| MovieLens100K | 42 | Row 3 | 0.25132 | 0.18973 | 0.17357 | 0.25132 | 0.17285 | 0.15822 |
| MovieLens100K | 87 | Row 1 | 0.22587 | 0.20281 | 0.19388 | 0.22587 | 0.19661 | 0.18685 |
| MovieLens100K | 87 | Row 2 | 0.38918 | 0.29599 | 0.25469 | 0.38918 | 0.27296 | 0.22503 |
| MovieLens100K | 87 | Row 3 | 0.23011 | 0.18576 | 0.16597 | 0.23011 | 0.17688 | 0.15334 |

### Short statistical readout from the available results

Across the 5 MovieLens100K seeds only:

- The best-performing result row was consistently Row 2 on every reported metric.
- Seed variation was present but modest relative to the metric scale.
- Approximate ranges across seeds:
  - Row 1:
    - nDCG@10: 0.19087 to 0.20072
    - Precision@10: 0.18261 to 0.19268
  - Row 2:
    - nDCG@10: 0.25469 to 0.27385
    - Precision@10: 0.22503 to 0.24115
  - Row 3:
    - nDCG@10: 0.16597 to 0.18602
    - Precision@10: 0.15334 to 0.16999
- The largest observed spread among these reported metrics appears in Row 2 and Row 3 depending on metric, but the experiment output does not include the final computed summary/statistics tables because execution stopped before they were printed.

## Limitations

- The experiment did not complete.
- AmazonVideoGames failed immediately with:
  - `Download failed for LocalAmazonVideoGames. None of the provides download urls were successful!`
- Because of that crash, no Amazon Video Games results were produced, and Last.FM was never run.
- The requested cross-dataset comparison and full seed-sensitivity analysis therefore cannot be completed from the provided output.
- The code requested ALS, but the actual algorithm instantiated in code was `LensKit.ImplicitMFScorer`; based on the provided materials alone, I can only report what was run, not equate it definitively to ALS.
- The split was not exactly a simple 80/20 holdout: the code used `UserHoldout(0.01, 0.2)`, so there was also a small validation split.
- The printed evaluation tables truncate algorithm names, so exact row-to-algorithm labeling is not explicitly visible in the output.

## Conclusion

The provided run only completed for MovieLens100K across 5 random seeds. On that dataset, all three reported result rows showed some seed-dependent variation, but performance stayed in relatively narrow bands across seeds. However, the requested full experiment was not achieved: Amazon Video Games failed to load, Last.FM never ran, and no final aggregate statistical tables were printed before the crash. As a result, only a partial conclusion is supported: split random seeds affected MovieLens100K results to a limited but noticeable extent, while no factual conclusion can be drawn for Amazon Video Games or Last.FM from the provided output.