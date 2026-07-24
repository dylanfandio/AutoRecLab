# Experiment Summary

## User Request

Run a reproducible recommender-systems experiment with the installed software in AutoRecLab v1.0.0, without requiring LensKit 0.14.4, using the local dataset files in `../../../study/data/`:

- MovieLens100K: `u.data`
- Amazon Video Games: `VideoGames.csv`
- Last.FM: `UserTaggedArtiststimestamps.dat`

Use these algorithms:

- ALS for implicit feedback
- ItemKNN
- MostPopular

Apply the requested preprocessing, 5-core filtering, 5 fixed random seeds, 80/20 user holdout, and report nDCG and Precision at k = 1, 5, 10, plus package versions, algorithm classes, counts, and seed sensitivity.

## What Was Run

The provided experiment code attempted to:

- Detect package versions for `omnirec`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn`
- Map each raw file to the expected schema
- Convert MovieLens100K and Amazon Video Games to implicit feedback using threshold `> 3`
- Treat Last.FM tagging events as implicit interactions directly
- Apply 5-core pruning
- Evaluate:
  - `LensKit.ImplicitMFScorer`
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`
- Use seeds `7, 13, 29, 42, 87`
- Run user-based 80/20 holdout
- Measure `NDCG` and `Precision` at `1, 5, 10`
- Save results incrementally

The run started successfully for MovieLens100K preprocessing and reported:

- Before implicit conversion: `100000` interactions
- After implicit conversion: `82520`
- After 5-core pruning: `81697`

However, the experiment then crashed while constructing the Amazon Video Games dataset:

- Error: `ValueError: The truth value of a DataFrame is ambiguous`
- Location: `RecSysDataSet(raw_df)`

Because of this failure, the full experiment did not complete.

## Key Results

| Item | Result |
|---|---|
| MovieLens100K implicit conversion | 100000 → 82520 interactions |
| MovieLens100K after 5-core pruning | 81697 interactions |
| Amazon Video Games processed | Not completed |
| Last.FM processed | Not completed |
| Algorithms evaluated successfully | None completed end-to-end |
| nDCG / Precision results | N/A — no metric rows were produced before the crash |
| Seed summary | N/A — no completed seed-level results |
| Package versions | N/A in output; the run failed before metadata was printed |
| Exact algorithm classes | `omnirec.runner.algos.LensKit.ImplicitMFScorer`, `omnirec.runner.algos.LensKit.ItemKNNScorer`, `omnirec.runner.algos.LensKit.PopScorer` |

## Limitations

- The experiment did **not** complete, so there are **no valid final metrics tables** to report.
- No completed train/test result rows are available for the requested dataset–algorithm–seed–cutoff combinations.
- The output does not include the requested package-version metadata because the program crashed before reaching the metadata-writing stage.
- The crash occurred when creating `RecSysDataSet(raw_df)` for Amazon Video Games, so the requested evaluation over all three datasets was not executed.
- Although the code was designed to persist partial results, the provided output does not show any saved result rows or summary tables.
- The exact raw-file schema mapping was only partially exercised in execution; only MovieLens100K preprocessing is explicitly confirmed by output.

## Conclusion

The experiment setup matches the requested design in code, including the intended algorithms, preprocessing, seeds, and metrics. But the actual run failed after preprocessing MovieLens100K and before any algorithm evaluation or metric reporting could complete. Therefore, the only factual results available are the MovieLens100K preprocessing counts and the exact algorithm class names used in the code. All requested accuracy tables and seed-effect summaries remain unavailable from the provided output.