# Experiment Summary

## User Request

The request was to run a reproducible recommender-systems experiment in the existing AutoRecLab v1.0.0 environment using local files only, without requiring LensKit 0.14.4, and to evaluate:

- ALS for implicit feedback
- ItemKNN
- MostPopular

on three datasets:

- MovieLens100K (`u.data`)
- Amazon Video Games (`VideoGames.csv`)
- Last.FM (`UserTaggedArtiststimestamps.dat`)

with:

- implicit conversion for MovieLens100K and Amazon Video Games using ratings `> 3`
- Last.FM tag events treated directly as implicit interactions
- 5-core filtering
- user-based 80/20 holdout
- seeds `7, 13, 29, 42, 87`
- metrics `nDCG` and `Precision` at `k = 1, 5, 10`
- incremental persistence of completed results
- exact package versions, algorithm classes, preprocessing counts, and train/test counts

## What Was Run

The provided experiment code attempted to:

- inspect local files from `../../../study/data/`
- map each raw file according to its schema
- convert interactions to implicit feedback
- apply 5-core filtering
- run a holdout-based evaluation with `omnirec`
- use these algorithm classes:
  - `LensKit.ImplicitMFScorer`
  - `LensKit.ItemKNNScorer`
  - `LensKit.PopScorer`

The run did detect these package versions:

- `omnirec`: `unknown`
- `numpy`: `1.26.4`
- `pandas`: `2.3.3`
- `scipy`: `1.16.2`
- `scikit-learn`: `1.7.1`
- `lenskit`: `2025.6.2`
- `matplotlib`: `3.10.7`

However, the experiment failed before producing any result rows.

## Key Results

| Dataset | Algorithm | Seed | Cutoff | nDCG | Precision |
|---|---|---:|---:|---:|---:|
| MovieLens100K | N/A | N/A | N/A | N/A | N/A |
| Amazon Video Games | N/A | N/A | N/A | N/A | N/A |
| Last.FM | N/A | N/A | N/A | N/A | N/A |

No valid metric rows were produced because the run crashed during dataset loading.

The failure occurred here:

- `RecSysDataSet.load(csv_path)`
- underlying error: `zipfile.BadZipFile: File is not a zip file`

This happened while converting the first processed dataframe to an `omnirec.RecSysDataSet` in `_to_rsds`.

## Limitations

- No completed dataset/algorithm/seed/cutoff evaluations are available.
- No `nDCG` or `Precision` values were produced, so no summary statistics across seeds can be computed.
- The output does not include the exact preprocessing counts, train/test interaction counts, or any successful split/evaluation counts.
- Although the code defines a schema mapping and preprocessing intent, the run terminated before those processed datasets could be evaluated.
- The request mentioned “Do not require or install LensKit 0.14.4”; the detected installed `lenskit` version was `2025.6.2`, but the experiment still did not complete.

## Conclusion

The experiment was launched with the intended local files, preprocessing logic, seeds, and algorithm classes, but it failed before any recommender results were generated. The only factual outputs available are the detected package versions, the intended algorithm implementations, and the `BadZipFile` crash during `RecSysDataSet.load`. Therefore, the requested metrics, per-seed tables, and seed-sensitivity analysis cannot be reported from this run.