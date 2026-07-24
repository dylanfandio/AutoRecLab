# Step 0 — human-authored positive control

A reference implementation of the task in `study/prompts/P0_original.txt`,
written by hand and executed under the pinned environment, at zero model-API
cost. Its purpose is to separate two explanations for AutoRecLab's failures:

* the agent could not do it, or
* the requested experiment is not doable in this environment.

The answer turns out to be **both, in separable parts**. See
`FINDINGS.md` for the argument and the numbers.

## Layout

| File | Role |
| --- | --- |
| `pc_common.py` | Raw-file loading, canonicalisation, preprocessing, the 80/20 splitter, resource measurement, environment fingerprinting |
| `pc_tier_b.py` | **The positive control.** Full 270-entry matrix on the pinned LensKit, direct API |
| `pc_tier_a.py` | Ceiling probe on the native OmniRec runner path — the path AutoRecLab was asked to use |
| `pc_validate.py` | Independent validator: 270/270 coverage plus semantic checks |
| `out/` | All produced artifacts (see below) |

## Running it

```bash
cd study/positive_control

# full matrix (~1-2h, dominated by Amazon); resumes from out/cells/ if interrupted
../../.venv/Scripts/python.exe pc_tier_b.py

# native-path probe
../../.venv/Scripts/python.exe pc_tier_a.py --datasets MovieLens100K HetrecLastFM
../../.venv/Scripts/python.exe pc_tier_a.py --datasets AmazonVideoGames   # refuses, with projection

# validate
../../.venv/Scripts/python.exe pc_validate.py
```

`pc_tier_b.py` checkpoints every cell to `out/cells/`, so an interrupted run
resumes rather than recomputing. Pass `--force` to recompute.

## Outputs

| Artifact | Contents |
| --- | --- |
| `out/environment.json` | Full `pip freeze` of the main env *and* of the OmniRec LensKit runner env, git commit, CPU/RAM, seeds, and an explicit record that LensKit 0.14.4 is absent |
| `out/preprocessing_counts.csv` | Per dataset: raw, post-dedup, post-threshold, post-5-core interaction counts, users/items, SHA-256 of both the raw file and the canonical CSV |
| `out/split_counts.csv` | Per dataset x seed: train/test/val interactions, users, items, realised test fraction |
| `out/metrics.csv` | The 270 metric entries, one row each, with a stable `key` |
| `out/resources.csv` | Per dataset x algorithm x seed: wall time and peak RSS, split into fit / predict / evaluate phases |
| `out/cells/*.json` | Per-cell record including the exact algorithm class path and its resolved config |
| `out/validation_report.{json,md}` | Coverage and semantic-violation report |
| `out/tier_a/` | Native-path results, refusal records, and failure tracebacks |

## Decisions that the prompt left ambiguous

These are recorded here because each one is a place where the AutoRecLab runs
diverged, and a positive control is only meaningful if its own choices are
explicit.

1. **"ratings greater than 3"** is implemented as `MakeImplicit(4)`.
   OmniRec's `MakeImplicit(t)` keeps `rating >= t`, so `MakeImplicit(3)` means
   `rating >= 3` — the interpretation the audit found in the runs, retaining
   82,520 of 100,000 MovieLens ratings. The literal reading of the prompt
   retains **55,375**.

2. **Order of operations** is threshold, then 5-core, matching the ordering in
   OmniRec's own documented example pipeline. The count under the reverse order
   is also recorded in `preprocessing_counts.csv`
   (`interactions_reverse_order`) because the prompt's wording does not settle
   it: for MovieLens the two orders give 54,413 vs 55,165.

3. **"standard hyperparameters"** means library defaults. The only config value
   set explicitly is `feedback="implicit"` on ItemKNN, where the component
   exposes it. Every resolved config is dumped per cell.

4. **Five seeds** are fixed at `(42, 43, 44, 45, 46)` and recorded, rather than
   drawn from entropy, so the control is reproducible.

5. **80/20 with no validation set.** OmniRec's `UserHoldout` always carves a
   validation slice out of train and rejects `validation_size=0`; `pc_common`
   supplies a minimal per-user 80/20 splitter instead. Users with fewer than 2
   interactions are kept entirely in train and counted in
   `users_without_holdout` (after 5-core filtering this is zero everywhere).

6. **Evaluation population** is users present in both train and test, matching
   the intersection the OmniRec LensKit runner performs, so Tier A and Tier B
   are comparable.

## Environment

Executed against the repository's own `.venv`, resolved from the same
`uv.lock` as the Docker image (`lenskit 2025.6.2`, `numpy 1.26.4`,
`scikit-learn 1.7.1`). `out/environment.json` captures the full package set so
it can be diffed against a container run for parity.
