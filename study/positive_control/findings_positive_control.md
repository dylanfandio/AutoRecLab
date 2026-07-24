# Positive control — findings

Generated 2026-07-24. Reproduce with the commands in `README.md`.
Model-API cost: **USD 0.00**.

## Headline

**The requested experiment is achievable in this environment, and it is cheap.**

A hand-written reference implementation produced all **270/270** requested
metric entries, with **0 semantic violations** from an independent validator,
in **49.4 minutes** of wall time, peaking at **737 MiB** RSS on a 31.8 GiB
host. Nothing about the task is intrinsically hard for this machine.

So the 0/19 completion rate is not explained by the task being impossible.

**But the task as literally specified is not executable on the path AutoRecLab
was asked to use.** Eight independent structural blockers sit between the prompt
and a valid result, and five of them fail silently — they produce numbers, or
no numbers and no error, rather than a diagnosable failure. That combination is
the more damaging finding: an agent can satisfy a reviewer checklist while
producing results that cannot be correct.

On clean checkpoints the native OmniRec path produces valid ranking metrics on
**1 of the 3 datasets** (MovieLens only), and even that one's nDCG is
systematically wrong (B6). LastFM completes with no results and no raised
error (B7 + B8); Amazon is infeasible (B3).

## Part 1 — the positive control

| | |
| --- | --- |
| Requested entries | 270 (3 datasets x 3 algorithms x 5 seeds x 3 cutoffs x 2 metrics) |
| Produced | 270, each exactly once |
| Validator checks passed | 45 / 46 (1 benign warning) |
| Semantic violations | 0 |
| Total wall time | 49.4 min |
| Peak RSS, worst cell | 737 MiB (Amazon / ItemKNN) |
| Host | 31.8 GiB RAM |

Per-cell cost, averaged over seeds:

| Dataset | Algorithm | Wall (s) | Peak RSS (MiB) |
| --- | --- | ---: | ---: |
| MovieLens100K | ALS / ItemKNN / Pop | 24.2 / 23.4 / 23.4 | 364 / 378 / 370 |
| HetrecLastFM | ALS / ItemKNN / Pop | 22.6 / 23.5 / 22.2 | 518 / 544 / 501 |
| AmazonVideoGames | ALS / ItemKNN / Pop | 154.4 / 156.4 / 142.8 | 681 / 705 / 675 |

### Preprocessing counts

| Dataset | Raw | After dedup | After `rating > 3` | After 5-core | Users | Items |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MovieLens100K | 100,000 | 100,000 | 55,375 | 54,413 | 938 | 1,008 |
| AmazonVideoGames | 2,565,349 | 2,489,395 | 1,844,699 | 291,945 | 33,621 | 12,455 |
| HetrecLastFM | 186,479 | 71,064 | n/a | 52,551 | 1,090 | 3,646 |

Every figure was recomputed by `pc_validate.py` from the raw files using an
independent implementation, and matched.

### The answer to the original research question

Data-split seed materially affects measured accuracy, and the effect is largest
exactly where the data is sparsest and the algorithm weakest. Relative range of
nDCG@10 across the five seeds:

| Dataset | ALS | ItemKNN | Pop |
| --- | ---: | ---: | ---: |
| MovieLens100K | 7.7% | 4.6% | 4.0% |
| HetrecLastFM | 21.1% | 14.8% | **58.5%** |
| AmazonVideoGames | 28.7% | 8.8% | **34.2%** |

A single-seed evaluation on Amazon or LastFM can move nDCG@10 by a quarter to
a half of its own value. This is a publishable result on its own, and the
control produced it as a side effect.

## Part 2 — eight structural blockers on the requested path

### B1. LensKit 0.14.4 does not exist in this stack (fails loudly)

The prompt asks for LensKit 0.14.4. The pinned environment has
`lenskit==2025.6.2` ([pyproject.toml](../../pyproject.toml)), and OmniRec's
runner builds a *separate* Python 3.11 venv pinned to `lenskit==2025.2.0`
([runner/registry.py](../../.venv/Lib/site-packages/omnirec/runner/registry.py)).
The 0.14.x API (`lenskit.algorithms.*`) shares almost no surface with 2025.x
(`lenskit.knn`, `lenskit.als`, pipelines). Every run was unsatisfiable at this
clause before any code was written.

### B2. The requested split cannot be expressed (fails loudly)

The prompt asks for an 80/20 holdout. OmniRec's `UserHoldout` always carves a
validation slice out of train, and `validation_size=0` reaches
`train_test_split(test_size=0.0)`, which raises `InvalidParameterError`.
There is no parameterisation of `UserHoldout` that yields train/test only.

This matches the audit's observation that runs "attempted a zero-sized
validation set, which the installed stack rejects" and then "silently used
non-requested proportions to satisfy the framework".

### B3. Amazon cannot be evaluated through the runner (fails loudly, late)

`omnirec_runner/lenskit_runner.py` calls `recommend(self.model, self.test)`
with **no list length**. In LensKit 2025.x, `n=None` ranks every candidate item
for every user. For Amazon after 5-core filtering that is

    33,621 users x 12,455 items = 418,749,555 rows  (~12.5 GiB in memory)

before the coordinator serialises it to `predictions.json`. Tier A refuses this
cell by projection; the audit records the real runs hitting an actual ~3.64 GiB
allocation failure here, which corroborates.

Tier B, issuing `recommend(..., n=10)`, evaluates the same dataset in 154 s at
705 MiB. **The dataset is not the problem; the missing argument is.** The same
defect makes even MovieLens ~9x slower on the native path (208 s for 3
algorithms from clean checkpoints vs ~23 s per cell in Tier B).

### B4. All five seeds share one checkpoint directory (fails silently)

```python
def dataset_hash(dataset): return sha256(json.dumps(dataset.num_interactions()))
```

The key is derived only from train/val/test partition *sizes*. A per-user 80/20
holdout produces identical partition sizes for every seed, so all five seeds
hash to the same checkpoint directory, and seeds 2-5 resume seed 1's model and
predictions.

A seed-sensitivity study is precisely the experiment this collides on. Run on
the native path with default checkpointing, it would report five identical
results and a seed effect of exactly zero. Tier A gives each seed its own
`checkpoint_dir` to work around it.

### B5. Model training is never seeded (fails silently)

The runner calls `self.model.train(dataset, TrainingOptions())` — no `rng`. For
a stochastic learner like ALS the model is therefore unseeded, so variation
across "seeds" mixes split variation with uncontrolled training variation. The
requested experiment isolates split-seed effects; on the native path that
isolation is impossible.

Visible in the Tier A/Tier B comparison below: deterministic ItemKNN Precision
agrees to 0.000000, while ALS disagrees at every cutoff.

### B6. OmniRec's nDCG cannot reach 1.0 (fails silently) — most serious

`omnirec/metrics/ranking.py` computes

    IDCG@k = sum_{i=1..k} 1/log2(i+1)

always over the full k, regardless of how many relevant items the user has.
The correct ideal list is truncated at `min(k, |Rel(u)|)`.

Demonstrated directly:

| | perfect ranking, user with 1 relevant item |
| --- | --- |
| LensKit nDCG@10 | **1.000** |
| OmniRec nDCG@10 | **0.220** |

For any user with fewer than k relevant test items — the overwhelming majority
under 5-core filtering — OmniRec's nDCG@k is understated, increasingly so with
k and with sparsity. **Every nDCG number ever produced through OmniRec is
systematically depressed and is not comparable to any published nDCG value.**

### B7. Evaluation errors are swallowed (fails silently) — near-invisible

`Coordinator.run()` wraps the whole experiment in

```python
except Exception:
    traceback.print_exc()
    exception_occurred = True
finally:
    self.stop(...)
self._evaluator.save_results(...)
return self._evaluator
```

An exception during evaluation is printed to stderr and then *discarded*:
`run_omnirec` returns normally with an evaluator that has zero rows. A caller —
human or agent — that checks only "did it raise?" sees success. This is the
audit's "ran without a recorded exception" population: 78/188 nodes, many empty.
Tier A only caught it because it explicitly checks the evaluator row count and
re-captures stderr; the first, naive version of Tier A reported LastFM as `ok`.

### B8. Feedback type is inferred from a column name (fails silently)

The runner decides implicit vs explicit by

```python
if "rating" in self.train.columns:  # -> explicit, rating prediction
else:                                # -> implicit, top-N recommendation
```

LastFM is already implicit and is never passed through `MakeImplicit`, so it
keeps its constant `rating = 1` column. The runner therefore misclassifies it
as *explicit*, runs rating prediction instead of top-N, and produces output with
no `rank` column. The ranking metric then dies with `KeyError: 'rank'` — which
B7 swallows. Reproduced on the clean run:

    HetrecLastFM/seed=42 -> completed_without_results (KeyError: 'rank', swallowed)

MovieLens survives only because `MakeImplicit(4)` happens to drop its rating
column. The distinction is accidental, not designed.

## Part 3 — Tier A vs Tier B cross-check (MovieLens, seed 42)

Both tiers share preprocessing and splitting exactly; they differ only in the
train/predict/evaluate backend. This isolates B5 and B6.

Clean run (no resumed checkpoints); MovieLens is the only dataset the native
path yields metrics for.

| Algorithm | Metric | Tier A (OmniRec) | Tier B (LensKit) | Diff |
| --- | --- | ---: | ---: | ---: |
| ItemKNN | Precision@1 | 0.303838 | 0.303838 | **0.000000** |
| ItemKNN | Precision@5 | 0.227079 | 0.227079 | **0.000000** |
| ItemKNN | Precision@10 | 0.183582 | 0.183582 | **0.000000** |
| ItemKNN | nDCG@5 | 0.243470 | 0.255076 | 0.011607 |
| ItemKNN | nDCG@10 | 0.207503 | 0.252049 | **0.044546** |
| Pop | Precision@1 | 0.213220 | 0.213220 | **0.000000** |
| Pop | nDCG@10 | 0.138584 | 0.156928 | 0.018344 |
| ALS | nDCG@1 | 0.200426 | 0.202559 | 0.002132 |
| ALS | Precision@10 | 0.147655 | 0.144670 | 0.002985 |

Reading:

* **Precision matches to 0.000000** for the deterministic algorithms (ItemKNN,
  Pop) at every cutoff. The two implementations are genuinely equivalent; this
  validates the reference pipeline against the framework.
* **nDCG diverges, and the gap grows with k** (0.000 at k=1, 0.012 at k=5,
  0.045 at k=10 for ItemKNN) — the signature of B6.
* **ALS disagrees even at k=1**, where the metric definitions coincide, and by
  varying amounts between runs — the signature of B5 (unseeded training).

## Part 4 — semantic traps the prompt sets

These are decision points where a defensible-looking choice silently changes
the result. Each is a place the audited runs diverged.

**Threshold interpretation.** "Ratings greater than 3" is `MakeImplicit(4)`,
since `MakeImplicit(t)` keeps `rating >= t`. The off-by-one is expensive:

| | MovieLens interactions retained |
| --- | ---: |
| `MakeImplicit(3)` — what the runs used | 82,520 |
| `MakeImplicit(4)` — what the prompt asks | **55,375** |

**Order of operations.** The prompt says 5-core "first", then describes the
implicit conversion. Threshold-then-core and core-then-threshold both read as
defensible, and they differ materially:

| Dataset | Threshold then core | Core then threshold | Delta |
| --- | ---: | ---: | ---: |
| MovieLens100K | 54,413 | 55,165 | 1.4% |
| AmazonVideoGames | 291,945 | 360,015 | **23.3%** |

The control uses threshold-then-core (matching OmniRec's own documented example)
and records both.

**"80/20" is unreachable per-user.** User interaction counts are integers and
sklearn rounds the per-user test size up, so the realised aggregate exceeds 20%:

| Dataset | Realised test fraction | Users with exactly 5 interactions |
| --- | ---: | ---: |
| MovieLens100K | 20.68% | 0.4% |
| HetrecLastFM | 20.84% | 7.2% |
| AmazonVideoGames | **24.10%** | **32.4%** |

The control reports this rather than reshaping the split to hit 20% — silently
reshaping is exactly what the audit criticised. The validator verifies the
per-user rule `ceil(0.2 x n_u)` exactly, for every user, independently.

## What this licenses you to say

Supported:

* The task was achievable in the environment: 270/270, 0 violations, 49 min,
  under 1 GiB, at zero API cost. A positive control exists.
* AutoRecLab's failures are **not** primarily explained by task infeasibility.
* They are **partly** explained by the environment: eight structural blockers,
  three of which (B1, B2, B3) make the literal specification unexecutable on
  the requested path no matter how good the agent is.
* Five blockers (B4, B5, B6, B7, B8) corrupt results without a diagnosable
  failure, so agent output and reviewer scores cannot be taken at face value.
  B6 means reported nDCG values are not comparable across frameworks; B7+B8
  mean the native path returns *no error and no results* on LastFM.
* On clean checkpoints the native path yields valid ranking metrics on only
  1 of 3 datasets, and even that one's nDCG is depressed by B6.
* Data-split seed materially affects accuracy (4-58% relative range in nDCG@10),
  which answers the study's original research question.

Not supported by this control:

* Any claim about which model or prompt is better — this is one hand-written
  pipeline, not a comparison.
* That fixing the eight blockers would raise AutoRecLab's completion rate. That
  is a hypothesis this control makes testable, not a result.
* An observed (as opposed to projected) Amazon failure on the native path. Run
  `pc_tier_a.py --datasets AmazonVideoGames --force` to capture the real
  traceback; expect it to attempt ~12.5 GiB.

## Compute cost of producing this control

| Phase | Compute | Notes |
| --- | ---: | --- |
| Tier B, full 270-entry matrix | 49.4 min | 45 cells; Amazon is 38 min of it |
| Tier A, MovieLens probe | 3.5 min | clean checkpoints, 3 algorithms |
| Tier A, LastFM probe | 3.3 min | completes without results (B7+B8) |
| Tier A, Amazon probe | 0 s | refused by projection, no compute |
| **Total** | **~56 min** | single workstation, 31.8 GiB RAM |

Model-API cost is zero throughout; this is a hand-written pipeline. Wall-clock
between the two runs was longer only because of idle waiting, not computation.
