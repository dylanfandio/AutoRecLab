# Preregistered phase — native-path control protocol (predeclared)

Written **before** running, so success cannot be redefined after the fact.
The human positive control is frozen in `freeze_positive_control.json`.

The preregistered phase asks a single question: **with the task proven achievable (by the positive control),
can the AutoRecLab agent itself reach a valid 270/270 matrix under fixed
conditions?** It is a control, not a search for a better configuration.

## Frozen parameters — identical for all five repetitions

| Parameter | Value |
| --- | --- |
| Prompt file | `study/prompts/P2_compatible_detailed.txt` |
| Prompt SHA-256 (head) | `967d4987dc088da3…` |
| Code-base commit | `ff5f89870e8e7ba7211e62f994a6bfb5f389012d` |
| Model | `gpt-5.4-mini` |
| Max iterations | 5 (`config.toml [treesearch] max_iterations`) |
| Exec timeout | 5400 s (`config.toml [exec] timeout`) |
| Datasets | `u.data`, `VideoGames.csv`, `UserTaggedArtiststimestamps.dat` (hashes in `study/data_checksums.csv`) |
| Seeds | 7, 13, 29, 42, 87 (declared inside P2; do not override) |
| Repetitions | 5 (R01–R05) |

**No parameter above changes between R01 and R05.** The prompt file is not
edited. `config.toml` is not edited between runs.

## Predeclared success criterion

A repetition **succeeds** if and only if its produced results, checked by the
validator, show:

1. **270/270** unique metric entries
   (3 datasets × 3 algorithms × 5 seeds × 3 cutoffs × 2 metrics), and
2. **zero semantic violations** under the positive-control validator, adapted only to
   read the agent's native output format and to expect seeds {7,13,29,42,87}.

Anything else — 269/270, an all-zero block, a mock recommender, copied
Precision/Recall columns, a swallowed evaluation, NaNs — is a **failure**, no
matter how complete the agent's own summary claims to be. This is the same bar
the positive control cleared.

The predeclared outcome hypothesis (to be confirmed or refuted, not adjusted):
based on the audit's 0/19, **≥4 of 5 repetitions are expected to fall short of
270/270**. Recording this now so a low success rate reads as a result, not a
disappointment.

## Preservation rule

Every run's full output directory is kept, including all intermediate tree
nodes and per-node artifacts — not just the finalized summary. The positive-control
audit showed intermediate nodes are sometimes better than the finalized one, so
discarding them would lose evidence. Nothing under a run directory is deleted or
overwritten.

## Launch (run by the operator, in the uv venv)

Five times, R01 through R05, changing only the output directory name:

```bash
# from repo root
uv run python main.py \
  --prompt-file study/prompts/P2_compatible_detailed.txt \
  --model gpt-5.4-mini \
  > study_logs/V100_P2_MINI_I05_R01.console.log 2>&1
# then move ./out to study_runs/V100_P2_MINI_I05_R01, and repeat for R02..R05
```

(`out_dir` is `./out` from `config.toml`; rename it to the run label after each
run, matching the existing `study_runs/` naming convention.)

## Cost

Estimated ~USD 0.75–1.00 per run from prior Mini runs, ~USD 4 for five. Billed
to the operator's OpenAI key; not incurred by the positive control.
