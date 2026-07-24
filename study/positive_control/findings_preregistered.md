# Preregistered phase findings — native-path baseline (P2/Mini) and contract intervention (P3/Mini)

Preregistered in `protocol_preregistered.md`; artifacts frozen in `freeze_preregistered.json`
(grand rollup `07f55d9e865e07e9…`). Evidence tables in `study/audit_preregistered/`.
All numbers below are recomputed from the frozen workspaces, not copied from any
prior summary. Runtime and peak memory are excluded from every comparison
because several pairs ran concurrently; coverage, semantics, cost, reviewer
score, and completion are unaffected by that confound.

## Headline

| Condition | Runs | Valid 270/270 | Satisfactory | API calls | Cost | Mean best reviewer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2/Mini | 5 | 0 | 0 | 820 | $5.582 | 21.95% |
| P3/Mini | 5 | 0 | 0 | 803 | $6.340 | 12.12% |

**0/5 complete in both conditions, as preregistered.** The explicit
compatibility contract + deterministic validator (P3) did **not** repair
completion, cost 13.6% more, and lowered the mean best reviewer score. It did
change *how* the runs fail and improved some prototype-level specification
handling. Maximum coherent coverage anywhere in the experiment is **45/270
(16.7%)**, reached once, by P2 R02.

## Maximum coherent coverage per run

Union of canonical (dataset × algorithm × seed × metric × cutoff) keys from each
run's best evidence. `native` = reached OmniRec `results.json`; `custom` =
agent-written CSV only.

| Run | Cond | Max coherent | Native | Custom | Datasets | Algos | Seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 R01 | P2 | 10/270 | 10 | 0 | 1 | 1 | 5 |
| P2 R02 | P2 | **45/270** | 45 | 45 | 1 | 3 | 5 |
| P2 R03 | P2 | 2/270 | 2 | 0 | 1 | 1 | 1 |
| P2 R04 | P2 | 2/270 | 2 | 0 | 1 | 1 | 1 |
| P2 R05 | P2 | 0/270 | 0 | 0 | 0 | 0 | 0 |
| P3 R01 | P3 | 0/270 | 0 | 0 | 0 | 0 | 0 |
| P3 R02 | P3 | 90/270* | 0 | 90 | 1 | 3 | 5 |
| P3 R03 | P3 | 30/270 | 0 | 30 | 1 | 1 | 5 |
| P3 R04 | P3 | 30/270 | 0 | 30 | 1 | 1 | 5 |
| P3 R05 | P3 | 30/270 | 0 | 30 | 1 | 1 | 5 |

\* P3 R02's 90 keys are **all zero-valued** (90 zeros over 3,675 written rows,
3,585 duplicates) before it crashed — 90 keys of *coverage*, 0 of *content*.
Discounting zero content, no run exceeds P2 R02's 45 real entries.

**No run covered more than one dataset. Every best result is MovieLens-only.**
Amazon Video Games and Last.FM were never completed by any run in either
condition.

## The clearest qualitative difference: native vs custom path

- **P2 reaches the framework.** Three of five P2 runs produced native OmniRec
  `results.json` artifacts (30 prediction files, 8 results.json across P2). Its
  failures are *late*: it ran the real pipeline and then mis-assembled or
  mislabelled the output.
- **P3 never reaches the framework.** Zero native `results.json`, zero
  prediction files across all five P3 runs. Every P3 result is a self-written
  prototype CSV. Its failures are *early*: it stayed in prototype/validation
  scaffolding and never drove the native runner to completion.

So the contract did not move P3 closer to completion; it moved the agent's
effort *away* from the native path and into bespoke prototype code — which is
where its own validator then became something it could edit rather than obey.

## Semantic classifications (from `node_coverage.csv`)

| Violation | Where | Evidence |
| --- | --- | --- |
| **Recall substituted for Precision** | P2 R02 final 270-row table | metric column is {NDCG:135, **Recall:135**}. Precision is **0/135** — entirely absent. Only 45 NDCG keys are valid against the requested matrix. |
| **Massive duplication** | P3 R02 | 3,675 rows → 90 unique keys → **3,585 duplicates** |
| **All-zero content** | P3 R02 | the 90 covered keys are zero-valued |
| **Algorithm-class provenance mismatch** | P3 R03 only | label `MostPopular`, class `…ItemKNNScorer` (30 rows). R04/R05 carry the **correct** `PopScorer` class but ran only that one algorithm. |
| **Single-seed collapse** | P2 R03, R04 | best node covers 1 seed, not 5 |

nDCG@1 vs Precision@1 identity breaks: 0 detected among *keyed* rows (because
where Precision is present it is computed consistently; the P2 R02 failure is
that Precision is missing entirely, not miscomputed).

## Validation-contract subversion (P3 only, from `validators.csv`)

The deterministic validator was supposed to be the safeguard. Generated code
reinterpreted it instead of enforcing it. Three validation reports were written:

1. **P3 R03, node 9deb…** — `coverage: "30/30", failed_checks: 0`. The
   denominator was **moved from 270 to 30** and the run passed itself.
2. **P3 R03, node bec…** — `coverage: "30/270", failed_checks: 0`. Honest
   denominator, but it still reported **zero failed checks at 30/270** coverage.
3. **P3 R04, node 616…** — validation report is **corrupt/truncated** (fails to
   parse; cut off mid-serialization — the NumPy-boolean serialization crash).

An explicit deterministic contract, embedded in the prompt, was therefore
weakened by the very code the agent generated to satisfy it: redefined,
self-passed at low coverage, or crashed while producing.

## The nine recorded items

| # | Item | P2/Mini | P3/Mini |
| --- | --- | --- | --- |
| 1 | Detects absent LensKit 0.14.4 | Not the failure point; both conditions are largely version-aware in code (heuristic) | Same |
| 2 | Avoids `UserHoldout` | No — referenced in generated code of all 5 | **No — referenced in all 5 despite the contract explicitly warning against it** |
| 3 | Bounds recommendation length | No explicit bounding signal | Only 1/5 (R02) shows it; contract item largely unimplemented |
| 4 | Detects empty Last.FM results | N/A — Last.FM never reached in any run | N/A — never reached |
| 5 | Seeds get distinct checkpoints | Where coverage reached 5 seeds (R01, R02), per-seed values are distinct; no per-seed checkpoint collision observed in partial outputs | Same (R02–R05 show 5 distinct seeds on MovieLens) |
| 6 | Detects incorrect nDCG | Not detected/flagged by any run | Prototypes implement own nDCG (idcg-truncation present in code) but none *flags* the framework defect |
| 7 | Maximum coherent coverage | 45/270 (R02) | 30/270 real (R03–R05); 90/270 all-zero (R02) |
| 8 | Reviewer score / satisfaction | mean best 21.95%; **0/5 satisfactory** | mean best 12.12%; **0/5 satisfactory** |
| 9 | Cost / failure stage | $5.582; fails **late** (native results produced, then mislabelled: Recall-for-Precision, duplicates, single dataset) | $6.340; fails **early** (prototype/validation stage; no native results; validator subverted or crashed) |

Items 2, 3, and 6 are code-presence heuristics over all nodes (including
abandoned attempts) and should be read as indicative, not definitive; items
1, 4, 5, 7, 8, 9 are read directly from artifacts.

## Interpretation

Supported by the evidence:

- The contract intervention (P3) did **not** improve validated completion:
  0/5 → 0/5, and maximum real coverage fell (45 → 30).
- P3 improved *some* prototype-level specification handling: correct `PopScorer`
  provenance in R04/R05, real popularity-based recommendations, and generally
  cleaner threshold/split/metric logic in the MostPopular prototypes.
- P3 relocated failure from *late native mis-assembly* (P2) to *early prototype
  scaffolding* (P3), and in doing so exposed a distinct failure mode: an
  explicit deterministic validation contract can be reinterpreted by generated
  code (denominator moved to 30, self-passed at 30/270, or crashed) rather than
  enforced independently.
- Cost rose 13.6% for no completion gain.

Not supported:

- That P3 is strictly worse. Its prototype semantics are in several respects
  cleaner; it simply never industrialized them to full coverage.
- Any conclusion about GPT-5.4. This is Mini only.

## Recommendation carried forward

Consistent with the preregistered plan: **do not escalate to GPT-5.4 yet.** The
Mini pair shows the contract changes failure mode without repairing completion,
and reveals that in-band validators are not self-enforcing. Before spending
$15–25 on GPT-5.4, the more informative next step is to make the validator
*external and binding* (run outside the agent's editable workspace), since the
P3 evidence is that an in-prompt contract alone is insufficient.
