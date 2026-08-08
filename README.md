# Can AI Run Its Own Experiments? — AutoRecLab Replication & Reliability Study

Team 2 fork of [AutoRecLab](https://github.com/ISG-Siegen/AutoRecLab), produced
for the Machine Learning Lab (Praktikum Maschinelles Lernen, SoSe 2026) at the
University of Siegen. This fork adds an **artifact-based replication and
reliability evaluation** of AutoRecLab. The upstream agent code is used as
released; our contribution is the study, its evidence, and its reproducibility
materials, all under [`study/`](study/).

**Authors:** Javier Briceño Ticona, Dylan Fandio, Jawdat Alqawi (Team 2).
**Original system:** Beel et al., *From AutoRecSys to AutoRecLab* — arXiv:[2510.18104](https://arxiv.org/abs/2510.18104).

## What this fork contains

| | |
| --- | --- |
| The paper | *Can AI Run Its Own Experiments? An Artifact-Based Replication and Reliability Evaluation of AutoRecLab* — the final report accompanying this repository. |
| The study | [`study/`](study/) — audit tables, curated evidence, a human positive control, and figure-reproduction scripts. |
| Upstream agent | The unmodified AutoRecLab v1.0.0 code (see [Running AutoRecLab](#running-autoreclab-upstream)). |

## Summary of the study

We evaluate how reliably AutoRecLab turns a natural-language specification into an
executable **and semantically valid** experiment, judged at the level of retained
artifacts rather than reported completion. Across **29 fully retained follow-up
searches** (19 exploratory and 10 preregistered) executed under AutoRecLab
v1.0.0, together with 36 historical replication runs, **no search produced a
coherent, semantically valid 270-entry result matrix**. A **human-authored
positive control completed that same matrix in 49 minutes at no model cost**, so
the failures are not due to task infeasibility. A preregistered compatibility
contract plus a deterministic validator did not repair completion, and the
validator was reinterpreted by the generated code rather than enforced. The
central finding: code execution, output presence, and an approving automated
reviewer are each insufficient evidence that the intended experiment was actually
performed.

## The `study/` directory

All replication evidence and reproducibility material lives under [`study/`](study/):

| Path | Contents |
| --- | --- |
| `study/audit/` | Machine-readable audit tables for the 19 exploratory runs (`file_manifest.csv` is the SHA-256 inventory of the full workspaces). |
| `study/audit_preregistered/` | Audit tables and node-level coverage / semantic analysis for the 10 preregistered P2/P3 runs. |
| `study/audit_tools/` | The scripts that regenerate the audit tables from the raw run workspaces. |
| `study/reproducibility/` | The figure-generation scripts and their dependencies. |
| `study/positive_control/` | The human-authored feasibility pipeline, its external validator, freeze manifests, protocol, findings, and summary outputs. |
| `study/evidence/` | Curated case-study artifacts (prompt, config, logs, code, native results, and validation reports) for the runs discussed in the paper. |
| `study/study_context/` | The phase-specific prompts (P0–P3), dataset source URLs, and SHA-256 checksums. |
| `study/PACKAGE_MANIFEST.csv` | SHA-256 of every file in the package. |

The complete 47.47 GiB of raw run workspaces are **not** stored in Git; they are
referenced by the SHA-256 inventory instead. Raw datasets are likewise not
redistributed — see `study/study_context/SOURCES.md` for download URLs and
checksums.

## Runs and prompt context

The following table lists all runs contained in `study/evidence/` and links each run to the corresponding prompt in `study/study_context/prompts/`. `I05` denotes five configured iterations and `Rxx` identifies the replicate. Additionally a machine-readable run overview is being provided in `study/evidence/run_inventory_evidence.csv`  [Run - Overview](study/evidence/run_inventory_evidence.csv)

| Run | Prompt | Model / phase | Iter. | Replicate | Brief description |
| --- | --- | --- | ---: | ---: | --- |
| `SMOKE_V100_P0_I01` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / smoke | 1 | — | Excluded smoke run; approximately 3.64 GiB Amazon allocation |
| `V100_P0_GPT54_I05_R01` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / GPT-5.4-mini | 5 | 01 | Second 90-entry MovieLens run; per-seed native result files |
| `V100_P0_MINI_I05_R02` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / GPT-5.4-mini | 5 | 02 | 90-entry MovieLens run; split/threshold deviation |
| `V100_P0_MINI_I05_R03` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / GPT-5.4-mini | 5 | 03 | Recall relabelled as Precision |
| `V100_P0_MINI_I05_R04` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / GPT-5.4-mini | 5 | 04 | `KeyError: 'rating'`; schema/loader mismatch |
| `V100_P0_NANO_I05_R02` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / GPT-5.4-mini | 5 | 02 | OmniRec/LensKit API mismatch |
| `V100_P0_NANO_I05_R05` | [P0 – Original](study/study_context/prompts/P0_original.txt) | P0 / GPT-5.4-mini | 5 | 05 | Mock popularity results under three algorithm labels; no valid execution |
| `V100_P2_GPT54_I05_R02` | [P2 – Compatible detailed](study/study_context/prompts/P2_compatible_detailed.txt) | P2 / GPT-5.4-mini | 5 | 02 | 81.8% reviewer alignment versus 2/270 coverage |
| `V100_P2_MINI_I05_R01` | [P2 – Compatible detailed](study/study_context/prompts/P2_compatible_detailed.txt) | P2 / GPT-5.4-mini | 5 | 01 | Detailed prompt; native metrics, partial coverage, 0 valid |
| `V100_P2_MINI_I05_R02` | [P2 – Compatible detailed](study/study_context/prompts/P2_compatible_detailed.txt) | P2 / GPT-5.4-mini | 5 | 02 | Native execution in AutoRecLab/OmniRec; reproducibility artifacts |
| `V100_P2_MINI_I05_R03` | [P2 – Compatible detailed](study/study_context/prompts/P2_compatible_detailed.txt) | P2 / GPT-5.4-mini | 5 | 03 | Compatible detailed-prompt run; result-generation/validation evidence |
| `V100_P2_MINI_I05_R04` | [P2 – Compatible detailed](study/study_context/prompts/P2_compatible_detailed.txt) | P2 / GPT-5.4-mini | 5 | 04 | Compatible detailed-prompt run; validation/result artifacts |
| `V100_P2_MINI_I05_R05` | [P2 – Compatible detailed](study/study_context/prompts/P2_compatible_detailed.txt) | P2 / GPT-5.4-mini | 5 | 05 | Compatible detailed-prompt run; reproducibility and persistence artifacts |
| `V100_P3_MINI_I05_R01` | [P3 – Contract validated](study/study_context/prompts/P3_contract_validated.txt) | P3 / GPT-5.4-mini | 5 | 01 | Contract-prompt run; no valid final metrics |
| `V100_P3_MINI_I05_R02` | [P3 – Contract validated](study/study_context/prompts/P3_contract_validated.txt) | P3 / GPT-5.4-mini | 5 | 02 | Contract-prompt run; no valid final metrics |
| `V100_P3_MINI_I05_R03` | [P3 – Contract validated](study/study_context/prompts/P3_contract_validated.txt) | P3 / GPT-5.4-mini | 5 | 03 | Validator-subversion finding; validation report rewrote denominator from 270 to 30 |
| `V100_P3_MINI_I05_R04` | [P3 – Contract validated](study/study_context/prompts/P3_contract_validated.txt) | P3 / GPT-5.4-mini | 5 | 04 | Corrupt validation report |
| `V100_P3_MINI_I05_R05` | [P3 – Contract validated](study/study_context/prompts/P3_contract_validated.txt) | P3 / GPT-5.4-mini | 5 | 05 | Contract-prompt run; no valid final metrics |

### Prompt definitions

| Prompt | File | Purpose |
| --- | --- | --- |
| **P0** | [`P0_original.txt`](study/study_context/prompts/P0_original.txt) | Original LensKit 0.14.4 experiment request using ALS, ItemKNN and Pop on three datasets |
| **P1** | [`P1_simplified.txt`](study/study_context/prompts/P1_simplified.txt) | Simplified version of the original experiment request; no evidence run currently uses P1 |
| **P2** | [`P2_compatible_detailed.txt`](study/study_context/prompts/P2_compatible_detailed.txt) | Detailed, environment-compatible request using the installed AutoRecLab stack instead of requiring LensKit |
| **P3** | [`P3_contract_validated.txt`](study/study_context/prompts/P3_contract_validated.txt) | Contract-oriented prompt adding explicit validation requirements for actual, finite metrics and result integrity |
## Reproducing the figures

The paper figures are built from the committed audit tables (no raw workspaces
needed):

```bash
cd study/reproducibility
python -m pip install -r requirements-figures.txt
python generate_individual_figures.py      # writes to study/figures/individual/
```

`generate_figures.py` is a library dependency of the individual-figures script
and does not need to be run on its own. The scripts in `study/audit_tools/`
reproduce the audit tables themselves, but require the raw run workspaces, which
are not part of the repository.

## Running AutoRecLab (upstream)

The autonomous agent is unchanged from AutoRecLab v1.0.0. In brief:

```bash
# 1. provide an OpenAI key
echo "OPENAI_API_KEY=your-key" > .env

# 2a. local workflow with uv
uv sync
uv run python -m cli.embeddings.main generate --all   # documentation embeddings
uv run main.py

# 2b. or the isolated Docker workflow
docker compose run --build sandbox
```

Common flags: `--prompt-file ./prompt.txt`, `--model gpt-5.4-mini`,
`--list-models`, `--timestamp-out-dir`. Runtime behaviour is configured in
`config.toml`. For full setup, configuration, outputs, and examples, see the
upstream documentation in [docs/README.md](docs/README.md) and
[docs/usage-and-examples.md](docs/usage-and-examples.md).

## Citation and attribution

This work replicates and evaluates the AutoRecLab proof of concept:

> Joeran Beel, Bela Gipp, Tobias Vente, Moritz Baumgart, and Philipp Meister.
> *From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern
> Autonomous Recommender-Systems Research Labs.* arXiv:2510.18104, 2025.

The AutoRecLab system and its source code are the work of the Information Systems
Group, University of Siegen ([upstream repository](https://github.com/ISG-Siegen/AutoRecLab)).
Our replication study, audit tooling, positive control, and figures are the
contribution of Team 2.
