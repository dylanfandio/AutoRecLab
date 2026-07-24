# AutoRecLab reliability audit -- evidence and reproducibility package

The committable evidence and reproducibility bundle for the paper. The paper
source is maintained in Overleaf and is not included here.

## Contents

- `audit/` -- audit tables for the 19 exploratory follow-up runs (evidence root; `file_manifest.csv` is the SHA-256 inventory of the full workspaces).
- `audit_preregistered/` -- audit tables and node-level coverage/semantic analysis for the 10 preregistered P2/P3 Mini runs.
- `audit_tools/` -- scripts that regenerate both audits from the workspaces.
- `reproducibility/` -- figure scripts (see its README).
- `positive_control/` -- the human-authored feasibility control: pipeline, external validator, freeze manifests, protocol, findings, and summary outputs.
- `evidence/` -- curated case-study artifacts (see `evidence/README.md`).
- `study_context/` -- prompts P0-P3, dataset checksums, and source URLs.

## Scope

Covers the 19 exploratory and 10 preregistered fully retained follow-up runs, plus one excluded smoke run and one human positive control. Raw datasets and full workspaces are referenced by checksum, not redistributed.

## Reproduce the figures

```text
cd reproducibility
python -m pip install -r requirements-figures.txt
python generate_individual_figures.py
```

`PACKAGE_MANIFEST.csv` lists the SHA-256 of every file in this package.
