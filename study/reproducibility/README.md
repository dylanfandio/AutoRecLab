# Reproducing the paper figures

The paper is maintained in Overleaf; only figure *sources* live here.
`generate_individual_figures.py` renders the paper figures from `../audit/`
(exploratory phase) and `../audit_preregistered/` (preregistered P2/P3 phase) into
`../figures/individual/`.

```text
cd reproducibility
python -m pip install -r requirements-figures.txt
python generate_individual_figures.py
```

Figures are build outputs and are not committed. `generate_figures.py` is a
library dependency of the individual-figures script (shared data loading and
the `AUDIT` path); you do not need to run it directly. Paths are adjusted to
resolve inside this package.
