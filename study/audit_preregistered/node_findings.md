# Preregistered phase — node-level coverage and semantic classification

Maximum coherent coverage per run (union of canonical keys from a single run's best evidence):

| Run | Condition | Max coherent | Native | Custom | Datasets | Algos | Seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V100_P2_MINI_I05_R01 | P2/Mini | 10/270 | 10 | 0 | 1 | 1 | 5 |
| V100_P2_MINI_I05_R02 | P2/Mini | 45/270 | 45 | 45 | 1 | 3 | 5 |
| V100_P2_MINI_I05_R03 | P2/Mini | 2/270 | 2 | 0 | 1 | 1 | 1 |
| V100_P2_MINI_I05_R04 | P2/Mini | 2/270 | 2 | 0 | 1 | 1 | 1 |
| V100_P2_MINI_I05_R05 | P2/Mini | 0/270 | 0 | 0 | 0 | 0 | 0 |
| V100_P3_MINI_I05_R01 | P3/Mini | 0/270 | 0 | 0 | 0 | 0 | 0 |
| V100_P3_MINI_I05_R02 | P3/Mini | 90/270 | 0 | 90 | 1 | 3 | 5 |
| V100_P3_MINI_I05_R03 | P3/Mini | 30/270 | 0 | 30 | 1 | 1 | 5 |
| V100_P3_MINI_I05_R04 | P3/Mini | 30/270 | 0 | 30 | 1 | 1 | 5 |
| V100_P3_MINI_I05_R05 | P3/Mini | 30/270 | 0 | 30 | 1 | 1 | 5 |

## Validation-contract subversion

- Validation reports found: **3**
- Reports that moved the denominator away from 270: **1** (V100_P3_MINI_I05_R03)
- Reports that self-passed on a shrunken denominator: **1**
- Corrupt / truncated reports: **1** (V100_P3_MINI_I05_R04)

## Semantic violation totals (across all node artifacts)

- node artifacts with metric rows: 16
- total duplicate rows: 3675
- total zero values: 90
- node artifacts with a zero block: 0
- node artifacts with algorithm-class provenance mismatch: 2
- node artifacts with nDCG@1 != Precision@1 breaks: 0
