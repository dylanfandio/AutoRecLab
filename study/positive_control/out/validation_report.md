# Positive control validation report

**Coverage: 270/270 requested metric entries**

**Semantic violations: 0**  
Checks passed: 45/46  
Warnings: 1

| Check | Status | Detail |
| --- | --- | --- |
| expected matrix size is 270 | pass | 3 datasets x 3 algorithms x 5 seeds x 3 cutoffs x 2 metrics = 270 |
| all 270 requested entries present | pass | 270/270 present |
| no unrequested entries | pass | 0 unexpected keys |
| no duplicate entries | pass | 0 duplicated keys |
| exactly five distinct seeds | pass | seeds present: [42, 43, 44, 45, 46]; declared: [42, 43, 44, 45, 46] |
| seeds produce different results | pass | all dataset/algorithm pairs vary across seeds |
| all metric values within [0, 1] | pass | 0 out-of-range values |
| no NaN metric values | pass | 0 NaN values |
| no all-zero result blocks | pass | none |
| each algorithm label maps to one class | pass | {'ALS': ['lenskit.als._implicit.ImplicitMFScorer'], 'ItemKNN': ['lenskit.knn.item.ItemKNNScorer'], 'Pop': ['lenskit.basic.popularity.PopScorer']} |
| three distinct algorithm classes | pass | 3 distinct classes: ['lenskit.als._implicit.ImplicitMFScorer', 'lenskit.basic.popularity.PopScorer', 'lenskit.knn.item.ItemKNNScorer'] |
| algorithms produce distinct results | pass | no two algorithms share an identical metric vector |
| nDCG@1 == Precision@1 (binary relevance identity) | pass | max absolute difference 0.000e+00 |
| Precision does not increase from k=1 to k=10 | WARN | 1 cells rise: ["('HetrecLastFM', 'Pop', 42)"] |
| MovieLens100K: interactions_raw matches independent recomputation | pass | pipeline=100000 independent=100000 |
| MovieLens100K: interactions_after_dedup matches independent recomputation | pass | pipeline=100000 independent=100000 |
| MovieLens100K: interactions_after_threshold matches independent recomputation | pass | pipeline=55375 independent=55375 |
| MovieLens100K: interactions_after_core matches independent recomputation | pass | pipeline=54413 independent=54413 |
| MovieLens100K: users_after_core matches independent recomputation | pass | pipeline=938 independent=938 |
| MovieLens100K: items_after_core matches independent recomputation | pass | pipeline=1008 independent=1008 |
| MovieLens100K: 5-core actually satisfied | pass | min user interactions=5, min item interactions=5 |
| MovieLens100K: implicit rule is 'rating > 3', not 'rating >= 3' | pass | MakeImplicit(4) keeps rating >= 4, i.e. rating > 3 |
| AmazonVideoGames: interactions_raw matches independent recomputation | pass | pipeline=2565349 independent=2565349 |
| AmazonVideoGames: interactions_after_dedup matches independent recomputation | pass | pipeline=2489395 independent=2489395 |
| AmazonVideoGames: interactions_after_threshold matches independent recomputation | pass | pipeline=1844699 independent=1844699 |
| AmazonVideoGames: interactions_after_core matches independent recomputation | pass | pipeline=291945 independent=291945 |
| AmazonVideoGames: users_after_core matches independent recomputation | pass | pipeline=33621 independent=33621 |
| AmazonVideoGames: items_after_core matches independent recomputation | pass | pipeline=12455 independent=12455 |
| AmazonVideoGames: 5-core actually satisfied | pass | min user interactions=5, min item interactions=5 |
| AmazonVideoGames: implicit rule is 'rating > 3', not 'rating >= 3' | pass | MakeImplicit(4) keeps rating >= 4, i.e. rating > 3 |
| HetrecLastFM: interactions_raw matches independent recomputation | pass | pipeline=186479 independent=186479 |
| HetrecLastFM: interactions_after_dedup matches independent recomputation | pass | pipeline=71064 independent=71064 |
| HetrecLastFM: interactions_after_threshold matches independent recomputation | pass | pipeline=71064 independent=71064 |
| HetrecLastFM: interactions_after_core matches independent recomputation | pass | pipeline=52551 independent=52551 |
| HetrecLastFM: users_after_core matches independent recomputation | pass | pipeline=1090 independent=1090 |
| HetrecLastFM: items_after_core matches independent recomputation | pass | pipeline=3646 independent=3646 |
| HetrecLastFM: 5-core actually satisfied | pass | min user interactions=5, min item interactions=5 |
| split counts recorded for every dataset and seed | pass | 15 rows, expected 15 |
| every user's holdout is ceil(0.2 x interactions), verified independently | pass | per-user 80/20 rule holds for all users in all splits |
| realised aggregate test fraction reported, not silently reshaped | pass | per-user integer rounding puts the aggregate above 20%: MovieLens100K=0.2068, AmazonVideoGames=0.2410, HetrecLastFM=0.2084 (Amazon is highest because 32% of its users have exactly 5 interactions after 5-core filtering) |
| no validation partition was created | pass | all splits are train/test only, as requested |
| train + test accounts for every post-filtering interaction | pass | no interactions lost or duplicated by splitting |
| runtime and peak memory recorded for every cell | pass | 45 rows, expected 45 |
| all recorded runtimes are positive | pass | min 21.829s, max 168.1s |
| environment fingerprint captured | pass | lenskit 2025.6.2, 160 packages pinned, git ff5f89870e8e |
| requested LensKit 0.14.4 documented as unavailable | pass | 0.14.4 (requested by P0_original.txt; not installable in this stack and not present in any environment) |
