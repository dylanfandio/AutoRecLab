"""Tier B: human-authored reference pipeline on the pinned LensKit stack.

This is the positive control proper. It executes the full matrix requested by
``study/prompts/P0_original.txt``:

    3 datasets x 3 algorithms x 5 seeds x 3 cutoffs x 2 metrics = 270 entries

It uses LensKit directly (the version pinned in ``pyproject.toml``) rather than
routing through OmniRec's runner service, because that service issues
``recommend(model, test)`` with no list length and therefore materialises a full
user x item ranking. See ``pc_tier_a.py`` for that path and why it cannot
complete on Amazon Video Games.

Every cell is checkpointed to ``out/cells/``; re-running resumes rather than
recomputing. Usage:

    python pc_tier_b.py                      # full matrix
    python pc_tier_b.py --datasets MovieLens100K --seeds 42   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
from lenskit.als import ImplicitMFConfig, ImplicitMFScorer
from lenskit.basic.popularity import PopConfig, PopScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, from_interactions_df
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.metrics import NDCG, Precision, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.training import TrainingOptions

from pc_common import (
    ALGORITHMS,
    CUTOFFS,
    DATASETS,
    METRICS,
    OUT_DIR,
    SEEDS,
    entry_key,
    environment_fingerprint,
    measure,
    preprocess,
    split_counts,
    user_holdout_80_20,
    write_json,
)

MAX_N = max(CUTOFFS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tier_b")

# OmniRec logs every preprocessing step through a rich handler; the counts it
# prints are captured structurally in preprocessing_counts.csv, so quiet it.
logging.getLogger("omnirec").setLevel(logging.WARNING)


# "Train all models using standard hyperparameters" -> library defaults, with
# only the feedback mode set to implicit where the component exposes it.
def build_scorer(algorithm: str):
    if algorithm == "ALS":
        config = ImplicitMFConfig()
        return ImplicitMFScorer(config), ImplicitMFScorer, config
    if algorithm == "ItemKNN":
        config = ItemKNNConfig(feedback="implicit")
        return ItemKNNScorer(config), ItemKNNScorer, config
    if algorithm == "Pop":
        config = PopConfig()
        return PopScorer(config), PopScorer, config
    raise ValueError(f"unknown algorithm {algorithm}")


def evaluate(recs, test_df: pd.DataFrame) -> dict[str, float]:
    truth = ItemListCollection.from_df(
        test_df.rename(columns={"user": "user_id", "item": "item_id"}),
        ["user_id"],
    )
    analysis = RunAnalysis()
    for k in CUTOFFS:
        analysis.add_metric(NDCG(k), label=f"NDCG@{k}")
        analysis.add_metric(Precision(k), label=f"Precision@{k}")
    result = analysis.compute(recs, truth)

    summary = result.list_summary()
    values: dict[str, float] = {}
    for metric in METRICS:
        for k in CUTOFFS:
            label = f"{metric}@{k}"
            if label not in summary.index:
                raise KeyError(
                    f"metric {label} missing from analysis output; "
                    f"available: {list(summary.index)}"
                )
            values[label] = float(summary.loc[label, "mean"])
    return values


def run_cell(dataset: str, algorithm: str, seed: int, split, counts: dict) -> dict:
    train, test = split.train, split.test

    # Only users with training history can be scored; mirror the runner's
    # intersect so Tier A and Tier B evaluate the same user population.
    users = sorted(set(test["user"].unique()) & set(train["user"].unique()))
    test_eval = test[test["user"].isin(users)]

    scorer, cls, config = build_scorer(algorithm)
    pipeline = topn_pipeline(scorer, n=MAX_N)

    phases: dict[str, dict] = {}

    with measure() as total:
        with measure() as fit_res:
            data = from_interactions_df(train[["user", "item"]])
            pipeline.train(data, TrainingOptions(rng=seed))
        phases["fit"] = {
            "wall_seconds": fit_res.wall_seconds,
            "peak_rss_mib": fit_res.peak_rss_mib,
        }

        with measure() as pred_res:
            recs = recommend(pipeline, users, n=MAX_N)
        phases["predict"] = {
            "wall_seconds": pred_res.wall_seconds,
            "peak_rss_mib": pred_res.peak_rss_mib,
        }

        with measure() as eval_res:
            values = evaluate(recs, test_eval)
        phases["evaluate"] = {
            "wall_seconds": eval_res.wall_seconds,
            "peak_rss_mib": eval_res.peak_rss_mib,
        }

    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "algorithm_class": f"{cls.__module__}.{cls.__qualname__}",
        "algorithm_config": json.loads(config.model_dump_json()),
        "seed": seed,
        "users_scored": len(users),
        "test_interactions_evaluated": int(len(test_eval)),
        "recommendation_list_length": MAX_N,
        "wall_seconds": total.wall_seconds,
        "peak_rss_mib": total.peak_rss_mib,
        "delta_rss_mib": total.delta_rss_mib,
        "phases": phases,
        "metrics": values,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--algorithms", nargs="*", default=list(ALGORITHMS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = parser.parse_args()

    cells_dir = OUT_DIR / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    log.info("recording environment fingerprint")
    write_json(OUT_DIR / "environment.json", environment_fingerprint())

    all_counts: list[dict] = []
    all_splits: list[dict] = []
    metric_rows: list[dict] = []
    resource_rows: list[dict] = []
    failures: list[dict] = []

    for dataset in args.datasets:
        log.info("preprocessing %s", dataset)
        with measure() as prep:
            df, counts = preprocess(dataset)
        counts["preprocess_wall_seconds"] = round(prep.wall_seconds, 3)
        counts["preprocess_peak_rss_mib"] = round(prep.peak_rss_mib, 1)
        all_counts.append(counts)
        log.info(
            "%s: raw=%d dedup=%d threshold=%d core=%d (%d users, %d items)",
            dataset,
            counts["interactions_raw"],
            counts["interactions_after_dedup"],
            counts["interactions_after_threshold"],
            counts["interactions_after_core"],
            counts["users_after_core"],
            counts["items_after_core"],
        )

        for seed in args.seeds:
            split = user_holdout_80_20(df, seed)
            sc = split_counts(dataset, seed, split)
            all_splits.append(sc)
            log.info(
                "  seed %d: train=%d test=%d (%.4f held out)",
                seed,
                sc["train_interactions"],
                sc["test_interactions"],
                sc["test_fraction"],
            )

            for algorithm in args.algorithms:
                cell_path = cells_dir / f"{dataset}__{algorithm}__seed{seed}.json"
                if cell_path.exists() and not args.force:
                    cell = json.loads(cell_path.read_text(encoding="utf-8"))
                    log.info("    %s: cached", algorithm)
                else:
                    log.info("    %s: running", algorithm)
                    try:
                        cell = run_cell(dataset, algorithm, seed, split, counts)
                        log.info(
                            "    %s: done in %.1fs, peak %.0f MiB, NDCG@10=%.4f",
                            algorithm,
                            cell["wall_seconds"],
                            cell["peak_rss_mib"],
                            cell["metrics"]["NDCG@10"],
                        )
                    except Exception as exc:
                        cell = {
                            "dataset": dataset,
                            "algorithm": algorithm,
                            "seed": seed,
                            "status": "failed",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        log.error("    %s: FAILED %r", algorithm, exc)
                    write_json(cell_path, cell)

                if cell.get("status") != "ok":
                    failures.append(cell)
                    continue

                resource_rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "seed": seed,
                        "algorithm_class": cell["algorithm_class"],
                        "wall_seconds": round(cell["wall_seconds"], 3),
                        "peak_rss_mib": round(cell["peak_rss_mib"], 1),
                        "delta_rss_mib": round(cell["delta_rss_mib"], 1),
                        "fit_seconds": round(cell["phases"]["fit"]["wall_seconds"], 3),
                        "predict_seconds": round(
                            cell["phases"]["predict"]["wall_seconds"], 3
                        ),
                        "evaluate_seconds": round(
                            cell["phases"]["evaluate"]["wall_seconds"], 3
                        ),
                        "users_scored": cell["users_scored"],
                    }
                )
                for metric in METRICS:
                    for k in CUTOFFS:
                        metric_rows.append(
                            {
                                "key": entry_key(dataset, algorithm, seed, metric, k),
                                "dataset": dataset,
                                "algorithm": algorithm,
                                "algorithm_class": cell["algorithm_class"],
                                "seed": seed,
                                "metric": metric,
                                "k": k,
                                "value": cell["metrics"][f"{metric}@{k}"],
                            }
                        )

    pd.DataFrame(all_counts).to_csv(
        OUT_DIR / "preprocessing_counts.csv", index=False
    )
    pd.DataFrame(all_splits).to_csv(OUT_DIR / "split_counts.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "metrics.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(OUT_DIR / "resources.csv", index=False)
    if failures:
        write_json(OUT_DIR / "tier_b_failures.json", failures)

    log.info(
        "tier B complete: %d metric entries, %d failed cells",
        len(metric_rows),
        len(failures),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
