"""Tier A: the native OmniRec path, as a ceiling probe.

Tier B answers "was the task achievable in this environment?". Tier A answers
the narrower question "was it achievable *the way AutoRecLab was asked to do
it*?" -- i.e. through ``omnirec.runner``, which spawns an isolated LensKit
subprocess and evaluates through ``Evaluator``.

It is a probe rather than a full run because of a hard structural limit:
``omnirec_runner/lenskit_runner.py`` calls ``recommend(self.model, self.test)``
with no list length. In LensKit 2025.x, ``n=None`` means *rank every candidate
item for every user*, and the resulting frame is then serialised to
``predictions.json``. For Amazon Video Games that is 33,621 users x 12,455 items
= 4.2e8 rows. This script computes that projection before running and refuses
the cell unless ``--force`` is passed.

Two further framework behaviours are relevant and are recorded in the output:

* ``Coordinator.dataset_hash`` is derived only from ``num_interactions()``, and
  a per-user 80/20 holdout produces identical partition sizes for every seed.
  All five seeds therefore hash to the same checkpoint directory. This script
  gives each (dataset, seed) its own ``checkpoint_dir`` to avoid silently
  resuming another seed's model.
* ``UserHoldout`` cannot express an 80/20 split with no validation partition,
  so the split from ``pc_common`` is injected directly as ``SplitData``.

Usage:  python pc_tier_a.py [--datasets ...] [--seeds ...] [--force]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import traceback
from contextlib import redirect_stderr
from pathlib import Path

import pandas as pd
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.recsys_data_set import RecSysDataSet
from omnirec.runner.algos import LensKit
from omnirec.runner.coordinator import Coordinator
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.util import set_random_state

from pc_common import (
    ALGORITHMS,
    CUTOFFS,
    DATASETS,
    OUT_DIR,
    SEEDS,
    measure,
    preprocess,
    user_holdout_80_20,
    write_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tier_a")
logging.getLogger("omnirec").setLevel(logging.WARNING)

TIER_A_DIR = OUT_DIR / "tier_a"

# The prompt's ALS / ItemKNN / Pop, mapped onto the runner's registered names.
ALGO_MAP = {
    "ALS": LensKit.ImplicitMFScorer,
    "ItemKNN": LensKit.ItemKNNScorer,
    "Pop": LensKit.PopScorer,
}

# Refuse to materialise a full ranking larger than this many rows without
# --force. 20 million rows is already several GiB once serialised.
ROW_BUDGET = 20_000_000


def project_cost(n_users: int, n_items: int) -> dict:
    rows = n_users * n_items
    return {
        "users": n_users,
        "items": n_items,
        "projected_recommendation_rows": rows,
        # user, item, score, rank -> 8+8+8+8 bytes in-memory, before the
        # JSON serialisation the coordinator performs afterwards.
        "projected_in_memory_bytes": rows * 32,
        "projected_in_memory_gib": round(rows * 32 / 1024**3, 2),
        "within_budget": rows <= ROW_BUDGET,
        "row_budget": ROW_BUDGET,
    }


def run_cell(dataset_name: str, seed: int, split, algorithms: list[str]) -> dict:
    set_random_state(seed)

    dataset = RecSysDataSet(split)
    dataset._meta.name = f"{dataset_name}_seed{seed}"

    plan = ExperimentPlan(f"{dataset_name}_seed{seed}")
    for algo in algorithms:
        plan.add_algorithm(ALGO_MAP[algo])

    evaluator = Evaluator(NDCG(list(CUTOFFS)), Precision(list(CUTOFFS)))

    # Distinct checkpoint dir per seed: dataset_hash() would otherwise collide.
    ckpt = TIER_A_DIR / "checkpoints" / dataset_name / f"seed{seed}"
    ckpt.mkdir(parents=True, exist_ok=True)

    # Coordinator.run() catches every exception internally, prints the
    # traceback to stderr, and returns normally with an empty evaluator. A
    # non-zero wall time and status "ok" therefore do NOT imply results were
    # produced; the only reliable signal is whether the evaluator has rows.
    # Capture stderr so a swallowed traceback is preserved rather than lost.
    stderr_capture = io.StringIO()
    with measure() as res:
        coordinator = Coordinator(checkpoint_dir=ckpt)
        with redirect_stderr(stderr_capture):
            coordinator.run(dataset, plan, evaluator)

    results = {}
    total_rows = 0
    for dataset_id, frame in evaluator.get_results().items():
        records = frame.to_dict(orient="records")
        results[dataset_id] = records
        total_rows += len(records)

    captured = stderr_capture.getvalue()
    swallowed = "Traceback (most recent call last)" in captured

    if total_rows == 0:
        # Ran to completion but produced no metrics: the native path swallowed
        # an error inside evaluation. This is a completion, not a success.
        return {
            "status": "completed_without_results",
            "wall_seconds": res.wall_seconds,
            "peak_rss_mib": res.peak_rss_mib,
            "checkpoint_dir": str(ckpt),
            "results": results,
            "swallowed_exception": swallowed,
            "captured_stderr_tail": captured[-2000:],
        }

    return {
        "status": "ok",
        "wall_seconds": res.wall_seconds,
        "peak_rss_mib": res.peak_rss_mib,
        "checkpoint_dir": str(ckpt),
        "result_rows": total_rows,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--algorithms", nargs="*", default=list(ALGORITHMS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[SEEDS[0]])
    parser.add_argument(
        "--force",
        action="store_true",
        help="attempt cells whose projected ranking exceeds the row budget",
    )
    args = parser.parse_args()

    TIER_A_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for dataset_name in args.datasets:
        df, counts = preprocess(dataset_name)
        projection = project_cost(counts["users_after_core"], counts["items_after_core"])
        log.info(
            "%s: %d users x %d items -> %s recommendation rows (%.2f GiB in memory)",
            dataset_name,
            projection["users"],
            projection["items"],
            f"{projection['projected_recommendation_rows']:,}",
            projection["projected_in_memory_gib"],
        )

        for seed in args.seeds:
            record = {
                "dataset": dataset_name,
                "seed": seed,
                "algorithms": args.algorithms,
                "projection": projection,
            }

            if not projection["within_budget"] and not args.force:
                record["status"] = "refused_projected_infeasible"
                record["detail"] = (
                    f"the OmniRec LensKit runner calls recommend() without a list "
                    f"length, which would rank all {projection['items']:,} items for "
                    f"each of {projection['users']:,} users "
                    f"({projection['projected_recommendation_rows']:,} rows, "
                    f"~{projection['projected_in_memory_gib']} GiB before JSON "
                    f"serialisation). Exceeds the {ROW_BUDGET:,} row budget."
                )
                log.warning("%s/seed=%d: %s", dataset_name, seed, record["status"])
                records.append(record)
                continue

            split = user_holdout_80_20(df, seed)
            log.info(
                "%s/seed=%d: running native OmniRec path (train=%d test=%d)",
                dataset_name,
                seed,
                len(split.train),
                len(split.test),
            )
            try:
                record.update(run_cell(dataset_name, seed, split, args.algorithms))
                log.info(
                    "%s/seed=%d: completed in %.1fs, peak %.0f MiB",
                    dataset_name,
                    seed,
                    record["wall_seconds"],
                    record["peak_rss_mib"],
                )
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = repr(exc)
                record["traceback"] = traceback.format_exc()
                log.error("%s/seed=%d: FAILED %r", dataset_name, seed, exc)
            records.append(record)

    write_json(TIER_A_DIR / "tier_a_results.json", records)

    rows = []
    for record in records:
        for dataset_id, entries in record.get("results", {}).items():
            for entry in entries:
                rows.append(
                    {
                        "dataset": record["dataset"],
                        "seed": record["seed"],
                        "omnirec_dataset_id": dataset_id,
                        **entry,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(TIER_A_DIR / "tier_a_metrics.csv", index=False)

    ok = sum(1 for r in records if r.get("status") == "ok")
    log.info("tier A: %d/%d cells completed on the native path", ok, len(records))
    for record in records:
        if record.get("status") != "ok":
            log.info(
                "  %s/seed=%s -> %s",
                record["dataset"],
                record["seed"],
                record.get("status"),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
