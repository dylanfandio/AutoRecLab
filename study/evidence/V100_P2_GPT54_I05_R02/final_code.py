import json
import os
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import SplitData
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def normalize_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename_map = {}
    for c in out.columns:
        lc = c.lower()
        if lc in {"dataset_name", "dataset"}:
            rename_map[c] = "dataset"
        elif lc in {"algorithm_name", "algorithm"}:
            rename_map[c] = "algorithm"
        elif lc in {"metric_name", "metric", "name"}:
            rename_map[c] = "metric"
        elif lc in {"metric_at", "k", "cutoff"}:
            rename_map[c] = "cutoff"
        elif lc in {"mean", "value", "score", "result"}:
            rename_map[c] = "value"
    return out.rename(columns=rename_map)


def find_predictions_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("predictions.json"))


def count_predictions(checkpoint_dir: Path) -> int:
    total = 0
    for p in find_predictions_jsons(checkpoint_dir):
        try:
            df = pd.read_json(p)
            total += len(df)
            continue
        except Exception:
            pass
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        total += len(v)
                        break
            elif isinstance(data, list):
                total += len(data)
        except Exception:
            pass
    return total


def make_user_holdout_train_test_only(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, user_df in df.groupby("user", sort=False):
        shuffled = user_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        n = len(shuffled)
        n_test = max(1, int(np.floor(n * test_size)))
        if n_test >= n:
            n_test = n - 1
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_split_dataset(base_dataset: RecSysDataSet[Any], train_df: pd.DataFrame, test_df: pd.DataFrame) -> RecSysDataSet[Any]:
    empty_val = train_df.iloc[0:0].copy()
    split_data = SplitData(train_df, empty_val, test_df)
    return RecSysDataSet(split_data, meta=base_dataset.meta)


def main() -> None:
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    work = Path(working_dir)

    results_csv = work / "prototype_results_incremental.csv"
    summary_csv = work / "prototype_summary.csv"
    plot_path = work / "prototype_metrics.png"
    metadata_path = work / "prototype_metadata.json"

    package_versions = {
        "omnirec": pkg_version("omnirec"),
        "numpy": pkg_version("numpy"),
        "pandas": pkg_version("pandas"),
        "matplotlib": pkg_version("matplotlib"),
        "scikit-learn": pkg_version("scikit-learn"),
    }

    dataset_raw_path = Path("../../../study/data/u.data").resolve()
    if not dataset_raw_path.exists():
        raise FileNotFoundError(f"Expected local dataset file not found: {dataset_raw_path}")

    raw_schema_df = pd.read_csv(
        dataset_raw_path,
        sep="\t",
        header=None,
        names=["user", "item", "rating", "timestamp"],
    )

    dataset_name = "MovieLens100K"
    algorithm_class = LensKit.PopScorer
    algorithm_id = "LensKit.PopScorer"
    algorithm_label = "MostPopular"
    algorithm_params = {}
    seed = 7
    cutoff = 10

    existing_rows: list[dict[str, Any]] = []
    if results_csv.exists():
        try:
            existing_rows = pd.read_csv(results_csv).to_dict(orient="records")
        except Exception:
            existing_rows = []

    existing_rows = [
        r for r in existing_rows
        if not (
            str(r.get("dataset")) == dataset_name
            and int(r.get("seed", -1)) == seed
            and int(r.get("cutoff", -1)) == cutoff
        )
    ]

    seed_dir = work / f"seed_{seed}"
    checkpoints_dir = seed_dir / "checkpoints"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_random_state(seed)

    raw_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K, raw_dir=str(dataset_raw_path.parent))
    raw_interactions = int(raw_dataset.num_interactions())
    raw_min_rating = float(raw_dataset.min_rating())
    raw_max_rating = float(raw_dataset.max_rating())

    dataset = MakeImplicit(3).process(raw_dataset)
    dataset = CorePruning(5).process(dataset)
    processed_interactions = int(dataset.num_interactions())

    base_df = dataset._data.df.copy()
    train_df, test_df = make_user_holdout_train_test_only(base_df, test_size=0.2, seed=seed)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Train/test split is empty after user holdout.")

    split_dataset = build_split_dataset(dataset, train_df, test_df)

    plan = ExperimentPlan(plan_name=f"prototype_ml100k_mostpopular_seed_{seed}")
    plan.add_algorithm(algorithm_class, algorithm_params)
    evaluator = Evaluator(NDCG(cutoff), Precision(cutoff))

    old_cwd = os.getcwd()
    os.chdir(seed_dir)
    try:
        run_omnirec(datasets=split_dataset, plan=plan, evaluator=evaluator)
    finally:
        os.chdir(old_cwd)

    prediction_rows = count_predictions(checkpoints_dir)
    if prediction_rows <= 0:
        raise ValueError("No actual recommendations/predictions found.")

    results_map = evaluator.get_results()
    if not results_map:
        raise RuntimeError("Evaluator returned no results.")

    metric_rows = []
    for ds_id, df in results_map.items():
        tmp = normalize_results_frame(df)
        tmp["dataset_id"] = ds_id
        metric_rows.append(tmp)
    metric_df = pd.concat(metric_rows, ignore_index=True)

    metric_df["metric"] = metric_df["metric"].astype(str)
    metric_df["cutoff"] = pd.to_numeric(metric_df["cutoff"], errors="coerce")
    metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")

    ndcg_series = metric_df.loc[
        (metric_df["metric"].str.lower() == "ndcg") & (metric_df["cutoff"] == cutoff),
        "value",
    ]
    precision_series = metric_df.loc[
        (metric_df["metric"].str.lower() == "precision") & (metric_df["cutoff"] == cutoff),
        "value",
    ]

    if len(ndcg_series) == 0 or len(precision_series) == 0:
        raise RuntimeError(f"Could not extract NDCG@{cutoff} and Precision@{cutoff} from evaluator results.\n{metric_df}")

    ndcg = float(ndcg_series.iloc[0])
    precision = float(precision_series.iloc[0])
    if not np.isfinite(ndcg) or not np.isfinite(precision):
        raise ValueError(f"Non-finite metric values: ndcg={ndcg}, precision={precision}")

    row = {
        "dataset": dataset_name,
        "algorithm": algorithm_id,
        "algorithm_label": algorithm_label,
        "seed": seed,
        "cutoff": cutoff,
        "ndcg": ndcg,
        "precision": precision,
        "raw_interactions": raw_interactions,
        "processed_interactions": processed_interactions,
        "train_interactions": int(len(train_df)),
        "test_interactions": int(len(test_df)),
        "prediction_rows": int(prediction_rows),
    }

    existing_rows.append(row)
    results_df = pd.DataFrame(existing_rows)
    results_df.to_csv(results_csv, index=False)

    summary = (
        results_df.groupby(["dataset", "algorithm", "cutoff"], as_index=False)
        .agg(
            mean_ndcg=("ndcg", "mean"),
            std_ndcg=("ndcg", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            min_ndcg=("ndcg", "min"),
            max_ndcg=("ndcg", "max"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            min_precision=("precision", "min"),
            max_precision=("precision", "max"),
        )
    )
    summary["range_ndcg"] = summary["max_ndcg"] - summary["min_ndcg"]
    summary["range_precision"] = summary["max_precision"] - summary["min_precision"]
    summary.to_csv(summary_csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.bar([f"NDCG@{cutoff}", f"Precision@{cutoff}"], [ndcg, precision], color=["steelblue", "darkorange"])
    plt.ylabel("Metric value")
    plt.title("Prototype: MovieLens100K + OmniRec LensKit.PopScorer")
    top = max(ndcg, precision)
    plt.ylim(0, top * 1.2 if top > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    metadata = {
        "package_versions": package_versions,
        "dataset": dataset_name,
        "dataset_raw_path": str(dataset_raw_path),
        "raw_file_schema": {
            "separator": "tab",
            "columns": ["user", "item", "rating", "timestamp"],
            "preview_rows": raw_schema_df.head(3).to_dict(orient="records"),
        },
        "algorithm_identifier": algorithm_id,
        "algorithm_class_used": algorithm_id,
        "algorithm_label": algorithm_label,
        "algorithm_params": algorithm_params,
        "seed": seed,
        "cutoff": cutoff,
        "preprocessing": {
            "MakeImplicit_threshold": 3,
            "CorePruning_k": 5,
            "manual_user_holdout_test_size": 0.2,
            "manual_no_validation": True,
            "splitdata_container_empty_val_rows": 0,
            "note": "Prototype kept to one dataset and one algorithm. LensKit.PopScorer is the OmniRec-exposed MostPopular baseline.",
        },
        "raw_dataset_stats": {
            "raw_interactions": raw_interactions,
            "raw_min_rating": raw_min_rating,
            "raw_max_rating": raw_max_rating,
        },
        "counts": {
            "processed_interactions": processed_interactions,
            "train_interactions": int(len(train_df)),
            "test_interactions": int(len(test_df)),
            "prediction_rows": int(prediction_rows),
        },
        "metrics": {
            f"ndcg@{cutoff}": ndcg,
            f"precision@{cutoff}": precision,
        },
        "artifacts": {
            "results_csv": str(results_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "plot_path": str(plot_path.resolve()),
            "seed_dir": str(seed_dir.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nResults:")
    print(results_df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {results_csv}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
