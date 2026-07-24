import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.metrics.ranking import Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state
from omnirec.rsds import RawData


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[["user", "item", "rating", "timestamp"]]
    return df


def load_raw_datasets():
    data_dir = Path(os.path.abspath(os.path.join(os.getcwd(), "../../../study/data")))

    ml_path = data_dir / "u.data"
    ml = pd.read_csv(ml_path, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])

    amz_path = data_dir / "VideoGames.csv"
    amz = pd.read_csv(amz_path)
    cols = {c.lower(): c for c in amz.columns}
    user_col = cols.get("user_id") or cols.get("reviewerid") or cols.get("user")
    item_col = cols.get("asin") or cols.get("item_id") or cols.get("item")
    rating_col = cols.get("overall") or cols.get("rating")
    time_col = cols.get("unixreviewtime") or cols.get("timestamp")
    amz = pd.DataFrame(
        {
            "user": amz[user_col],
            "item": amz[item_col],
            "rating": amz[rating_col],
            "timestamp": amz[time_col] if time_col is not None else range(len(amz)),
        }
    )

    lfm_path = data_dir / "UserTaggedArtiststimestamps.dat"
    lfm = pd.read_csv(lfm_path, sep="\t|,|::", engine="python")
    cols = {c.lower(): c for c in lfm.columns}
    user_col = cols.get("userid") or cols.get("user")
    item_col = cols.get("artistid") or cols.get("item")
    time_col = cols.get("timestamp")
    lfm = pd.DataFrame(
        {
            "user": lfm[user_col],
            "item": lfm[item_col],
            "rating": 1,
            "timestamp": lfm[time_col] if time_col is not None else range(len(lfm)),
        }
    )

    return {
        "MovieLens100K": RecSysDataSet(RawData(_standardize_df(ml))),
        "AmazonVideoGames": RecSysDataSet(RawData(_standardize_df(amz))),
        "LastFM": RecSysDataSet(RawData(_standardize_df(lfm))),
    }


def preprocess_dataset(dataset_name, dataset):
    steps = [CorePruning(5)]
    if dataset_name in {"MovieLens100K", "AmazonVideoGames"}:
        steps.insert(0, MakeImplicit(3))
    pipeline = Pipe(*steps)
    dataset = pipeline.process(dataset)
    dataset = UserHoldout(validation_size=0.0, test_size=0.2).process(dataset)
    return dataset


def build_plan():
    plan = ExperimentPlan(plan_name="seed_sensitivity_implicit_feedback")
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)
    return plan


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    seeds = [7, 13, 21, 42, 87]
    all_rows = []
    datasets = load_raw_datasets()
    plan = build_plan()

    for seed in seeds:
        print(f'Running seed={seed}')
        set_random_state(seed)
        for dataset_name, raw_dataset in datasets.items():
            dataset = preprocess_dataset(dataset_name, raw_dataset)
            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            results = evaluator.get_results()
            for dataset_id, df in results.items():
                df = df.copy()
                df['seed'] = seed
                df['dataset_name'] = dataset_name
                df['dataset_id'] = dataset_id
                all_rows.append(df)
                print(df)

    results_df = pd.concat(all_rows, ignore_index=True)
    results_path = Path(working_dir) / 'prototype_seed_results.csv'
    results_df.to_csv(results_path, index=False)

    per_seed_table = (
        results_df.groupby(['seed', 'dataset_name', 'algorithm', 'name', 'k'], as_index=False)['value']
        .mean()
        .sort_values(['seed', 'dataset_name', 'algorithm', 'name', 'k'])
    )
    table_path = Path(working_dir) / 'prototype_seed_table.csv'
    per_seed_table.to_csv(table_path, index=False)
    print(per_seed_table)

    summary = (
        per_seed_table.groupby(['dataset_name', 'algorithm', 'name', 'k'])['value']
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )
    summary_path = Path(working_dir) / 'prototype_seed_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(summary)

    stat_rows = []
    for (dataset_name, algorithm, name, k), sub in per_seed_table.groupby(['dataset_name', 'algorithm', 'name', 'k']):
        stat_rows.append({
            'dataset_name': dataset_name,
            'algorithm': algorithm,
            'metric': f'{name}@{k}',
            'mean': sub['value'].mean(),
            'std': sub['value'].std(),
            'min': sub['value'].min(),
            'max': sub['value'].max(),
        })
    stat_df = pd.DataFrame(stat_rows)
    stat_path = Path(working_dir) / 'prototype_statistical_analysis.csv'
    stat_df.to_csv(stat_path, index=False)
    print(stat_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = per_seed_table.copy()
    plot_df['metric'] = plot_df['name'] + '@' + plot_df['k'].astype(str)
    for (dataset_name, algorithm, metric), sub in plot_df.groupby(['dataset_name', 'algorithm', 'metric']):
        label = f'{dataset_name} | {algorithm} | {metric}'
        ax.plot(sub['seed'], sub['value'], marker='o', linewidth=1.2, label=label)
    ax.set_xlabel('Random seed')
    ax.set_ylabel('Metric value')
    ax.set_title('Seed sensitivity across datasets, algorithms, and metrics')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig_path = Path(working_dir) / 'seed_variation_plot.png'
    fig.savefig(fig_path, dpi=150)

    print(f'Saved results to {results_path}')
    print(f'Saved per-seed table to {table_path}')
    print(f'Saved summary to {summary_path}')
    print(f'Saved statistical analysis to {stat_path}')
    print(f'Saved plot to {fig_path}')


if __name__ == '__main__':
    main()
