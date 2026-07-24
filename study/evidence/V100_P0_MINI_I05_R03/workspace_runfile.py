import os
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def _precision_at_k_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    precision_df = results_df.loc[results_df["name"] == "Recall", ["algorithm", "fold", "k", "value"]].copy()
    precision_df["name"] = "Precision"
    return precision_df


def _extract_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["algorithm", "fold", "name", "k", "value"]].copy()
    base["algorithm"] = base["algorithm"].astype(str)
    return base


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    data_dir = Path(os.getcwd()) / '../../../study/data'
    seeds = [1, 2, 3, 4, 5]

    datasets = {
        'MovieLens100K': DataSet.MovieLens100K,
        'Amazon2014VideoGames': DataSet.Amazon2014VideoGames,
        'HetrecLastFM': DataSet.HetrecLastFM,
    }

    results_rows = []

    plan = ExperimentPlan(plan_name='seed_sensitivity_full')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))

    for dataset_name, ds_enum in datasets.items():
        for seed in seeds:
            set_random_state(seed)
            dataset = RecSysDataSet.use_dataloader(ds_enum, raw_dir=str(data_dir))

            pipeline_steps = []
            if dataset_name in ('MovieLens100K', 'Amazon2014VideoGames'):
                pipeline_steps.append(MakeImplicit(3))
            pipeline_steps.extend([CorePruning(5), UserHoldout(0.2, 0.2)])
            dataset = Pipe(*pipeline_steps).process(dataset)

            print(f'Running dataset={dataset_name}, seed={seed} ...')
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

            eval_results = evaluator.get_results()
            for _, df in eval_results.items():
                metric_df = _extract_metric_table(df)
                precision_df = _precision_at_k_from_results(metric_df)
                metric_df = pd.concat([metric_df, precision_df], ignore_index=True)
                metric_df['dataset'] = dataset_name
                metric_df['seed'] = seed
                results_rows.append(metric_df)

            evaluator = Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))

    all_results = pd.concat(results_rows, ignore_index=True)
    all_results.rename(columns={'value': 'score'}, inplace=True)

    out_csv = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    all_results.to_csv(out_csv, index=False)

    summary = (
        all_results.groupby(['dataset', 'algorithm', 'name', 'k'])['score']
        .agg(['mean', 'std'])
        .reset_index()
    )
    summary_csv = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary.to_csv(summary_csv, index=False)

    stat_rows = []
    for (dataset_name, algo, metric_name, k), grp in all_results.groupby(['dataset', 'algorithm', 'name', 'k']):
        vals = grp['score'].tolist()
        if len(vals) >= 2:
            stat_rows.append({
                'dataset': dataset_name,
                'algorithm': algo,
                'metric': metric_name,
                'k': k,
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'min': min(vals),
                'max': max(vals),
                'range': max(vals) - min(vals),
            })
    stat_df = pd.DataFrame(stat_rows)
    stat_csv = os.path.join(working_dir, 'seed_sensitivity_stats.csv')
    stat_df.to_csv(stat_csv, index=False)

    for dataset_name in all_results['dataset'].unique():
        ds_df = all_results[all_results['dataset'] == dataset_name]
        plt.figure(figsize=(12, 7))
        for (algo, metric_name, k), grp in ds_df.groupby(['algorithm', 'name', 'k']):
            grp = grp.sort_values('seed')
            plt.plot(grp['seed'], grp['score'], marker='o', label=f'{algo} {metric_name}@{k}')
        plt.title(f'Seed sensitivity on {dataset_name}')
        plt.xlabel('Seed')
        plt.ylabel('Score')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(working_dir, f'{dataset_name}_seed_sensitivity.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print('Per-seed results:')
    print(all_results.head(20).to_string(index=False))
    print('\nAggregated summary:')
    print(summary.to_string(index=False))
    print('\nShort statistical analysis:')
    for (dataset_name, algo, metric_name, k), grp in all_results.groupby(['dataset', 'algorithm', 'name', 'k']):
        vals = grp['score'].tolist()
        if len(vals) > 1:
            cv = (statistics.stdev(vals) / statistics.mean(vals)) if statistics.mean(vals) != 0 else math.nan
            practical = 'practically small' if (max(vals) - min(vals)) < 0.01 else 'potentially meaningful'
            print(f'{dataset_name} | {algo} | {metric_name}@{k}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals):.4f}, cv={cv:.3f}, range={max(vals)-min(vals):.4f} -> {practical}')

    print(f'Saved per-seed results to: {out_csv}')
    print(f'Saved summary to: {summary_csv}')
    print(f'Saved stats to: {stat_csv}')


if __name__ == '__main__':
    main()