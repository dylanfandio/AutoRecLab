import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from omnirec import RecSysDataSet, NDCG
from omnirec.metrics.ranking import Precision
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


def _extract_results_df(evaluator, dataset_name):
    results = evaluator.get_results()
    if dataset_name in results:
        return results[dataset_name].copy()
    if isinstance(results, dict):
        matches = [key for key in results.keys() if str(key).startswith(dataset_name)]
        if len(matches) == 1:
            return results[matches[0]].copy()
        if len(matches) > 1:
            return results[matches[0]].copy()
    raise KeyError(f'Could not find results for dataset {dataset_name}. Available keys: {list(results.keys())}')


def _load_and_preprocess(dataset_enum, make_implicit):
    dataset = RecSysDataSet.use_dataloader(dataset_enum)
    steps = []
    if make_implicit:
        steps.append(MakeImplicit(3))
    steps.extend([
        CorePruning(5),
        UserHoldout(validation_size=0.0, test_size=0.2),
    ])
    return Pipe(*steps).process(dataset)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    datasets_cfg = [
        ('MovieLens100K', DataSet.MovieLens100K, True),
        ('Amazon2014VideoGames', DataSet.Amazon2014VideoGames, True),
        ('HetrecLastFM', DataSet.HetrecLastFM, False),
    ]
    seeds = [11, 22, 33, 44, 55]
    rows = []

    for seed in seeds:
        set_random_state(seed)
        for dataset_name, dataset_enum, make_implicit in datasets_cfg:
            dataset = _load_and_preprocess(dataset_enum, make_implicit)

            plan = ExperimentPlan(f'prototype_{dataset_name}_seed_{seed}')
            plan.add_algorithm(LensKit.ImplicitMFScorer, {})
            plan.add_algorithm(LensKit.ItemKNNScorer, {})
            plan.add_algorithm(LensKit.PopScorer, {})

            evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)

            df = _extract_results_df(evaluator, dataset.meta.name).copy()
            df['seed'] = seed
            df['dataset'] = dataset_name
            df.to_csv(os.path.join(working_dir, f'results_{dataset_name}_seed_{seed}.csv'), index=False)
            rows.append(df)
            print(f'Seed {seed} results for {dataset_name}:\n{df}')

    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(os.path.join(working_dir, 'all_results.csv'), index=False)

    summary = (
        all_results.groupby(['dataset', 'algorithm', 'name', 'k'], as_index=False)['value']
        .agg(mean='mean', std='std', min='min', max='max')
    )
    summary.to_csv(os.path.join(working_dir, 'summary.csv'), index=False)
    print('Summary:\n', summary)

    stat_rows = []
    for (dataset_name, algorithm, metric_name, k), g in all_results.groupby(['dataset', 'algorithm', 'name', 'k']):
        values = g['value'].dropna().values
        if len(values) >= 2:
            stat_rows.append({
                'dataset': dataset_name,
                'algorithm': algorithm,
                'name': metric_name,
                'k': k,
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)),
                'cv': float(values.std(ddof=1) / values.mean()) if values.mean() != 0 else None,
                'range': float(values.max() - values.min()),
                'n': int(len(values)),
                't_stat_vs_zero': float(stats.ttest_1samp(values, popmean=0.0, nan_policy='omit').statistic),
                'p_value_vs_zero': float(stats.ttest_1samp(values, popmean=0.0, nan_policy='omit').pvalue),
            })
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_stats.csv'), index=False)
    print('Seed sensitivity stats:\n', stat_df)

    pivot = all_results.pivot_table(index='seed', columns=['dataset', 'algorithm', 'name', 'k'], values='value', aggfunc='mean')
    ax = pivot.plot(kind='bar', figsize=(14, 6))
    ax.set_title('Split-seed sensitivity across datasets and algorithms')
    ax.set_ylabel('Metric value')
    ax.set_xlabel('Seed')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'seed_sensitivity_plot.png'), dpi=150)
    plt.close()

    print(json.dumps({'working_dir': working_dir, 'datasets': [d[0] for d in datasets_cfg], 'seeds': seeds}, indent=2))


if __name__ == '__main__':
    main()
