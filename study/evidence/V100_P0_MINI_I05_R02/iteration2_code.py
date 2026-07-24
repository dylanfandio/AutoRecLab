import os
import pandas as pd
import matplotlib.pyplot as plt

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

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)


def build_dataset(dataset_name: str, seed: int):
    set_random_state(seed)
    if dataset_name == 'MovieLens100K':
        ds = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
        pipe = Pipe(
            MakeImplicit(3),
            CorePruning(5),
            UserHoldout(validation_size=1e-6, test_size=0.2),
        )
    elif dataset_name == 'AmazonVideoGames':
        ds = RecSysDataSet.use_dataloader('AmazonVideoGames')
        pipe = Pipe(
            MakeImplicit(3),
            CorePruning(5),
            UserHoldout(validation_size=1e-6, test_size=0.2),
        )
    elif dataset_name == 'LastFM':
        ds = RecSysDataSet.use_dataloader(DataSet.HetrecLastFM)
        pipe = Pipe(
            CorePruning(5),
            UserHoldout(validation_size=1e-6, test_size=0.2),
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return pipe.process(ds)


def flatten_results(results_dict, seed, dataset_name):
    if not results_dict:
        return pd.DataFrame()
    dataset_id, results = next(iter(results_dict.items()))
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)
    results = results.copy()
    results['seed'] = seed
    results['dataset_name'] = dataset_name
    return results


def main():
    seeds = [11, 22, 33, 44, 55]
    datasets = ['MovieLens100K', 'AmazonVideoGames', 'LastFM']

    plan = ExperimentPlan(plan_name='Seed_Sensitivity_MultiDataset')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))

    all_results = []

    for dataset_name in datasets:
        for seed in seeds:
            dataset = build_dataset(dataset_name, seed)
            print(f'Running {dataset_name} with seed={seed}...')
            run_omnirec(datasets=dataset, plan=plan, evaluator=evaluator)
            results_dict = evaluator.get_results()
            results = flatten_results(results_dict, seed, dataset_name)
            if not results.empty:
                all_results.append(results)

    if not all_results:
        print('No results were returned by the evaluator.')
        return

    results = pd.concat(all_results, ignore_index=True)
    results['metric_k'] = results['name'].astype(str) + '@' + results['k'].astype(str)

    csv_path = os.path.join(working_dir, 'seed_sensitivity_results.csv')
    results.to_csv(csv_path, index=False)

    summary = (
        results.groupby(['dataset_name', 'algorithm', 'name', 'k'])['value']
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values(['dataset_name', 'algorithm', 'name', 'k'])
    )
    summary_path = os.path.join(working_dir, 'seed_sensitivity_summary.csv')
    summary.to_csv(summary_path, index=False)

    pivot = summary.pivot_table(index=['dataset_name', 'algorithm'], columns=['name', 'k'], values='mean')
    plot_path = os.path.join(working_dir, 'seed_sensitivity_summary_plot.png')
    ax = pivot.plot(kind='bar', figsize=(14, 6), rot=30)
    ax.set_title('Seed sensitivity summary: mean nDCG and Precision across seeds')
    ax.set_xlabel('Dataset / Algorithm')
    ax.set_ylabel('Mean metric value')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print('\n## Documentation Verified')
    print('Datasets: MovieLens100K, AmazonVideoGames, LastFM')
    print('Preprocessing: 5-core filtering on all datasets; MakeImplicit(3) for MovieLens100K and AmazonVideoGames')
    print('Split: user-based 80/20 holdout via UserHoldout(validation_size=1e-6, test_size=0.2)')
    print('Seeds:', seeds)
    print('Algorithms: LensKit.ImplicitMFScorer (ALS), LensKit.ItemKNNScorer, LensKit.PopScorer')
    print('Metrics: NDCG@1, NDCG@5, NDCG@10, Precision@1, Precision@5, Precision@10')
    print('Results CSV:', csv_path)
    print('Summary CSV:', summary_path)
    print('Plot path:', plot_path)
    print('\nResults summary:')
    print(summary)
    print('\nShort statistical analysis of seed sensitivity:')
    for (dataset_name, algorithm), grp in summary.groupby(['dataset_name', 'algorithm']):
        mean_std = grp['std'].mean()
        print(f'- {dataset_name} / {algorithm}: average std across metrics = {mean_std:.6f}')


if __name__ == '__main__':
    main()
