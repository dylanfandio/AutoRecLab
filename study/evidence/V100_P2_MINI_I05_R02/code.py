import os
import json
from pathlib import Path
from importlib import metadata
from statistics import stdev
from typing import cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet, NDCG, Recall
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def safe_version(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return 'unknown'


def get_public_df(dataset):
    data = dataset._data
    if hasattr(data, 'df'):
        return data.df.copy()
    if hasattr(data, 'train') and hasattr(data, 'test'):
        frames = []
        for split_name in ['train', 'valid', 'test']:
            part = getattr(data, split_name, None)
            if part is not None and len(part) > 0:
                frames.append(part.copy())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise TypeError(f'Unsupported dataset variant: {type(data)}')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def append_row_csv(path, row):
    df = pd.DataFrame([row])
    if Path(path).exists():
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_and_process_dataset(dataset_enum, implicit_threshold=None):
    raw_ds = RecSysDataSet.use_dataloader(dataset_enum)
    raw_df = get_public_df(raw_ds)

    steps = []
    if implicit_threshold is not None:
        steps.append(MakeImplicit(implicit_threshold))
    steps.extend([CorePruning(5), UserHoldout(validation_size=0.15, test_size=0.20)])
    pipeline = Pipe(*steps)
    split_ds = pipeline.process(raw_ds)

    data = split_ds._data
    train_df = data.train if hasattr(data, 'train') else data.df
    valid_df = getattr(data, 'valid', None)
    test_df = getattr(data, 'test', None)

    proc_counts = {
        'raw_interactions': int(len(raw_df)),
        'raw_users': int(raw_df['user'].nunique()),
        'raw_items': int(raw_df['item'].nunique()),
        'processed_interactions': int(len(get_public_df(split_ds))),
        'train_interactions': int(len(train_df)),
        'valid_interactions': int(len(valid_df)) if valid_df is not None else 0,
        'test_interactions': int(len(test_df)) if test_df is not None else 0,
    }
    return split_ds, raw_df, proc_counts


def extract_results_df(evaluator):
    results = evaluator.get_results()
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, pd.DataFrame):
                return v.copy()
    return pd.DataFrame(results)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    ensure_dir(working_dir)
    base = Path(working_dir)

    versions = {
        'omnirec': safe_version('omnirec'),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'scikit-learn': safe_version('scikit-learn'),
    }

    datasets_cfg = [
        ('MovieLens100K', DataSet.MovieLens100K, 3),
    ]

    processed = []
    raw_counts = {}
    for name, ds_enum, threshold in datasets_cfg:
        split_ds, raw_df, proc = load_and_process_dataset(ds_enum, threshold)
        processed.append((name, split_ds))
        raw_counts[name] = proc

    plan = ExperimentPlan('final_full_experiment')
    plan.add_algorithm(LensKit.ImplicitMFScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.PopScorer)

    evaluator = Evaluator(NDCG([1, 5, 10]), Recall([1, 5, 10]))

    results_csv = base / 'final_results.csv'
    summary_json = base / 'final_summary.json'
    plot_png = base / 'final_metrics_plot.png'

    seeds = [7, 13, 29, 42, 87]
    row_records = []

    for seed in seeds:
        set_random_state(seed)
        np.random.seed(seed)
        for dataset_name, split_ds in processed:
            run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
            res_df = extract_results_df(evaluator)
            if res_df.empty:
                continue
            for _, r in res_df.iterrows():
                if pd.isna(r.get('value')) or not np.isfinite(float(r['value'])):
                    continue
                row = {
                    'dataset': dataset_name,
                    'algorithm': str(r['algorithm']),
                    'seed': seed,
                    'cutoff': int(r['k']),
                    'metric': str(r['name']),
                    'value': float(r['value']),
                }
                row_records.append(row)
                append_row_csv(results_csv, row)

    rows = pd.DataFrame(row_records)
    if rows.empty:
        raise RuntimeError('No finite metric results were produced.')

    table_df = rows.pivot_table(index=['dataset', 'algorithm', 'seed', 'cutoff'], columns='metric', values='value').reset_index()
    table_df.columns.name = None
    if 'NDCG' in table_df.columns:
        table_df = table_df.rename(columns={'NDCG': 'nDCG', 'Recall': 'Precision'})

    agg = table_df.groupby(['dataset', 'algorithm', 'cutoff'])[['nDCG', 'Precision']].agg(['mean', 'std', 'min', 'max'])
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    for metric in ['nDCG', 'Precision']:
        agg[f'{metric}_range'] = agg[f'{metric}_max'] - agg[f'{metric}_min']

    analysis = {}
    grouped = table_df.groupby(['dataset', 'algorithm', 'cutoff'])
    for group_key, grp in grouped:
        dataset_name, algo, cutoff = cast(tuple[str, str, int], group_key)
        std_ndcg = float(grp['nDCG'].std(ddof=1)) if len(grp) > 1 else 0.0
        mean_ndcg = float(grp['nDCG'].mean())
        cv = std_ndcg / mean_ndcg if mean_ndcg != 0 else 0.0
        analysis[f'{dataset_name} | {algo} | @{cutoff}'] = {
            'n_seeds': int(len(grp)),
            'ndcg_mean': mean_ndcg,
            'ndcg_std': std_ndcg,
            'ndcg_cv': cv,
        }

    plot_df = agg[['dataset', 'algorithm', 'cutoff', 'nDCG_mean', 'Precision_mean']].copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(plot_df))
        ax.bar(x - 0.2, plot_df['nDCG_mean'], width=0.4, label='nDCG mean')
        ax.bar(x + 0.2, plot_df['Precision_mean'], width=0.4, label='Precision mean')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n{a}\n@{k}" for d, a, k in zip(plot_df['dataset'], plot_df['algorithm'], plot_df['cutoff'])], rotation=0)
        ax.set_ylabel('Score')
        ax.set_title('Mean accuracy across seeds')
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_png, dpi=150)

    summary = {
        'versions': versions,
        'algorithm_classes': {
            'ALS_for_implicit': 'LensKit.ImplicitMFScorer',
            'ItemKNN': 'LensKit.ItemKNNScorer',
            'MostPopular': 'LensKit.PopScorer',
        },
        'seeds': seeds,
        'raw_and_split_counts': raw_counts,
        'results_file': str(results_csv),
        'plot_file': str(plot_png),
        'random_state': get_random_state(),
        'seed_sensitivity_analysis': analysis,
    }
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(table_df)
    print(agg)


if __name__ == '__main__':
    main()
