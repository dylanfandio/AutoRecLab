import os
import json
import math
import platform
import traceback
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
from typing import Type, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def detect_versions():
    versions = {}
    for name in ["omnirec", "numpy", "pandas", "matplotlib"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "unavailable"
    try:
        import sklearn
        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        versions["scikit-learn"] = "unavailable"
    return versions


def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) and not pd.isna(v) else None
    except Exception:
        return None


def load_raw_with_schema(path):
    raw = pd.read_csv(path, sep=None, engine='python', header=None)
    return raw


def map_raw_dataset(dataset_key, raw_path):
    raw_df = load_raw_with_schema(raw_path)
    if dataset_key == 'MovieLens100K':
        raw_df = pd.read_csv(raw_path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    elif dataset_key == 'AmazonVideoGames':
        if raw_df.shape[1] >= 4:
            raw_df = raw_df.iloc[:, :4]
            raw_df.columns = ['user', 'item', 'rating', 'timestamp']
        else:
            raw_df.columns = ['user', 'item', 'rating'] + [f'col{i}' for i in range(4, raw_df.shape[1] + 1)]
            raw_df = raw_df[['user', 'item', 'rating']].copy()
            raw_df['timestamp'] = pd.NA
    elif dataset_key == 'LastFM':
        if raw_df.shape[1] >= 4:
            raw_df = raw_df.iloc[:, :4]
            raw_df.columns = ['user', 'item', 'tag', 'timestamp']
        else:
            raw_df.columns = ['user', 'item', 'tag'] + [f'col{i}' for i in range(4, raw_df.shape[1] + 1)]
            raw_df = raw_df[['user', 'item', 'tag']].copy()
            raw_df['timestamp'] = pd.NA
        raw_df['rating'] = 1
        raw_df = raw_df[['user', 'item', 'rating', 'timestamp']]
    else:
        raise ValueError(dataset_key)
    return raw_df


def build_results_rows_from_evaluator(dataset_name, algo_name, seed, evaluator):
    rows = []
    for ds_id, df in evaluator.get_results().items():
        if df is None or len(df) == 0:
            continue
        for metric_name in ['NDCG', 'Precision']:
            sub = df[df['name'].astype(str).str.lower() == metric_name.lower()]
            for _, r in sub.iterrows():
                cutoff = int(r['k']) if pd.notna(r['k']) else None
                value = safe_float(r['value'])
                if cutoff in (1, 5, 10) and value is not None:
                    rows.append({'dataset': dataset_name, 'algorithm': algo_name, 'seed': int(seed), 'cutoff': int(cutoff), 'nDCG': value if metric_name == 'NDCG' else None, 'Precision': value if metric_name == 'Precision' else None})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.groupby(['dataset', 'algorithm', 'seed', 'cutoff'], as_index=False).agg({'nDCG': 'first', 'Precision': 'first'})
    return out


def summarize_results(results_df):
    if len(results_df) == 0:
        return pd.DataFrame()
    rows = []
    for (dataset, algorithm, cutoff), grp in results_df.groupby(['dataset', 'algorithm', 'cutoff']):
        for metric in ['nDCG', 'Precision']:
            vals = grp[metric].dropna().astype(float).tolist()
            if not vals:
                continue
            rows.append({
                'dataset': dataset,
                'algorithm': algorithm,
                'cutoff': cutoff,
                'metric': metric,
                'mean': mean(vals),
                'std': stdev(vals) if len(vals) > 1 else 0.0,
                'min': min(vals),
                'max': max(vals),
                'range': max(vals) - min(vals),
            })
    return pd.DataFrame(rows)


def user_holdout_80_20(df, seed):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, grp in df.groupby('user', sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        n_test = max(1, int(round(0.2 * len(idx))))
        n_test = min(n_test, len(idx) - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
        train_parts.append(df.loc[train_idx])
        test_parts.append(df.loc[test_idx])
    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True) if train_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True) if test_parts else df.iloc[0:0].copy()
    return train_df, test_df


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    out_dir = Path(working_dir)
    results_path = out_dir / 'prototype_results.csv'
    summary_path = out_dir / 'prototype_summary.csv'
    plot_path = out_dir / 'prototype_plot.png'
    meta_path = out_dir / 'prototype_metadata.json'

    seeds = [7, 13, 29, 42, 87]
    cutoffs = [1, 5, 10]

    dataset_specs = [
        ('MovieLens100K', DataSet.MovieLens100K, Path('../../../study/data/u.data').resolve()),
        ('AmazonVideoGames', None, Path('../../../study/data/VideoGames.csv').resolve()),
        ('LastFM', None, Path('../../../study/data/UserTaggedArtiststimestamps.dat').resolve()),
    ]

    plan_specs: list[tuple[str, Any]] = [
        ('ALS_Implicit', LensKit.ImplicitMFScorer),
        ('ItemKNN', LensKit.ItemKNNScorer),
        ('MostPopular', LensKit.PopScorer),
    ]

    all_rows = []
    completed_runs = 0

    set_random_state(seeds[0])

    raw_report = []
    processed_counts = []

    for dataset_name, builtin_ds, raw_path in dataset_specs:
        raw_df = map_raw_dataset(dataset_name, raw_path)
        raw_report.append({'dataset': dataset_name, 'raw_schema': [{'column': c, 'dtype': str(raw_df[c].dtype)} for c in raw_df.columns], 'raw_counts': {'rows': int(len(raw_df)), 'users': int(raw_df['user'].nunique()), 'items': int(raw_df['item'].nunique())}})

        if dataset_name == 'MovieLens100K':
            base_dataset = RecSysDataSet.use_dataloader(builtin_ds if isinstance(builtin_ds, DataSet) else DataSet.MovieLens100K)
        else:
            base_dataset = RecSysDataSet(raw_df)

        if dataset_name in ('MovieLens100K', 'AmazonVideoGames'):
            pipeline = Pipe(MakeImplicit(3), CorePruning(5))
        else:
            pipeline = Pipe(CorePruning(5))

        processed = pipeline.process(base_dataset)
        counts = processed.num_interactions()
        if isinstance(counts, dict):
            train_rows = int(counts.get('train', 0))
            valid_rows = int(counts.get('valid', 0))
            test_rows = int(counts.get('test', 0))
        else:
            train_rows = int(len(processed._data.train)) if hasattr(processed._data, 'train') else 0
            valid_rows = int(len(processed._data.valid)) if hasattr(processed._data, 'valid') else 0
            test_rows = int(len(processed._data.test)) if hasattr(processed._data, 'test') else 0
        processed_counts.append({'dataset': dataset_name, 'after_pipeline_counts': counts, 'train_rows': train_rows, 'valid_rows': valid_rows, 'test_rows': test_rows})

        base_df = processed._data.df if hasattr(processed._data, 'df') else processed._data.train

        for algo_name, algo_cls in plan_specs:
            for seed in seeds:
                try:
                    set_random_state(seed)
                    train_df, test_df = user_holdout_80_20(base_df, seed)
                    split_ds = RecSysDataSet(pd.concat([train_df, test_df], ignore_index=True))
                    split_ds = split_ds.replace_data(type(processed._data)(train_df, pd.DataFrame(columns=train_df.columns), test_df)) if hasattr(processed._data, 'train') else split_ds
                    plan = ExperimentPlan(plan_name=f'{dataset_name}_{algo_name}_{seed}')
                    plan.add_algorithm(algo_cls)
                    evaluator = Evaluator(NDCG(cutoffs), Precision(cutoffs))
                    run_omnirec(datasets=split_ds, plan=plan, evaluator=evaluator)
                    new_rows = build_results_rows_from_evaluator(dataset_name, algo_name, seed, evaluator)
                    if len(new_rows) == 0:
                        raise RuntimeError('No finite metric rows were produced from actual recommendations.')
                    all_rows.extend(new_rows.to_dict(orient='records'))
                    completed_runs += len(new_rows)
                    pd.DataFrame(all_rows).to_csv(results_path, index=False)
                    summarize_results(pd.DataFrame(all_rows)).to_csv(summary_path, index=False)
                except Exception as e:
                    with open(out_dir / f'failure_{dataset_name}_{algo_name}_{seed}.txt', 'w', encoding='utf-8') as f:
                        f.write(str(e) + '\n')
                        f.write(traceback.format_exc())
                    continue

    if len(all_rows) == 0:
        raise RuntimeError('No finite metric rows were produced from actual recommendations.')

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(['dataset', 'algorithm', 'seed', 'cutoff']).reset_index(drop=True)
    summary_df = summarize_results(results_df)
    summary_df.to_csv(summary_path, index=False)
    results_df.to_csv(results_path, index=False)

    plt.figure(figsize=(8, 4))
    last_counts = processed_counts[-1]
    plt.bar(['train', 'test'], [last_counts['train_rows'], last_counts['test_rows']])
    plt.ylabel('Interactions')
    plt.title('Last processed dataset 80/20 Holdout')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    seed_effect = results_df.groupby(['dataset', 'algorithm', 'cutoff']).agg(nDCG_std=('nDCG', 'std'), Precision_std=('Precision', 'std'), nDCG_range=('nDCG', lambda s: s.max() - s.min()), Precision_range=('Precision', lambda s: s.max() - s.min())).reset_index()
    meta = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'random_state': get_random_state(),
        'package_versions': detect_versions(),
        'algorithm_classes': [str(cls) for _, cls in plan_specs],
        'dataset_reports': raw_report,
        'processed_counts': processed_counts,
        'completed_rows': int(len(results_df)),
        'train_test_counts_last_successful': processed_counts[-1] if processed_counts else {},
        'metrics': [f'NDCG@{k}' for k in cutoffs] + [f'Precision@{k}' for k in cutoffs],
        'seeds': seeds,
        'seed_effect_summary': seed_effect.to_dict(orient='records'),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('RESULTS TABLE')
    print(results_df.to_string(index=False))
    print('\nSUMMARY TABLE')
    print(summary_df.to_string(index=False))
    print('\nSEED EFFECT ANALYSIS')
    if len(results_df) > 0:
        print(seed_effect.to_string(index=False))
        print('Random split seed has a measurable but modest effect when std/range are small relative to the mean; larger std/range values indicate stronger seed sensitivity.')
    print('\nMETADATA')
    print(json.dumps(meta, indent=2))
    print(f'Saved: {results_path}')
    print(f'Saved: {summary_path}')
    print(f'Saved: {plot_path}')
    print(f'Saved: {meta_path}')


if __name__ == '__main__':
    main()
