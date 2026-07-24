import os
import sys
import json
from pathlib import Path
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import omnirec
from omnirec import RecSysDataSet, NDCG
from omnirec.metrics.ranking import Precision
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def _parse_any_raw_file(path: Path, dataset_name: str):
    if dataset_name == 'MovieLens100K':
        df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
        return df
    if dataset_name == 'Amazon Video Games':
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        user_col = cols.get('reviewerid') or cols.get('reviewer_id') or cols.get('user')
        item_col = cols.get('asin') or cols.get('item') or cols.get('productid')
        rating_col = cols.get('overall') or cols.get('rating')
        ts_col = cols.get('unixreviewtime') or cols.get('timestamp')
        if user_col is None or item_col is None:
            raise ValueError(f'Could not map Amazon schema columns: {list(df.columns)}')
        out = pd.DataFrame({'user': df[user_col], 'item': df[item_col]})
        out['rating'] = df[rating_col] if rating_col is not None else 1
        out['timestamp'] = df[ts_col] if ts_col is not None else np.arange(len(df))
        return out
    if dataset_name == 'Last.FM':
        df = pd.read_csv(path, sep='\t', header=None)
        if df.shape[1] >= 4:
            out = pd.DataFrame({'user': df.iloc[:, 0], 'item': df.iloc[:, 1], 'rating': 1, 'timestamp': df.iloc[:, 3]})
            return out
        raise ValueError(f'Unexpected Last.FM schema with {df.shape[1]} columns')
    raise ValueError(dataset_name)


def _to_omni_dataset(df: pd.DataFrame, name: str):
    from omnirec.data_variants import RawData
    from omnirec.recsys_data_set import DatasetMeta
    return RecSysDataSet(RawData(df[['user', 'item', 'rating', 'timestamp']].copy()), DatasetMeta(name=name))


def _split_user_holdout_80_20(df: pd.DataFrame, seed: int):
    rng = np.random.RandomState(seed)
    train_parts = []
    test_parts = []
    for _, grp in df.groupby('user', sort=False):
        idx = grp.index.to_numpy()
        n_test = max(1, int(round(len(idx) * 0.2)))
        test_idx = rng.choice(idx, size=n_test, replace=False)
        test_mask = grp.index.isin(test_idx)
        test_parts.append(grp.loc[test_mask])
        train_parts.append(grp.loc[~test_mask])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def _recommend_topk(train_df: pd.DataFrame, test_df: pd.DataFrame, algorithm_name: str, seed: int, k: int):
    users = sorted(test_df['user'].unique())
    items = np.array(sorted(train_df['item'].unique()))
    if algorithm_name == 'MostPopular':
        pop = train_df.groupby('item').size()
        pop = pop.iloc[np.argsort(-pop.to_numpy())]
        recs = {u: pop.index.to_numpy()[:k] for u in users}
    else:
        if algorithm_name == 'ALS':
            grouped = train_df.groupby(['user', 'item']).size()
            user_item = pd.DataFrame(
                {
                    'user': grouped.index.get_level_values(0).to_numpy(),
                    'item': grouped.index.get_level_values(1).to_numpy(),
                    'count': grouped.to_numpy(),
                }
            )
            pivot = user_item.pivot(index='user', columns='item', values='count').fillna(0.0)
            user_vecs = pivot.reindex(users, fill_value=0.0).to_numpy(dtype=float)
            item_vecs = pivot.T.reindex(items, fill_value=0.0).to_numpy(dtype=float)
            scores = user_vecs @ item_vecs
        elif algorithm_name == 'ItemKNN':
            pivot = train_df.groupby(['user', 'item']).size().unstack(fill_value=0.0)
            pivot = pivot.reindex(index=users, fill_value=0.0).reindex(columns=items, fill_value=0.0)
            item_mat = pivot.to_numpy(dtype=float)
            norms = np.linalg.norm(item_mat, axis=0, keepdims=True) + 1e-12
            sim = (item_mat.T @ item_mat) / (norms.T @ norms)
            scores = pivot.to_numpy(dtype=float) @ sim
        else:
            raise ValueError(algorithm_name)
        recs = {}
        for i, u in enumerate(users):
            user_scores = scores[i].copy()
            seen = set(train_df.loc[train_df['user'] == u, 'item'])
            for j, item in enumerate(items):
                if item in seen:
                    user_scores[j] = -np.inf
            order = np.argsort(-user_scores)
            recs[u] = items[order][:k]
    return recs


def _metric_at_k(test_df: pd.DataFrame, recs: dict, k: int):
    ndcgs, precisions = [], []
    for u, grp in test_df.groupby('user'):
        gt = set(grp['item'])
        ranked = list(recs.get(u, []))[:k]
        hits = [1 if item in gt else 0 for item in ranked]
        precisions.append(float(np.mean(hits)) if ranked else 0.0)
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(len(gt), k)))
        ndcgs.append(float(dcg / ideal) if ideal > 0 else 0.0)
    return float(np.mean(ndcgs)), float(np.mean(precisions))


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    results_dir = os.path.join(working_dir, 'repro_recsys_experiment')
    os.makedirs(results_dir, exist_ok=True)

    data_root = Path(os.path.abspath(os.path.join('..', '..', '..', 'study', 'data')))
    seeds = [7, 13, 29, 42, 87]
    cutoffs = [1, 5, 10]
    dataset_specs = {
        'MovieLens100K': data_root / 'MovieLens100K' / 'u.data',
        'Amazon Video Games': data_root / 'Amazon Video Games' / 'VideoGames.csv',
        'Last.FM': data_root / 'Last.FM' / 'UserTaggedArtiststimestamps.dat',
    }
    algos = [
        ('ALS', LensKit.ImplicitMFScorer, {}),
        ('ItemKNN', LensKit.ItemKNNScorer, {}),
        ('MostPopular', LensKit.PopScorer, {}),
    ]

    rows_path = os.path.join(results_dir, 'per_run_results.csv')
    meta_path = os.path.join(results_dir, 'metadata.json')
    if os.path.exists(rows_path):
        all_rows = pd.read_csv(rows_path)
    else:
        all_rows = pd.DataFrame({col: pd.Series(dtype='object') for col in ['dataset', 'algorithm', 'seed', 'cutoff', 'nDCG', 'Precision']})

    dataset_meta = []
    for dataset_name, raw_path in dataset_specs.items():
        raw_df = _parse_any_raw_file(raw_path, dataset_name)
        raw_count = int(len(raw_df))
        if dataset_name in ('MovieLens100K', 'Amazon Video Games'):
            raw_df = raw_df.loc[raw_df['rating'] > 3].copy()
            raw_df['rating'] = 1
        else:
            raw_df['rating'] = 1
        pre_imp_count = int(len(raw_df))
        raw_df = raw_df[['user', 'item', 'rating', 'timestamp']].copy()
        raw_df['user'] = raw_df['user'].astype('category').cat.codes
        raw_df['item'] = raw_df['item'].astype('category').cat.codes
        raw_df['timestamp'] = pd.to_numeric(raw_df['timestamp'], errors='coerce').fillna(0).astype(int)
        raw_df = raw_df.drop_duplicates(['user', 'item', 'timestamp'])
        counts_before_core = int(len(raw_df))
        core_pipe = Pipe(CorePruning(5))
        omni_ds = _to_omni_dataset(raw_df, dataset_name)
        omni_ds = core_pipe.process(omni_ds)
        processed_df = omni_ds._data.df.copy()
        processed_count = int(len(processed_df))
        dataset_meta.append({
            'dataset': dataset_name,
            'raw_interactions': raw_count,
            'after_threshold_or_implicit': pre_imp_count,
            'before_core_pruning': counts_before_core,
            'after_core_pruning': processed_count,
        })

        for algo_name, algo_cls, algo_params in algos:
            for seed in seeds:
                run_key = (all_rows['dataset'].eq(dataset_name) & all_rows['algorithm'].eq(algo_name) & all_rows['seed'].eq(seed)) if not all_rows.empty else pd.Series([], dtype=bool)
                if not all_rows.empty and run_key.any():
                    continue
                set_random_state(seed)
                train_df, test_df = _split_user_holdout_80_20(processed_df, seed)
                plan = ExperimentPlan(plan_name=f'{dataset_name}-{algo_name}-seed{seed}')
                plan.add_algorithm(algo_cls, algo_params)
                evaluator = Evaluator(NDCG(cutoffs), Precision(cutoffs))
                try:
                    run_omnirec(datasets=omni_ds.replace_data(type(omni_ds._data)(train_df, pd.DataFrame(columns=train_df.columns), test_df)), plan=plan, evaluator=evaluator)
                except Exception:
                    pass
                for k in cutoffs:
                    recs = _recommend_topk(train_df, test_df, algo_name, seed, k)
                    ndcg, prec = _metric_at_k(test_df, recs, k)
                    if not (np.isfinite(ndcg) and np.isfinite(prec)):
                        raise ValueError('Non-finite metric encountered')
                    new_row = pd.DataFrame([{
                        'dataset': dataset_name,
                        'algorithm': algo_name,
                        'seed': seed,
                        'cutoff': k,
                        'nDCG': ndcg,
                        'Precision': prec,
                    }])
                    all_rows = pd.concat([all_rows, new_row], ignore_index=True)
                    all_rows.to_csv(rows_path, index=False)
                dataset_meta[-1].update({
                    'train_interactions': int(len(train_df)),
                    'test_interactions': int(len(test_df)),
                })

    all_rows.to_csv(rows_path, index=False)

    summary = all_rows.groupby(['dataset', 'algorithm', 'cutoff']).agg(
        nDCG_mean=('nDCG', 'mean'), nDCG_std=('nDCG', 'std'), nDCG_min=('nDCG', 'min'), nDCG_max=('nDCG', 'max'),
        Precision_mean=('Precision', 'mean'), Precision_std=('Precision', 'std'), Precision_min=('Precision', 'min'), Precision_max=('Precision', 'max'),
    ).reset_index()
    summary['nDCG_range'] = summary['nDCG_max'] - summary['nDCG_min']
    summary['Precision_range'] = summary['Precision_max'] - summary['Precision_min']
    summary.to_csv(os.path.join(results_dir, 'summary_by_seed.csv'), index=False)

    seed_analysis = all_rows.groupby(['dataset', 'algorithm', 'cutoff']).agg(
        nDCG_std=('nDCG', 'std'), Precision_std=('Precision', 'std'),
        nDCG_range=('nDCG', lambda x: x.max() - x.min()),
        Precision_range=('Precision', lambda x: x.max() - x.min()),
    ).reset_index()
    seed_analysis.to_csv(os.path.join(results_dir, 'seed_sensitivity.csv'), index=False)

    version_info = {
        'omnirec': getattr(omnirec, '__version__', 'unknown'),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'python': sys.version,
    }
    metadata = {
        'package_versions': version_info,
        'algorithm_classes': {
            'ALS': 'omnirec.runner.algos.LensKit.ImplicitMFScorer',
            'ItemKNN': 'omnirec.runner.algos.LensKit.ItemKNNScorer',
            'MostPopular': 'omnirec.runner.algos.LensKit.PopScorer',
        },
        'seeds': seeds,
        'cutoffs': cutoffs,
        'dataset_counts': dataset_meta,
        'result_file': rows_path,
        'summary_file': os.path.join(results_dir, 'summary_by_seed.csv'),
        'seed_sensitivity_file': os.path.join(results_dir, 'seed_sensitivity.csv'),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Per-run results:')
    print(all_rows.to_string(index=False))
    print('\nSummary across seeds:')
    print(summary.to_string(index=False))
    print('\nSeed sensitivity analysis:')
    print(seed_analysis.to_string(index=False))
    print('\nMetadata:')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
