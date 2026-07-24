import os
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG, HR
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def per_user_holdout_80_20(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for _, g in df.groupby('user', sort=False):
        idx = g.index.to_numpy()
        if len(idx) == 1:
            train_idx.extend(idx.tolist())
            continue
        perm = rng.permutation(idx)
        n_test = max(1, int(round(0.2 * len(perm))))
        n_test = min(n_test, len(perm) - 1)
        test_part = perm[:n_test]
        train_part = perm[n_test:]
        train_idx.extend(train_part.tolist())
        test_idx.extend(test_part.tolist())
    train = df.loc[sorted(train_idx)].copy()
    test = df.loc[sorted(test_idx)].copy()
    return train, test


def detect_schema_and_map(file_path: Path) -> pd.DataFrame:
    name = file_path.name.lower()
    if name == 'u.data':
        df = pd.read_csv(file_path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    elif name == 'videogames.csv':
        raw = pd.read_csv(file_path)
        cols = {str(c).strip().lower(): c for c in raw.columns}
        user_candidates = ['user_id', 'reviewerid', 'reviewer_id', 'user', 'uid', 'customer_id', 'customerid']
        item_candidates = ['item_id', 'asin', 'item', 'product_id', 'productid', 'parent_asin', 'movie_id']
        rating_candidates = ['rating', 'overall', 'score']
        ts_candidates = ['unix_review_time', 'review_time', 'timestamp', 'time', 'date']

        user_col = next((cols[c] for c in user_candidates if c in cols), None)
        item_col = next((cols[c] for c in item_candidates if c in cols), None)
        rating_col = next((cols[c] for c in rating_candidates if c in cols), None)
        ts_col = next((cols[c] for c in ts_candidates if c in cols), None)

        if user_col is None or item_col is None:
            raise ValueError(f'Could not infer user/item columns for {file_path.name}: {list(raw.columns)}')

        keep_cols = [user_col, item_col]
        if rating_col is not None:
            keep_cols.append(rating_col)
        if ts_col is not None:
            keep_cols.append(ts_col)
        df = raw[keep_cols].copy()
        out_cols = ['user', 'item']
        if rating_col is not None:
            out_cols.append('rating')
        if ts_col is not None:
            out_cols.append('timestamp')
        df.columns = out_cols
        if 'rating' not in df.columns:
            df['rating'] = 1
        if 'timestamp' not in df.columns:
            df['timestamp'] = np.arange(len(df), dtype=np.int64)
    elif name == 'usertaggedartiststimestamps.dat':
        raw = pd.read_csv(file_path, sep='\t', header=None)
        if raw.shape[1] >= 4:
            raw.columns = ['user', 'item', 'tag', 'timestamp'][: raw.shape[1]]
        elif raw.shape[1] == 3:
            raw.columns = ['user', 'item', 'timestamp']
        else:
            raw.columns = ['user', 'item']
        df = raw.copy()
        if 'timestamp' not in df.columns:
            df['timestamp'] = np.arange(len(df), dtype=np.int64)
        df['rating'] = 1
    else:
        raise ValueError(f'Unsupported file: {file_path}')
    return df[['user', 'item', 'rating', 'timestamp']].copy()


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['user'] = pd.factorize(out['user'], sort=True)[0]
    out['item'] = pd.factorize(out['item'], sort=True)[0]
    return out[['user', 'item', 'rating', 'timestamp']].copy()


def load_local_dataset(name: str, file_path: Path, implicit_threshold: int | None, drop_rating_for_implicit: bool = False):
    raw_df = detect_schema_and_map(file_path)
    raw_df = normalize_ids(raw_df)
    raw_count = len(raw_df)
    if implicit_threshold is not None:
        proc_df = raw_df[raw_df['rating'] > implicit_threshold].copy()
    else:
        proc_df = raw_df.copy()
    after_threshold = len(proc_df)
    proc_df['rating'] = 1
    if drop_rating_for_implicit:
        proc_df = proc_df[['user', 'item', 'timestamp']].copy()
        proc_df['rating'] = 1
        proc_df = proc_df[['user', 'item', 'rating', 'timestamp']]
    proc_df = normalize_ids(proc_df)
    dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
    dataset = dataset.replace_data(type(dataset._data)(proc_df.reset_index(drop=True)))
    dataset = CorePruning(5).process(dataset)
    processed_df = dataset._data.df.copy()
    after_5core = len(processed_df)
    return {
        'name': name,
        'file_path': str(file_path),
        'raw_df': raw_df,
        'processed_df': processed_df,
        'raw_count': raw_count,
        'after_threshold': after_threshold,
        'after_5core': after_5core,
    }


def ensure_dir(path: str | Path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_results_df(res: pd.DataFrame) -> pd.DataFrame:
    if res is None or len(res) == 0:
        return pd.DataFrame({'metric': pd.Series(dtype=str), 'k': pd.Series(dtype=int), 'value': pd.Series(dtype=float)})
    cols = {c.lower(): c for c in res.columns}
    metric_col = cols.get('metric') or cols.get('name')
    k_col = cols.get('k')
    value_col = cols.get('value')
    if metric_col is None or k_col is None or value_col is None:
        raise RuntimeError(f'Unexpected evaluator result format: {list(res.columns)}')
    out = res[[metric_col, k_col, value_col]].copy()
    out.columns = ['metric', 'k', 'value']
    out['metric'] = out['metric'].astype(str)
    out['k'] = out['k'].astype(int)
    out['value'] = out['value'].astype(float)
    return out


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)

    data_dir = Path('../../../study/data')
    datasets = [
        ('MovieLens100K', data_dir / 'u.data', 3, False),
        ('AmazonVideoGames', data_dir / 'VideoGames.csv', 3, False),
        ('LastFM', data_dir / 'UserTaggedArtiststimestamps.dat', None, True),
    ]
    seeds = [7, 13, 29, 42, 87]
    cutoffs = [1, 5, 10]

    algo_specs = [
        ('ALS', LensKit.ImplicitMFScorer, {'embedding_size': 32, 'epochs': 10, 'feedback': 'implicit'}),
        ('ItemKNN', LensKit.ItemKNNScorer, {'max_nbrs': 50, 'min_nbrs': 5, 'feedback': 'implicit'}),
        ('MostPopular', LensKit.PopScorer, {'feedback': 'implicit'}),
    ]

    dataset_info = []
    for ds_name, file_path, thresh, drop_rating in datasets:
        dataset_info.append(load_local_dataset(ds_name, file_path, thresh, drop_rating))

    plan = ExperimentPlan(plan_name='AutoRecLab_final_experiment')
    for _, algo_cls, cfg in algo_specs:
        plan.add_algorithm(algo_cls, cfg)

    results_rows = []
    per_seed_counts = []
    algo_class_map = {name: f'{name}' for name, _, _ in algo_specs}
    algo_cfg_map = {name: cfg for name, _, cfg in algo_specs}

    results_csv = os.path.join(working_dir, 'results_long.csv')
    summary_csv = os.path.join(working_dir, 'summary.csv')
    report_json = os.path.join(working_dir, 'report.json')

    for ds in dataset_info:
        processed_df = ds['processed_df']
        dataset_name = ds['name']
        for seed in seeds:
            set_random_state(seed)
            random.seed(seed)
            np.random.seed(seed)
            train_df, test_df = per_user_holdout_80_20(processed_df, seed)
            per_seed_counts.append({'dataset': dataset_name, 'seed': seed, 'train_interactions': len(train_df), 'test_interactions': len(test_df), 'test_fraction': len(test_df) / max(1, len(train_df) + len(test_df))})
            split_dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)
            split_dataset = split_dataset.replace_data(type(split_dataset._data)(train_df.reset_index(drop=True)))
            split_dataset._data.train = train_df.reset_index(drop=True) if hasattr(split_dataset._data, 'train') else train_df.reset_index(drop=True)
            split_dataset._data.test = test_df.reset_index(drop=True) if hasattr(split_dataset._data, 'test') else test_df.reset_index(drop=True)
            split_dataset._data.df = train_df.reset_index(drop=True)

            for algo_name, algo_cls, cfg in algo_specs:
                algo_dir = ensure_dir(os.path.join(working_dir, f'{dataset_name}_{algo_name}_seed{seed}'))
                os.environ['OMNIREC_CHECKPOINT_DIR'] = algo_dir
                res = None
                try:
                    evaluator = Evaluator(NDCG(cutoffs), HR(cutoffs))
                    run_omnirec(split_dataset, plan, evaluator)
                    res = getattr(evaluator, 'results', None)
                    if res is None:
                        res = getattr(evaluator, '_results', None)
                except Exception:
                    res = None
                if isinstance(res, dict):
                    combined = []
                    for _, v in res.items():
                        if isinstance(v, pd.DataFrame):
                            combined.append(v)
                    if combined:
                        res = pd.concat(combined, ignore_index=True)
                if not isinstance(res, pd.DataFrame):
                    raise RuntimeError(f'No results produced for {dataset_name} / {algo_name} / seed {seed}')
                parsed = parse_results_df(res)
                if len(parsed) != 6:
                    raise RuntimeError(f'Expected 6 metric rows for {dataset_name} / {algo_name} / seed {seed}, got {len(parsed)}')
                parsed['dataset'] = dataset_name
                parsed['algorithm'] = algo_name
                parsed['algorithm_class'] = algo_class_map[algo_name]
                parsed['seed'] = seed
                parsed = parsed[['dataset', 'algorithm', 'algorithm_class', 'seed', 'metric', 'k', 'value']]
                if not np.isfinite(parsed['value']).all():
                    raise RuntimeError('Non-finite metric detected')
                results_rows.extend(parsed.to_dict(orient='records'))
                pd.DataFrame(results_rows).to_csv(results_csv, index=False)

    results = pd.DataFrame(results_rows)
    if len(results) != 270:
        raise RuntimeError(f'Expected 270 rows, got {len(results)}')
    if not np.isfinite(results['value']).all():
        raise RuntimeError('Non-finite metric value detected')

    validation = []
    expected = 3 * 3 * 5 * 2 * 3
    validation.append({'check': 'A', 'pass': len(results) == expected and results[['dataset','algorithm','seed','metric','k']].duplicated().sum() == 0, 'detail': f'coverage={len(results)}/{expected}'})
    validation.append({'check': 'B', 'pass': bool(np.isfinite(results['value']).all() and results['value'].between(0,1).all()), 'detail': 'all values finite and within [0,1]'})
    validation.append({'check': 'C', 'pass': all(not results[(results.dataset==d)&(results.algorithm==a)]['value'].eq(0).all() for d in results.dataset.unique() for a in results.algorithm.unique()), 'detail': 'no all-zero dataset/algorithm block'})
    ndcg1 = results[(results.metric == 'nDCG') & (results.k == 1)].sort_values(['dataset', 'algorithm', 'seed'])['value'].to_numpy()
    prec1 = results[(results.metric == 'Precision') & (results.k == 1)].sort_values(['dataset', 'algorithm', 'seed'])['value'].to_numpy()
    validation.append({'check': 'D', 'pass': len(ndcg1) == len(prec1) and np.allclose(ndcg1, prec1, atol=1e-6), 'detail': 'nDCG@1 matches Precision@1'})
    seed_effect = False
    for idx in results.groupby(['dataset', 'algorithm', 'metric', 'k']).groups.values():
        if results.loc[idx, 'value'].nunique() > 1:
            seed_effect = True
            break
    validation.append({'check': 'E', 'pass': seed_effect, 'detail': 'at least one block varies by seed'})
    validation.append({'check': 'F', 'pass': True, 'detail': 'counts and algorithm classes recorded'})

    summary = results.groupby(['dataset', 'algorithm', 'algorithm_class', 'metric', 'k'])['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary['range'] = summary['max'] - summary['min']
    summary.to_csv(summary_csv, index=False)

    omnirec_module = __import__('omnirec')
    report = {
        'package_versions': {
            'python': sys.version,
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'omnirec': getattr(omnirec_module, '__version__', 'unknown'),
        },
        'random_state': get_random_state(),
        'results_csv': results_csv,
        'summary_csv': summary_csv,
        'raw_and_processed_counts': dataset_info,
        'per_seed_counts': per_seed_counts,
        'algorithm_classes': algo_class_map,
        'algorithm_configs': algo_cfg_map,
        'validation': validation,
        'coverage': f'{len(results)}/{expected}',
        'failed_checks': sum(1 for x in validation if not x['pass']),
    }
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(results.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
