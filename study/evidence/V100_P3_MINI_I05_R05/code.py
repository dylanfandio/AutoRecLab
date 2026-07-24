import os
import json
import math
import random
import tempfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not_installed'


def fqcn(obj):
    cls = obj if isinstance(obj, type) else obj.__class__
    return cls.__module__ + '.' + cls.__qualname__


def load_local_dataset(name, raw_dir):
    raw_dir = Path(raw_dir)
    if name == 'MovieLens100K':
        path = raw_dir / 'u.data'
        df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
    elif name == 'Amazon Video Games':
        path = raw_dir / 'VideoGames.csv'
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        user_col = cols.get('user_id') or cols.get('reviewerid') or cols.get('user')
        item_col = cols.get('item_id') or cols.get('asin') or cols.get('item')
        rating_col = cols.get('rating')
        time_col = cols.get('timestamp') or cols.get('unixreviewtime') or cols.get('time')
        df = df.rename(columns={user_col: 'user', item_col: 'item', rating_col: 'rating', time_col: 'timestamp'})
        df = df[['user', 'item', 'rating', 'timestamp']]
    elif name == 'Last.FM':
        path = raw_dir / 'UserTaggedArtiststimestamps.dat'
        df = pd.read_csv(path, sep='\t')
        cols = {c.lower(): c for c in df.columns}
        user_col = cols.get('userid') or cols.get('user')
        item_col = cols.get('artistid') or cols.get('artist') or cols.get('item')
        tag_col = cols.get('tagid') or cols.get('tag')
        time_col = cols.get('timestamp')
        rename_map = {user_col: 'user', item_col: 'item'}
        if time_col:
            rename_map[time_col] = 'timestamp'
        df = df.rename(columns=rename_map)
        if tag_col:
            df = df.drop(columns=[tag_col])
        df['rating'] = 1
        df = df[['user', 'item', 'rating', 'timestamp']]
    else:
        raise ValueError(name)
    return df


def implicit_and_core(df, dataset_name):
    raw_count = len(df)
    if dataset_name in ['MovieLens100K', 'Amazon Video Games']:
        df = df[df['rating'] > 3].copy()
    else:
        df = df.copy()
    implicit_count = len(df)
    if dataset_name == 'Last.FM' and 'rating' in df.columns:
        df = df.drop(columns=['rating'])
        df['rating'] = 1
    changed = True
    while changed:
        changed = False
        user_counts = df.groupby('user').size()
        item_counts = df.groupby('item').size()
        good_users = user_counts[user_counts >= 5].index
        good_items = item_counts[item_counts >= 5].index
        new_df = df[df['user'].isin(good_users) & df['item'].isin(good_items)].copy()
        if len(new_df) != len(df):
            changed = True
            df = new_df
    core_count = len(df)
    return df, raw_count, implicit_count, core_count


def per_user_holdout(df, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for _, grp in df.groupby('user'):
        idx = grp.index.to_numpy()
        if len(idx) == 1:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_test = min(n_test, len(idx) - 1)
        test_sel = rng.choice(idx, size=n_test, replace=False)
        test_set = set(test_sel.tolist())
        for i in idx:
            (test_idx if i in test_set else train_idx).append(i)
    train = df.loc[sorted(train_idx)].copy()
    test = df.loc[sorted(test_idx)].copy()
    return train, test


def dataframe_to_dataset(df):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / 'dataset.csv'
        df.to_csv(tmp_path, index=False)
        return RecSysDataSet.load(tmp_path)


def build_plan(algorithm_name):
    plan = ExperimentPlan(f'exp_{algorithm_name}')
    if algorithm_name == 'ALS':
        algo = LensKit.ImplicitMFScorer
        params = {'embedding_size': 32, 'epochs': 10, 'reg': 0.1}
    elif algorithm_name == 'ItemKNN':
        algo = LensKit.ItemKNNScorer
        params = {'max_nbrs': 20, 'min_nbrs': 5}
    else:
        algo = LensKit.PopScorer
        params = {}
    plan.add_algorithm(algo, params)
    return plan, algo, params


def extract_results(result_tables):
    if not result_tables:
        return pd.DataFrame()
    first = next(iter(result_tables.values()))
    if isinstance(first, pd.DataFrame):
        return first.copy()
    return pd.DataFrame(first)


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    out_dir = Path(working_dir)
    print(json.dumps({
        'omnirec': pkg_version('omnirec'),
        'lenskit': pkg_version('lenskit'),
        'recbole': pkg_version('recbole'),
        'numpy': pkg_version('numpy'),
        'pandas': pkg_version('pandas'),
        'scipy': pkg_version('scipy'),
        'matplotlib': pkg_version('matplotlib'),
    }, indent=2))

    datasets_raw = {
        'MovieLens100K': '../../../study/data/',
        'Amazon Video Games': '../../../study/data/',
        'Last.FM': '../../../study/data/',
    }
    seeds = [7, 13, 29, 42, 87]
    algorithms = ['ALS', 'ItemKNN', 'MostPopular']
    rows = []
    meta = {'datasets': {}, 'runs': []}

    for dataset_name, raw_dir in datasets_raw.items():
        raw_df = load_local_dataset(dataset_name, raw_dir)
        proc_df, raw_count, implicit_count, core_count = implicit_and_core(raw_df, dataset_name)
        meta['datasets'][dataset_name] = {
            'raw_interactions': raw_count,
            'post_threshold_or_implicit_interactions': implicit_count,
            'post_5core_interactions': core_count,
            'schema_columns': list(raw_df.columns),
        }
        dataset_cache = proc_df
        for alg_name in algorithms:
            plan, algo_cls, params = build_plan(alg_name)
            for seed in seeds:
                set_random_state(seed)
                random.seed(seed)
                np.random.seed(seed)
                train_df, test_df = per_user_holdout(dataset_cache, seed)
                meta['runs'].append({
                    'dataset': dataset_name,
                    'algorithm': alg_name,
                    'seed': seed,
                    'algorithm_class': fqcn(algo_cls),
                    'resolved_hyperparameters': params,
                    'train_interactions': int(len(train_df)),
                    'test_interactions': int(len(test_df)),
                    'test_fraction': float(len(test_df) / max(1, len(train_df) + len(test_df))),
                })
                ds_train = dataframe_to_dataset(train_df)
                evaluator = Evaluator(NDCG([1, 5, 10]), Precision([1, 5, 10]))
                run_omnirec(ds_train, plan, evaluator)
                result_tables = evaluator.get_results()
                assert result_tables, 'No results returned'
                df_res = extract_results(result_tables)
                assert len(df_res) == 6, f'Expected 6 rows, got {len(df_res)}'
                assert np.isfinite(df_res['value']).all(), 'Non-finite metric values'
                for _, r in df_res.iterrows():
                    rows.append({
                        'dataset': dataset_name,
                        'algorithm': alg_name,
                        'algorithm_class': fqcn(algo_cls),
                        'seed': seed,
                        'metric': r['name'],
                        'k': int(r['k']),
                        'value': float(r['value']),
                    })
                pd.DataFrame(rows).to_csv(out_dir / 'incremental_results.csv', index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / 'results_long.csv', index=False)
    summary = results.groupby(['dataset', 'algorithm', 'algorithm_class', 'metric', 'k'])['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary['range'] = summary['max'] - summary['min']
    summary.to_csv(out_dir / 'summary.csv', index=False)

    validation = []
    def check(name, passed, detail):
        validation.append({'check': name, 'passed': bool(passed), 'detail': detail})
    check('A Completeness', len(results) == 270, f'{len(results)}/270 rows')
    check('B Finiteness and range', np.isfinite(results['value']).all() and results['value'].between(0, 1).all(), 'all values finite and in [0,1]')
    no_deg = not any((g['value'].abs().sum() == 0) for _, g in results.groupby(['dataset', 'algorithm']))
    check('C No degenerate blocks', no_deg, 'no all-zero dataset-algorithm block')
    id_ok = all(abs(g[g['metric'].eq('NDCG') & g['k'].eq(1)]['value'].iloc[0] - g[g['metric'].eq('Precision') & g['k'].eq(1)]['value'].iloc[0]) <= 1e-6 for _, g in results.groupby(['dataset', 'algorithm', 'seed']) if {'NDCG', 'Precision'}.issubset(set(g['metric'])))
    check('D Metric identity', id_ok, 'nDCG@1 equals Precision@1')
    seed_effect = any(len(set(g[g['metric'].eq('NDCG') & g['k'].eq(10)]['value'])) > 1 for _, g in results.groupby(['dataset', 'algorithm']))
    check('E Seed effect', seed_effect, 'at least one block varies across seeds')
    check('F Reproducibility counts recorded', len(meta['runs']) == 45 and len(meta['datasets']) == 3, 'counts recorded for all datasets and runs')
    val_df = pd.DataFrame(validation)
    val_df.to_json(out_dir / 'validation_report.json', orient='records', indent=2)

    print(json.dumps({
        'coverage': f"{len(results)}/270",
        'failed_checks': int((~val_df['passed']).sum()),
        'validation': validation,
        'dataset_counts': meta['datasets'],
        'runs_sample': meta['runs'][:3],
    }, indent=2))


if __name__ == '__main__':
    main()
