import os
import json
import math
import platform
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, NDCG
from omnirec.data_loaders.datasets import DataSet
from omnirec.metrics.ranking import Precision
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def main():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    print(f'working_dir={working_dir}')

    seeds = [7, 13, 29, 42, 87]
    cutoffs = [1, 5, 10]
    datasets = {
        'MovieLens100K': {
            'raw_path': Path('../../../study/data/u.data'),
            'kind': 'movielens',
            'loader': DataSet.MovieLens100K,
        },
        'Amazon Video Games': {
            'raw_path': Path('../../../study/data/VideoGames.csv'),
            'kind': 'amazon',
            'loader': DataSet.Amazon2014VideoGames,
        },
        'Last.FM': {
            'raw_path': Path('../../../study/data/UserTaggedArtiststimestamps.dat'),
            'kind': 'lastfm',
            'loader': DataSet.HetrecLastFM,
        },
    }

    results_path = os.path.join(working_dir, 'results_long.csv')
    summary_path = os.path.join(working_dir, 'summary_stats.csv')
    validation_path = os.path.join(working_dir, 'validation_report.json')
    metadata_path = os.path.join(working_dir, 'reproducibility_metadata.json')

    rows = []
    metadata = {
        'package_versions': {
            'python': platform.python_version(),
            'omnirec': getattr(__import__('omnirec'), '__version__', 'unknown'),
            'lenskit': getattr(__import__('lenskit'), '__version__', 'unknown'),
        },
        'algorithms': {},
        'datasets': {},
    }

    def inspect_and_map_raw(path, kind):
        if kind == 'movielens':
            raw = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        elif kind == 'amazon':
            raw = pd.read_csv(path)
            cols = list(raw.columns)
            user_col = next(c for c in cols if c.lower() in {'reviewerid', 'user_id', 'user', 'userid'})
            item_col = next(c for c in cols if c.lower() in {'asin', 'item_id', 'item', 'productid'})
            rating_col = next(c for c in cols if c.lower() in {'overall', 'rating', 'score'})
            time_col = next((c for c in cols if 'time' in c.lower()), None)
            keep = [user_col, item_col, rating_col] + ([time_col] if time_col else [])
            raw = raw[keep].copy()
            raw.columns = ['user', 'item', 'rating'] + (['timestamp'] if time_col else [])
            if 'timestamp' not in raw.columns:
                raw['timestamp'] = 0
        else:
            raw = pd.read_csv(path, sep='\t', header=None, comment='#', engine='python')
            if raw.shape[1] >= 3:
                raw = raw.iloc[:, :4].copy()
                raw.columns = ['user', 'item', 'timestamp', 'tag'][:raw.shape[1]]
            if 'timestamp' not in raw.columns:
                raw['timestamp'] = 0
            raw['rating'] = 1.0
            raw = raw[['user', 'item', 'rating', 'timestamp']].copy()
        return raw

    def iterative_core5(df):
        df = df.copy()
        while True:
            ucnt = df.groupby('user')['item'].transform('count')
            icnt = df.groupby('item')['user'].transform('count')
            keep = (ucnt >= 5) & (icnt >= 5)
            new_df = df.loc[keep].copy()
            if len(new_df) == len(df):
                return new_df
            df = new_df

    def user_holdout_80_20(df, seed):
        rng = np.random.default_rng(seed)
        train_idx = []
        test_idx = []
        for _, grp in df.groupby('user', sort=False):
            idx = grp.index.to_numpy()
            perm = rng.permutation(idx)
            n_test = max(1, int(math.ceil(len(perm) * 0.2)))
            test_idx.extend(perm[:n_test])
            train_idx.extend(perm[n_test:])
        train = df.loc[train_idx].copy()
        test = df.loc[test_idx].copy()
        return train, test

    def make_dataset_from_df(df):
        return df

    algo_map = {
        'ALS': LensKit.ImplicitMFScorer,
        'ItemKNN': LensKit.ItemKNNScorer,
        'MostPopular': LensKit.PopScorer,
    }
    metadata['algorithms'] = {name: {'algorithm_class': str(cls)} for name, cls in algo_map.items()}

    for dataset_name, spec in datasets.items():
        raw = inspect_and_map_raw(spec['raw_path'], spec['kind'])
        raw_rows = len(raw)
        if dataset_name == 'Last.FM' and 'rating' in raw.columns:
            raw = raw.drop(columns=['rating'])
            raw['rating'] = 1.0
        if dataset_name in {'MovieLens100K', 'Amazon Video Games'}:
            thresholded = raw.loc[raw['rating'] > 3, ['user', 'item', 'timestamp']].copy()
            thresholded['rating'] = 1.0
        else:
            thresholded = raw[['user', 'item', 'rating', 'timestamp']].copy()
            thresholded['rating'] = 1.0
        thresholded = thresholded[['user', 'item', 'rating', 'timestamp']].copy()
        core = iterative_core5(thresholded)
        metadata['datasets'][dataset_name] = {
            'raw_interactions': int(raw_rows),
            'after_thresholding': int(len(thresholded)),
            'after_core5': int(len(core)),
            'per_seed': {},
        }

        for seed in seeds:
            set_random_state(seed)
            train, test = user_holdout_80_20(core, seed)
            seed_dir = os.path.join(working_dir, f'{dataset_name.replace(" ", "_")}_seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            metadata['datasets'][dataset_name]['per_seed'][str(seed)] = {
                'train_interactions': int(len(train)),
                'test_interactions': int(len(test)),
                'test_fraction': float(len(test) / max(1, len(core))),
                'output_dir': seed_dir,
            }

            train_ds = make_dataset_from_df(train)
            for algo_name, algo_cls in algo_map.items():
                plan = ExperimentPlan(plan_name=f'{dataset_name}_{algo_name}_{seed}')
                plan.add_algorithm(algo_cls)
                evaluator = Evaluator(NDCG(cutoffs), Precision(cutoffs))
                _ = run_omnirec(datasets=train_ds, plan=plan, evaluator=evaluator)
                result_df = None
                if hasattr(_, 'results'):
                    result_df = _.results
                elif isinstance(_, pd.DataFrame):
                    result_df = _
                if result_df is None:
                    result_df = pd.DataFrame({col: [] for col in ['metric', 'k', 'value']})
                if len(result_df) != 6:
                    raise RuntimeError(f'Expected 6 metric rows for {dataset_name}/{algo_name}/seed {seed}, got {len(result_df)}')
                if not np.isfinite(result_df['value']).all():
                    raise RuntimeError(f'Non-finite metric value for {dataset_name}/{algo_name}/seed {seed}')
                for _, r in result_df.iterrows():
                    rows.append({
                        'dataset': dataset_name,
                        'algorithm': algo_name,
                        'algorithm_class': str(algo_cls),
                        'seed': seed,
                        'metric': str(r['metric']),
                        'k': int(r['k']),
                        'value': float(r['value']),
                    })
                pd.DataFrame(rows).to_csv(results_path, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)

    summary = results.groupby(['dataset', 'algorithm', 'metric', 'k'])['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary['range'] = summary['max'] - summary['min']
    summary.to_csv(summary_path, index=False)

    checks = []
    expected = 3 * 3 * 5 * 2 * 3
    checks.append({'check': 'A', 'pass': len(results) == expected, 'detail': f'coverage {len(results)}/{expected}'})
    finite_ok = len(results) == expected and np.isfinite(results['value']).all() and results['value'].between(0, 1).all()
    checks.append({'check': 'B', 'pass': finite_ok, 'detail': 'all values finite in [0,1]' if finite_ok else 'invalid values present'})
    nondegenerate = not any(g['value'].eq(0).all() for _, g in results.groupby(['dataset', 'algorithm']))
    checks.append({'check': 'C', 'pass': nondegenerate, 'detail': 'no all-zero dataset-algorithm block'})
    ndcg1 = results[results['metric'] == 'nDCG']
    prec1 = results[results['metric'] == 'Precision']
    merged = ndcg1.merge(prec1, on=['dataset', 'algorithm', 'algorithm_class', 'seed', 'k'], suffixes=('_ndcg', '_prec'))
    identity_ok = np.allclose(merged.loc[merged['k'] == 1, 'value_ndcg'], merged.loc[merged['k'] == 1, 'value_prec'], atol=1e-6)
    checks.append({'check': 'D', 'pass': bool(identity_ok), 'detail': 'nDCG@1 equals Precision@1'})
    seed_effect = any(g['value'].nunique() > 1 for _, g in ndcg1.groupby(['dataset', 'algorithm', 'k']))
    checks.append({'check': 'E', 'pass': bool(seed_effect), 'detail': 'at least one block varies across seeds'})
    checks.append({'check': 'F', 'pass': True, 'detail': 'preprocessing counts and per-seed train/test counts recorded'})

    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump({'checks': checks, 'coverage': f'{len(results)}/{expected}', 'failed_checks': sum(not c['pass'] for c in checks)}, f, indent=2)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(results.head())
    print(summary.head())
    print(json.dumps({'checks': checks, 'coverage': f'{len(results)}/{expected}', 'failed_checks': sum(not c['pass'] for c in checks)}, indent=2))


if __name__ == '__main__':
    main()
