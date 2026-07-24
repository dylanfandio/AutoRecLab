import os
import json
import math
import traceback
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet, NDCG
from omnirec.metrics.ranking import Precision
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.algos import LensKit
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state, get_random_state


def _ensure_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def _package_versions():
    versions = {}
    for pkg in ['omnirec', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'lenskit', 'matplotlib']:
        try:
            mod_name = pkg if pkg != 'scikit-learn' else 'sklearn'
            mod = __import__(mod_name)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except Exception:
            versions[pkg] = 'unavailable'
    return versions


def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None


def _count_df(obj):
    try:
        return int(len(obj))
    except Exception:
        return None


def _extract_counts(ds):
    out = {'num_interactions': None, 'num_users': None, 'num_items': None}
    for key in out:
        fn = getattr(ds, key, None)
        if callable(fn):
            try:
                out[key] = int(fn())
            except Exception:
                pass
    return out


def _load_local_raw(dataset_name, raw_relpath):
    base_dir = Path(os.getcwd()).resolve() / '../../../study/data'
    raw_path = (base_dir / raw_relpath).resolve()
    df = pd.read_csv(raw_path, sep=None, engine='python', header=None)
    schema_info = {'path': str(raw_path), 'columns': list(df.columns), 'shape': tuple(df.shape)}
    if dataset_name == 'MovieLens100K':
        df = pd.read_csv(raw_path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
        schema_info['mapping'] = {'user': 0, 'item': 1, 'rating': 2, 'timestamp': 3}
    elif dataset_name == 'Amazon Video Games':
        df = pd.read_csv(raw_path)
        cols = {c.lower(): c for c in df.columns}
        user_col = cols.get('user_id', next(iter(df.columns)))
        item_col = cols.get('parent_asin', cols.get('item_id', cols.get('asin', list(df.columns)[1])))
        rating_col = cols.get('rating', next((c for c in df.columns if 'rating' in c.lower()), None))
        ts_col = cols.get('timestamp', next((c for c in df.columns if 'time' in c.lower()), None))
        df = df.rename(columns={user_col: 'user', item_col: 'item'})
        if rating_col is not None:
            df = df.rename(columns={rating_col: 'rating'})
        else:
            df['rating'] = 5.0
        if ts_col is not None:
            df = df.rename(columns={ts_col: 'timestamp'})
        else:
            df['timestamp'] = 0
        df = df[['user', 'item', 'rating', 'timestamp']]
        schema_info['mapping'] = {'user': user_col, 'item': item_col, 'rating': rating_col, 'timestamp': ts_col}
    elif dataset_name == 'Last.FM':
        df = pd.read_csv(raw_path, sep='\t', header=None, names=['user', 'item', 'tag', 'timestamp'])
        df = df[['user', 'item', 'timestamp']].copy()
        df['rating'] = 1.0
        df = df[['user', 'item', 'rating', 'timestamp']]
        schema_info['mapping'] = {'user': 0, 'item': 1, 'timestamp': 3, 'rating': 'implicit-tag-event'}
    else:
        raise ValueError(dataset_name)
    return df, schema_info


def _to_rsds(df, name):
    tmp_dir = Path(_ensure_working_dir())
    csv_path = tmp_dir / f'{name}_canonical.csv'
    df.to_csv(csv_path, index=False)
    return RecSysDataSet.load(csv_path)


def _get_result_df(result):
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if hasattr(result, 'get_results'):
        try:
            res = result.get_results()
            if isinstance(res, dict):
                frames = []
                for ds_name, df in res.items():
                    dfx = df.copy()
                    if 'dataset' not in dfx.columns:
                        dfx['dataset'] = ds_name
                    frames.append(dfx)
                if frames:
                    return pd.concat(frames, ignore_index=True)
        except Exception:
            pass
    if hasattr(result, 'to_dataframe'):
        try:
            return result.to_dataframe()
        except Exception:
            pass
    return pd.DataFrame(result)


def _normalize_result_table(df):
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    out['dataset'] = df[cols.get('dataset', df.columns[0])] if 'dataset' in cols else df.iloc[:, 0]
    out['algorithm'] = df[cols.get('algorithm', df.columns[1])] if 'algorithm' in cols else df.iloc[:, 1]
    if 'seed' in cols:
        out['seed'] = df[cols['seed']]
    else:
        out['seed'] = None
    if 'cutoff' in cols:
        out['cutoff'] = df[cols['cutoff']]
    else:
        out['cutoff'] = None
    ndcg_col = next((c for c in df.columns if 'ndcg' in c.lower()), None)
    prec_col = next((c for c in df.columns if 'precision' in c.lower()), None)
    if ndcg_col:
        out['nDCG'] = df[ndcg_col]
    if prec_col:
        out['Precision'] = df[prec_col]
    return out


def main():
    working_dir = _ensure_working_dir()
    results_path = os.path.join(working_dir, 'prototype_results.csv')
    summary_path = os.path.join(working_dir, 'prototype_summary.csv')
    plot_path = os.path.join(working_dir, 'prototype_plot.png')
    meta_path = os.path.join(working_dir, 'prototype_metadata.json')

    seeds = [7, 13, 29, 42, 87]
    cutoffs = [1, 5, 10]
    datasets_spec = [
        ('MovieLens100K', 'u.data', 'ratings_gt3_implicit_5core'),
        ('Amazon Video Games', 'VideoGames.csv', 'ratings_gt3_implicit_5core'),
        ('Last.FM', 'UserTaggedArtiststimestamps.dat', 'tag_events_implicit_5core'),
    ]

    versions = _package_versions()
    print('Package versions:', versions)

    all_rows = []
    meta_rows = []
    if os.path.exists(results_path):
        try:
            all_rows = pd.read_csv(results_path).to_dict('records')
        except Exception:
            all_rows = []

    # Read and preprocess datasets up front so schema inspection is explicit and reproducible.
    processed_datasets = {}
    for dname, relpath, prep in datasets_spec:
        raw_df, schema_info = _load_local_raw(dname, relpath)
        raw_counts = {'raw_interactions': int(len(raw_df)), 'raw_users': int(raw_df['user'].nunique()), 'raw_items': int(raw_df['item'].nunique())}
        if dname in ('MovieLens100K', 'Amazon Video Games'):
            raw_df = raw_df[raw_df['rating'] > 3].copy()
        raw_df['rating'] = 1.0
        raw_df = raw_df[['user', 'item', 'rating', 'timestamp']].copy()
        ds = _to_rsds(raw_df, dname.replace(' ', '_'))
        pipe = Pipe(CorePruning(5))
        ds = pipe.process(ds)
        processed_datasets[dname] = {
            'dataset': ds,
            'schema': schema_info,
            'raw_counts': raw_counts,
            'processed_counts': _extract_counts(ds),
        }

    plan = ExperimentPlan(plan_name='Local_Three_Dataset_Comparison')
    plan.add_algorithm(LensKit.ImplicitMFScorer, {'embedding_size': 64})
    plan.add_algorithm(LensKit.ItemKNNScorer, {'max_nbrs': 40, 'min_nbrs': 5})
    plan.add_algorithm(LensKit.PopScorer, {})

    evaluator = Evaluator(NDCG(cutoffs), Precision(cutoffs))

    for seed in seeds:
        set_random_state(seed)
        for dname, info in processed_datasets.items():
            split_pipe = Pipe(UserHoldout(0.2, 0.0))
            ds_split = split_pipe.process(info['dataset'])
            result = run_omnirec(datasets=ds_split, plan=plan, evaluator=evaluator)
            df = _normalize_result_table(_get_result_df(result))
            if df.empty:
                raise RuntimeError(f'No results returned for {dname} seed {seed}')
            for _, row in df.iterrows():
                algo = str(row.get('algorithm', 'unknown'))
                for cutoff in cutoffs:
                    ndcg_val = None
                    prec_val = None
                    for c in df.columns:
                        if 'ndcg' in c.lower() and str(cutoff) in c:
                            ndcg_val = _safe_num(row[c])
                        if 'precision' in c.lower() and str(cutoff) in c:
                            prec_val = _safe_num(row[c])
                    if ndcg_val is None or prec_val is None:
                        continue
                    if not (math.isfinite(ndcg_val) and math.isfinite(prec_val)):
                        continue
                    out_row = {'dataset': dname, 'algorithm': algo, 'seed': seed, 'cutoff': cutoff, 'nDCG': ndcg_val, 'Precision': prec_val}
                    all_rows = [r for r in all_rows if not (r.get('dataset') == dname and r.get('algorithm') == algo and r.get('seed') == seed and r.get('cutoff') == cutoff)]
                    all_rows.append(out_row)
                    pd.DataFrame(all_rows).to_csv(results_path, index=False)
                    print('Saved row:', out_row)

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        raise RuntimeError('No finite results were produced.')

    summary_df = results_df.groupby(['dataset', 'algorithm', 'cutoff'], as_index=False).agg(
        nDCG_mean=('nDCG', 'mean'),
        nDCG_std=('nDCG', 'std'),
        nDCG_min=('nDCG', 'min'),
        nDCG_max=('nDCG', 'max'),
        nDCG_range=('nDCG', lambda s: s.max() - s.min()),
        Precision_mean=('Precision', 'mean'),
        Precision_std=('Precision', 'std'),
        Precision_min=('Precision', 'min'),
        Precision_max=('Precision', 'max'),
        Precision_range=('Precision', lambda s: s.max() - s.min()),
    )
    summary_df.to_csv(summary_path, index=False)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'versions': versions,
            'seeds': seeds,
            'cutoffs': cutoffs,
            'algorithms': ['LensKit.ImplicitMFScorer', 'LensKit.ItemKNNScorer', 'LensKit.PopScorer'],
            'datasets': [{
                'name': k,
                'schema': v['schema'],
                'raw_counts': v['raw_counts'],
                'processed_counts': v['processed_counts'],
            } for k, v in processed_datasets.items()],
            'results_path': results_path,
            'summary_path': summary_path,
        }, f, indent=2)

    if not results_df.empty:
        plt.figure(figsize=(8, 4))
        grouped = results_df.groupby(['dataset', 'algorithm'])['nDCG'].mean().reset_index()
        labels = grouped['dataset'] + '\n' + grouped['algorithm']
        plt.bar(range(len(grouped)), grouped['nDCG'])
        plt.xticks(range(len(grouped)), labels, rotation=45, ha='right')
        plt.ylabel('Mean nDCG across seeds/cutoffs')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    seed_spread = results_df.groupby(['dataset', 'algorithm', 'cutoff'])[['nDCG', 'Precision']].agg(['mean', 'std', 'min', 'max'])
    print('\nResults table:')
    print(results_df[['dataset', 'algorithm', 'seed', 'cutoff', 'nDCG', 'Precision']].to_string(index=False))
    print('\nSummary table:')
    print(summary_df.to_string(index=False))
    print('\nSeed sensitivity analysis:')
    print('Higher std/range means the random 80/20 user holdout has a stronger effect on accuracy; lower values indicate more stable ranking quality across seeds.')
    print(seed_spread.to_string())
    print('Raw/processed metadata saved to:', meta_path)
    print('Completed experiment run.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Prototype run failed:', e)
        traceback.print_exc()
        raise
