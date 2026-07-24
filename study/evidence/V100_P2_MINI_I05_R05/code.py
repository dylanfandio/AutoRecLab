import os
import json
from importlib import metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omnirec import RecSysDataSet
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.plan import ExperimentPlan
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.metrics.ranking import NDCG, Precision
from omnirec.util.run import run_omnirec
from omnirec.util.util import set_random_state


def ensure_working_dir():
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def version_safe(pkg_name):
    try:
        return metadata.version(pkg_name)
    except Exception:
        return 'unknown'


def inspect_and_load_ml100k(raw_dir):
    raw_path = os.path.join(raw_dir, 'u.data')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'MovieLens100K raw file not found: {raw_path}')
    df = pd.read_csv(raw_path, sep='\t', header=None)
    if df.shape[1] != 4:
        raise ValueError(f'Unexpected MovieLens100K schema with {df.shape[1]} columns; expected 4')
    df.columns = ['user', 'item', 'rating', 'timestamp']
    return df, raw_path, {'sep': '\t', 'columns': ['user', 'item', 'rating', 'timestamp']}


def inspect_and_load_amazon_vg(raw_dir):
    raw_path = os.path.join(raw_dir, 'VideoGames.csv')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'Amazon Video Games raw file not found: {raw_path}')

    # The local file may be headerless and not actually comma-delimited; try the robust headerless forms first.
    attempts = [
        ('\t', None),
        (',', None),
        ('\t', 0),
        (',', 0),
    ]
    df = None
    used = None
    for sep, header in attempts:
        try:
            cand = pd.read_csv(raw_path, sep=sep, header=header)
            if header is None and cand.shape[1] == 4:
                df = cand
                used = (sep, header)
                break
            if header == 0 and cand.shape[1] >= 4:
                df = cand
                used = (sep, header)
                break
        except Exception:
            continue
    if df is None or used is None:
        raise ValueError('Unable to parse Amazon Video Games file with expected 4-column schema')

    if df.shape[1] == 4 and df.columns.tolist() == [0, 1, 2, 3]:
        df.columns = ['user', 'item', 'rating', 'timestamp']
        mapped = {'user': 0, 'item': 1, 'rating': 2, 'timestamp': 3}
    else:
        cols = {str(c).lower(): c for c in df.columns}
        user_col = cols.get('user_id') or cols.get('reviewerid') or cols.get('user') or df.columns[0]
        item_col = cols.get('item_id') or cols.get('asin') or cols.get('item') or df.columns[1]
        rating_col = cols.get('rating') or cols.get('overall') or cols.get('stars') or df.columns[2]
        timestamp_col = cols.get('timestamp') or cols.get('unixreviewtime') or cols.get('reviewtime') or df.columns[3]
        df = pd.DataFrame({
            'user': df[user_col],
            'item': df[item_col],
            'rating': df[rating_col],
            'timestamp': df[timestamp_col],
        })
        mapped = {'user': str(user_col), 'item': str(item_col), 'rating': str(rating_col), 'timestamp': str(timestamp_col)}
    return df, raw_path, {'parsed_with': {'sep': used[0], 'header': used[1]}, 'mapped': mapped, 'columns': list(df.columns)}


def inspect_and_load_lastfm(raw_dir):
    raw_path = os.path.join(raw_dir, 'UserTaggedArtiststimestamps.dat')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'Last.FM raw file not found: {raw_path}')
    df = pd.read_csv(raw_path, sep='\t', header=None)
    if df.shape[1] < 3:
        raise ValueError(f'Unexpected Last.FM schema with {df.shape[1]} columns; expected at least 3')
    if df.shape[1] == 3:
        df.columns = ['user', 'item', 'timestamp']
    else:
        df.columns = ['user', 'item', 'tag', 'timestamp'] + [f'extra_{i}' for i in range(df.shape[1] - 4)]
    out = pd.DataFrame({
        'user': df['user'],
        'item': df['item'],
        'rating': 1,
        'timestamp': df['timestamp'] if 'timestamp' in df.columns else np.arange(len(df))
    })
    return out, raw_path, {'sep': '\t', 'columns': list(df.columns), 'mapped': {'user': 'user', 'item': 'item', 'timestamp': 'timestamp', 'rating': 'implicit_1'}}


def canonicalize_ids(df):
    out = df.copy()
    out['user'] = pd.Categorical(out['user']).codes
    out['item'] = pd.Categorical(out['item']).codes
    return out.reset_index(drop=True)


def core_filter(df, core=5):
    current = df.copy().reset_index(drop=True)
    changed = True
    while changed:
        before = len(current)
        user_counts = current.groupby('user')['item'].size()
        keep_users = user_counts[user_counts >= core].index
        current = current[current['user'].isin(keep_users)]
        item_counts = current.groupby('item')['user'].size()
        keep_items = item_counts[item_counts >= core].index
        current = current[current['item'].isin(keep_items)]
        current = canonicalize_ids(current)
        changed = len(current) != before
    return current


def save_jsonl_row(path, row):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


def load_or_empty_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preprocess_dataset(raw_df, dataset_name):
    counts = {
        'raw_interactions': int(len(raw_df)),
        'raw_users': int(raw_df['user'].nunique()),
        'raw_items': int(raw_df['item'].nunique()),
    }
    if dataset_name in ('MovieLens100K', 'Amazon Video Games'):
        raw_df = raw_df.loc[raw_df['rating'] > 3, ['user', 'item', 'rating', 'timestamp']].copy()
        raw_df['rating'] = 1
        counts['implicit_gt3_interactions'] = int(len(raw_df))
    else:
        raw_df = raw_df.loc[:, ['user', 'item', 'rating', 'timestamp']].copy()
        raw_df['rating'] = 1
        counts['implicit_events'] = int(len(raw_df))
    canon = canonicalize_ids(raw_df)
    filtered = core_filter(canon, core=5)
    counts['after_5core_interactions'] = int(len(filtered))
    counts['after_5core_users'] = int(filtered['user'].nunique())
    counts['after_5core_items'] = int(filtered['item'].nunique())
    return filtered, counts


def get_split_counts(split_obj):
    data = split_obj._data
    return int(len(data.train)), int(len(data.test))


def main():
    working_dir = ensure_working_dir()
    results_path = os.path.join(working_dir, 'results.jsonl')
    summary_path = os.path.join(working_dir, 'summary.csv')
    meta_path = os.path.join(working_dir, 'metadata.json')
    plot_path = os.path.join(working_dir, 'seed_sensitivity_ndcg10.png')

    raw_dir = os.path.abspath(os.path.join(os.getcwd(), '../../../study/data'))

    datasets_raw = {
        'MovieLens100K': inspect_and_load_ml100k(raw_dir),
        'Amazon Video Games': inspect_and_load_amazon_vg(raw_dir),
        'Last.FM': inspect_and_load_lastfm(raw_dir),
    }

    datasets = {}
    preprocessing_counts = {}
    raw_schemas = {}
    source_files = {}
    for name, (raw_df, raw_path, schema_info) in datasets_raw.items():
        processed, counts = preprocess_dataset(raw_df, name)
        datasets[name] = processed
        preprocessing_counts[name] = counts
        raw_schemas[name] = schema_info
        source_files[name] = raw_path

    versions = {
        'omnirec': version_safe('omnirec'),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'matplotlib': plt.matplotlib.__version__,
    }

    algorithms = [
        ('LensKit.ImplicitMFScorer', LensKit.ImplicitMFScorer, {}),
        ('LensKit.ItemKNNScorer', LensKit.ItemKNNScorer, {}),
        ('LensKit.PopScorer', LensKit.PopScorer, {}),
    ]
    seeds = [7, 13, 29, 42, 87]
    cutoffs = [1, 5, 10]

    completed_keys = set()
    for row in load_or_empty_jsonl(results_path):
        if 'error' not in row and 'metric' in row:
            completed_keys.add((row['dataset'], row['algorithm'], row['seed'], row['cutoff'], row['metric']))

    all_rows = []
    if os.path.exists(summary_path):
        try:
            prev = pd.read_csv(summary_path)
            all_rows.extend(prev.to_dict(orient='records'))
        except Exception:
            pass

    for dataset_name, df in datasets.items():
        dataset = RecSysDataSet(df)
        for algo_name, algo_cls, algo_cfg in algorithms:
            for seed in seeds:
                set_random_state(seed)
                split = UserHoldout(0.0, 0.2).process(dataset)
                train_count, test_count = get_split_counts(split)
                plan = ExperimentPlan(f'{dataset_name}_seed_{seed}_{algo_name}')
                plan.add_algorithm(algo_cls, dict(algo_cfg))
                evaluator = Evaluator(NDCG(cutoffs), Precision(cutoffs))
                try:
                    run_omnirec(datasets=split, plan=plan, evaluator=evaluator)
                    results = evaluator.get_results()
                    if len(results) == 0:
                        raise RuntimeError('Evaluator returned no results')
                    for dataset_id, res_df in results.items():
                        for _, r in res_df.iterrows():
                            if int(r['k']) not in cutoffs or r['name'] not in ('NDCG', 'Precision'):
                                continue
                            metric = 'ndcg' if r['name'] == 'NDCG' else 'precision'
                            key = (dataset_name, algo_name, seed, int(r['k']), metric)
                            if key in completed_keys:
                                continue
                            value = float(r['value'])
                            if not np.isfinite(value):
                                raise ValueError('Non-finite metric value')
                            out_row = {
                                'dataset': dataset_name,
                                'algorithm': algo_name,
                                'seed': int(seed),
                                'cutoff': int(r['k']),
                                'metric': metric,
                                'value': value,
                                'dataset_id': dataset_id,
                                'raw_file': source_files[dataset_name],
                                'train_interactions': train_count,
                                'test_interactions': test_count,
                            }
                            all_rows.append(out_row)
                            completed_keys.add(key)
                            save_jsonl_row(results_path, out_row)
                except Exception as exc:
                    save_jsonl_row(results_path, {'dataset': dataset_name, 'algorithm': algo_name, 'seed': int(seed), 'error': repr(exc)})
                    continue

    results_df = pd.DataFrame(all_rows)
    if not results_df.empty:
        results_df.to_csv(summary_path, index=False)
        ndcg10 = results_df[(results_df['metric'] == 'ndcg') & (results_df['cutoff'] == 10)].sort_values(['dataset', 'algorithm', 'seed'])
        if not ndcg10.empty:
            plt.figure(figsize=(7, 4))
            for (dataset_name, algo_name), grp in ndcg10.groupby(['dataset', 'algorithm']):
                plt.plot(grp['seed'], grp['value'], marker='o', label=f'{dataset_name} | {algo_name}')
            plt.title('Seed sensitivity: NDCG@10')
            plt.xlabel('Seed')
            plt.ylabel('NDCG@10')
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()

    if not results_df.empty:
        summary = (
            results_df.groupby(['dataset', 'algorithm', 'cutoff', 'metric'])['value']
            .agg(['mean', 'std', 'min', 'max'])
            .reset_index()
        )
        summary['range'] = summary['max'] - summary['min']
        summary.to_csv(os.path.join(working_dir, 'seed_summary.csv'), index=False)
    else:
        summary = pd.DataFrame()

    analysis_lines = []
    if not results_df.empty:
        for metric in ['ndcg', 'precision']:
            mdf = results_df[results_df['metric'] == metric]
            if not mdf.empty:
                spread = mdf.groupby(['dataset', 'algorithm', 'cutoff'])['value'].agg(lambda x: float(np.max(x) - np.min(x))).mean()
                analysis_lines.append(f'Average across-combination range for {metric.upper()}: {spread:.4f}.')

    meta = {
        'package_versions': versions,
        'algorithm_classes': {
            'ALS for implicit feedback': 'LensKit.ImplicitMFScorer',
            'ItemKNN': 'LensKit.ItemKNNScorer',
            'MostPopular': 'LensKit.PopScorer',
        },
        'raw_files': source_files,
        'raw_schemas_and_column_mapping': raw_schemas,
        'preprocessing_counts': preprocessing_counts,
        'seeds': seeds,
        'cutoffs': cutoffs,
        'experiment_note': 'user-based 80/20 train/test holdout only; no validation split',
        'train_test_counts_per_dataset_example': {
            ds: {
                'train_interactions': int(results_df[results_df['dataset'] == ds]['train_interactions'].iloc[0]) if not results_df[results_df['dataset'] == ds].empty else None,
                'test_interactions': int(results_df[results_df['dataset'] == ds]['test_interactions'].iloc[0]) if not results_df[results_df['dataset'] == ds].empty else None,
            } for ds in datasets.keys()
        },
        'seed_analysis': analysis_lines,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Done')
    print(meta)
    if not results_df.empty:
        print(results_df[['dataset', 'algorithm', 'seed', 'cutoff', 'metric', 'value']])
        if not summary.empty:
            print(summary)
        if analysis_lines:
            print('\n'.join(analysis_lines))


if __name__ == '__main__':
    main()
