import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnirec import RecSysDataSet, DatasetMeta, register_dataloader, NDCG
from omnirec.metrics.ranking import Precision
from omnirec.data_loaders.base import Loader, DatasetInfo
from omnirec.data_loaders.datasets import DataSet
from omnirec.data_variants import RawData
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.preprocess.pipe import Pipe
from omnirec.preprocess.split import UserHoldout
from omnirec.runner.algos import LensKit
from omnirec.runner.evaluation import Evaluator
from omnirec.runner.plan import ExperimentPlan
from omnirec.util.run import run_omnirec
from omnirec.util.util import get_random_state, set_random_state


class LocalAmazonVideoGamesLoader(Loader):
    @staticmethod
    def info(name: str) -> DatasetInfo:
        return DatasetInfo(download_urls=[])

    @staticmethod
    def load(source_dir: Path, name: str) -> pd.DataFrame:
        path = source_dir / 'VideoGames.csv'
        if not path.exists():
            raise FileNotFoundError(f'Amazon Video Games file not found: {path}')

        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}

        user_col = None
        item_col = None
        rating_col = None
        time_col = None

        for candidate in ['user', 'userid', 'reviewerid']:
            if candidate in cols:
                user_col = cols[candidate]
                break
        for candidate in ['item', 'itemid', 'asin', 'productid']:
            if candidate in cols:
                item_col = cols[candidate]
                break
        for candidate in ['rating', 'score', 'overall']:
            if candidate in cols:
                rating_col = cols[candidate]
                break
        for candidate in ['timestamp', 'time', 'unixreviewtime']:
            if candidate in cols:
                time_col = cols[candidate]
                break

        if user_col is None or item_col is None:
            if df.shape[1] >= 4:
                temp = df.iloc[:, :4].copy()
                temp.columns = ['user', 'item', 'rating', 'timestamp']
                return temp
            raise RuntimeError(f'Could not infer Amazon columns from {list(df.columns)}')

        out = pd.DataFrame({
            'user': df[user_col],
            'item': df[item_col],
            'rating': pd.to_numeric(df[rating_col], errors='coerce') if rating_col is not None else 1,
            'timestamp': pd.to_numeric(df[time_col], errors='coerce') if time_col is not None else np.arange(len(df), dtype=np.int64),
        })
        out = out.dropna(subset=['user', 'item', 'rating', 'timestamp']).copy()
        return out


class LocalLastFMLoader(Loader):
    @staticmethod
    def info(name: str) -> DatasetInfo:
        return DatasetInfo(download_urls=[])

    @staticmethod
    def load(source_dir: Path, name: str) -> pd.DataFrame:
        path = source_dir / 'UserTaggedArtiststimestamps.dat'
        if not path.exists():
            alt = source_dir / 'user_taggedartists-timestamps.dat'
            path = alt if alt.exists() else path
        if not path.exists():
            raise FileNotFoundError(f'Last.FM file not found: {path}')

        df = pd.read_csv(path, sep='\t')
        cols = {c.lower(): c for c in df.columns}
        user_col = cols.get('userid') or cols.get('user')
        item_col = cols.get('artistid') or cols.get('item')
        time_col = cols.get('timestamp')
        if user_col is None or item_col is None:
            raise RuntimeError(f'Could not infer Last.FM columns from {list(df.columns)}')

        out = pd.DataFrame({
            'user': df[user_col],
            'item': df[item_col],
            'rating': 1,
            'timestamp': pd.to_numeric(df[time_col], errors='coerce') if time_col is not None else np.arange(len(df), dtype=np.int64),
        })
        out = out.dropna(subset=['user', 'item', 'timestamp']).copy()
        return out


try:
    register_dataloader('LocalAmazonVideoGames', LocalAmazonVideoGamesLoader)
except Exception:
    pass

try:
    register_dataloader('LocalLastFM', LocalLastFMLoader)
except Exception:
    pass


def safe_stat(values: Any) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float('nan'), float('nan')
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def make_raw_dataset(name: str, df: pd.DataFrame) -> RecSysDataSet[RawData]:
    meta = DatasetMeta(name=name)
    return RecSysDataSet(data=RawData(df), meta=meta)


def load_dataset_by_name(dataset_name: str, data_root: Path) -> RecSysDataSet:
    if dataset_name == 'MovieLens100K':
        canon_path = data_root / 'movielens100k_local_canonical.csv'
        return RecSysDataSet.use_dataloader(
            DataSet.MovieLens100K,
            raw_dir=str(data_root),
            canon_path=str(canon_path),
        )
    if dataset_name == 'AmazonVideoGames':
        return RecSysDataSet.use_dataloader('LocalAmazonVideoGames', raw_dir=str(data_root), canon_path=str(data_root / 'amazon_video_games_canonical.csv'))
    if dataset_name == 'LastFM':
        return RecSysDataSet.use_dataloader('LocalLastFM', raw_dir=str(data_root), canon_path=str(data_root / 'lastfm_local_canonical.csv'))
    raise ValueError(f'Unknown dataset: {dataset_name}')


def build_dataset(dataset_name: str, data_root: Path) -> RecSysDataSet:
    dataset = load_dataset_by_name(dataset_name, data_root)

    steps = []
    if dataset_name in {'MovieLens100K', 'AmazonVideoGames'}:
        steps.append(MakeImplicit(3))
    steps.append(CorePruning(5))
    steps.append(UserHoldout(0.01, 0.2))

    pipeline = Pipe(*steps)
    return pipeline.process(dataset)


def extract_metrics(evaluator: Evaluator) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = evaluator.get_results()
    if not results:
        raise RuntimeError('Evaluator returned no results after run_omnirec().')

    frames = []
    for dataset_id, df in results.items():
        cur = df.copy()
        cur['dataset_id'] = dataset_id
        frames.append(cur)

    all_results = pd.concat(frames, ignore_index=True)
    required_cols = {'algorithm', 'name', 'k', 'value', 'dataset_id'}
    missing = required_cols - set(all_results.columns)
    if missing:
        raise RuntimeError(f'Missing expected Evaluator result columns: {sorted(missing)}')

    all_results['name_norm'] = all_results['name'].astype(str).str.upper()
    all_results['k_num'] = pd.to_numeric(all_results['k'], errors='coerce')
    all_results['value_num'] = pd.to_numeric(all_results['value'], errors='coerce')

    rows = []
    for _, grp in all_results.groupby(['dataset_id', 'algorithm'], dropna=False):
        row = {
            'dataset_id': grp['dataset_id'].iloc[0],
            'algorithm': grp['algorithm'].iloc[0],
        }
        for metric_name in ['NDCG', 'PRECISION']:
            for k in [1, 5, 10]:
                sel = grp[(grp['name_norm'] == metric_name) & (grp['k_num'] == k)]['value_num'].dropna()
                row[f'{metric_name.lower()}@{k}'] = float(sel.iloc[0]) if not sel.empty else float('nan')
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    return metrics_df, all_results


def run_single_seed(seed: int, dataset_name: str, working_dir: Path, data_root: Path) -> pd.DataFrame:
    print('=' * 80)
    print(f'Running dataset={dataset_name}, seed={seed}')
    set_random_state(seed)
    print(f'OmniRec random state set to: {get_random_state()}')

    dataset = build_dataset(dataset_name, data_root)
    print('Processed dataset summary:')
    print(dataset)
    try:
        print(dataset.format_details())
    except Exception:
        pass

    plan = ExperimentPlan(plan_name=f'{dataset_name}_seed_{seed}')
    plan.add_algorithm(LensKit.PopScorer)
    plan.add_algorithm(LensKit.ItemKNNScorer)
    plan.add_algorithm(LensKit.ImplicitMFScorer)

    evaluator = Evaluator(
        NDCG([1, 5, 10]),
        Precision([1, 5, 10]),
    )

    seed_dir = working_dir / dataset_name / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    cwd_before = Path.cwd()
    os.chdir(seed_dir)
    try:
        run_omnirec(
            datasets=dataset,
            plan=plan,
            evaluator=evaluator,
        )
    finally:
        os.chdir(cwd_before)

    metrics_df, raw_results = extract_metrics(evaluator)
    metrics_df['dataset'] = dataset_name
    metrics_df['seed'] = seed

    raw_results_path = seed_dir / 'evaluator_results.csv'
    raw_results.to_csv(raw_results_path, index=False)
    metrics_path = seed_dir / 'metrics_wide.csv'
    metrics_df.to_csv(metrics_path, index=False)
    try:
        evaluator.save_results(seed_dir / 'evaluator_results.json')
    except Exception:
        pass

    print('Seed metrics:')
    print(metrics_df[['dataset', 'seed', 'algorithm', 'ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']])
    return metrics_df


def make_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    rows = []
    grouped = results_df.groupby(['dataset', 'algorithm'])
    for key, grp in grouped:
        dataset, algorithm = cast(tuple[object, object], key)
        row = {'dataset': dataset, 'algorithm': algorithm, 'n_seeds': int(grp['seed'].nunique())}
        for col in metric_cols:
            mean, std = safe_stat(grp[col].tolist())
            row[f'{col}_mean'] = mean
            row[f'{col}_std'] = std
            row[f'{col}_cv'] = (std / mean) if pd.notna(mean) and mean not in (0, 0.0) else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def make_stat_analysis(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_bases = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    rows = []
    for metric in metric_bases:
        std_col = f'{metric}_std'
        cv_col = f'{metric}_cv'
        sub = summary_df[['dataset', 'algorithm', std_col, cv_col]].copy()
        sub = sub.dropna(subset=[std_col])
        if sub.empty:
            continue
        max_idx = sub[std_col].idxmax()
        min_idx = sub[std_col].idxmin()
        rows.append({
            'metric': metric,
            'avg_std_across_pairs': float(sub[std_col].mean()),
            'avg_cv_across_pairs': float(sub[cv_col].dropna().mean()) if not sub[cv_col].dropna().empty else float('nan'),
            'most_sensitive_dataset': sub.loc[max_idx, 'dataset'],
            'most_sensitive_algorithm': sub.loc[max_idx, 'algorithm'],
            'most_sensitive_std': float(sub.loc[max_idx, std_col]),
            'least_sensitive_dataset': sub.loc[min_idx, 'dataset'],
            'least_sensitive_algorithm': sub.loc[min_idx, 'algorithm'],
            'least_sensitive_std': float(sub.loc[min_idx, std_col]),
        })
    return pd.DataFrame(rows)


def save_plots(results_df: pd.DataFrame, working_dir: Path) -> None:
    metric_cols = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    plots_dir = working_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped = results_df.groupby(['dataset', 'algorithm'])
    for key, grp in grouped:
        dataset, algorithm = cast(tuple[object, object], key)
        grp = grp.sort_values('seed')
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.ravel()
        for ax, metric in zip(axes, metric_cols):
            ax.plot(grp['seed'], grp[metric], marker='o')
            ax.set_title(metric.upper())
            ax.set_xlabel('Seed')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'{dataset} - {algorithm} across split seeds')
        plt.tight_layout()
        out = plots_dir / f'{dataset}_{str(algorithm).replace("/", "_").replace(".", "_")}_seed_sensitivity.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    working_dir = Path(working_dir)

    data_root = (Path(os.getcwd()) / '../../../study/data').resolve()
    print(f'Using data root: {data_root}')
    print(f'Expected local source directory: {data_root}')
    print('Using OmniRec exclusively with LensKit algorithms via OmniRec runner.')
    print('Note: OmniRec UserHoldout is documented as UserHoldout(validation, test); keeping a tiny validation split (0.01) and test_size=0.2 to preserve the working prototype behavior.')

    seeds = [7, 13, 29, 42, 87]
    datasets = ['MovieLens100K', 'AmazonVideoGames', 'LastFM']
    all_results = []

    for dataset_name in datasets:
        for seed in seeds:
            res_df = run_single_seed(seed, dataset_name, working_dir, data_root)
            all_results.append(res_df)

    results_df = pd.concat(all_results, ignore_index=True)
    metric_cols = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']
    ordered_cols = ['dataset', 'seed', 'algorithm'] + metric_cols + ['dataset_id']
    ordered_cols = [c for c in ordered_cols if c in results_df.columns]
    results_df = results_df[ordered_cols]

    results_csv = working_dir / 'prototype_results.csv'
    results_df.to_csv(results_csv, index=False)

    print('\nPer-run results table:')
    print(results_df)

    summary = make_summary(results_df)
    summary_csv = working_dir / 'prototype_summary.csv'
    summary.to_csv(summary_csv, index=False)

    print('\nSummary over seeds:')
    print(summary)

    stat_analysis = make_stat_analysis(summary)
    stat_csv = working_dir / 'seed_sensitivity_stats.csv'
    stat_analysis.to_csv(stat_csv, index=False)

    print('\nShort statistical analysis:')
    print(stat_analysis)
    if not stat_analysis.empty:
        for _, row in stat_analysis.iterrows():
            print(
                f"{row['metric']}: avg std={row['avg_std_across_pairs']:.6f}; most sensitive={row['most_sensitive_dataset']} / {row['most_sensitive_algorithm']} (std={row['most_sensitive_std']:.6f}); least sensitive={row['least_sensitive_dataset']} / {row['least_sensitive_algorithm']} (std={row['least_sensitive_std']:.6f})"
            )

    save_plots(results_df, working_dir)

    print(f'\nSaved per-run results to: {results_csv}')
    print(f'Saved summary to: {summary_csv}')
    print(f'Saved statistical analysis to: {stat_csv}')
    print(f'Saved plots to: {working_dir / "plots"}')
    print('\nPrototype complete.')
    print('This extends the minimal pilot to multiple datasets, algorithms, metrics, and seed-sensitivity analysis while preserving the original structure.')


if __name__ == '__main__':
    main()
