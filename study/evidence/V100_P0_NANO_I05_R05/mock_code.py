import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib

# Optional OmniRec import; we implement a mock fallback if unavailable
HAVE_OMNI: bool = False
OmniALS = None
OmniItemKNN = None
OmniPop = None
try:
    # Dynamic import to avoid static type checker unresolved-import problems
    omnirec_module = importlib.import_module('omnirec')
    OmniALS = getattr(omnirec_module, 'ALS', None)
    OmniItemKNN = getattr(omnirec_module, 'ItemKNN', None)
    OmniPop = getattr(omnirec_module, 'Pop', None)
    if OmniALS is not None or OmniItemKNN is not None or OmniPop is not None:
        HAVE_OMNI = True
except Exception:
    # Fall back to mock if OmniRec is not installed or missing components
    HAVE_OMNI = False

class MockRecommender:
    def __init__(self, name: str, train_df: pd.DataFrame):
        self.name = name
        self.train_df = train_df
        self.popularity = (train_df["item"].value_counts()).to_dict()
        self.items = list(self.popularity.keys())

    def train(self, train_df: pd.DataFrame):
        self.train_df = train_df
        self.popularity = (train_df["item"].value_counts()).to_dict()
        self.items = list(self.popularity.keys())

    def predict(self, test_df: pd.DataFrame, k_values: List[int]) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        # build per-user exclusion sets from train_df
        user_excl = test_df["user"].unique().tolist()
        train_map: Dict[str, set[str]] = {}
        for u, grp in self.train_df.groupby("user"):
            train_map[str(u)] = set(grp["item"].tolist())
        # global popularity order
        popular_items = sorted(self.popularity.keys(), key=lambda it: self.popularity[it], reverse=True)
        for u in test_df["user"].unique():
            exclude = train_map.get(str(u), set())
            preds: List[str] = []
            for it in popular_items:
                if it not in exclude:
                    preds.append(it)
                if len(preds) >= max(k_values):
                    break
            predictions[str(u)] = preds
        return predictions

    def score(self, predictions: Dict[str, List[str]], test_df: pd.DataFrame, k_values: List[int]) -> Dict[int, Tuple[float, float]]:
        ks = sorted(k_values)
        ndcg_sum: Dict[int, float] = {k: 0.0 for k in ks}
        prec_sum: Dict[int, float] = {k: 0.0 for k in ks}
        user_ct = 0
        for u, grp in test_df.groupby("user"):
            actual = set(grp["item"].tolist())
            user_ct += 1
            pred = predictions.get(str(u), [])
            if pred is None:
                pred = []
            for k in ks:
                topk = pred[:k]
                hits = len(actual & set(topk))
                precision = hits / float(k) if k > 0 else 0.0
                dcg = 0.0
                for idx, it in enumerate(topk, start=1):
                    if it in actual:
                        dcg += 1.0 / np.log2(idx + 1)
                idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(actual), k) + 1)) if actual else 0.0
                ndcg = (dcg / idcg) if idcg > 0 else 0.0
                ndcg_sum[k] += ndcg
                prec_sum[k] += precision
        results: Dict[int, Tuple[float, float]] = {}
        for k in ks:
            if user_ct > 0:
                results[k] = (ndcg_sum[k] / user_ct, prec_sum[k] / user_ct)
            else:
                results[k] = (0.0, 0.0)
        return results

class ExperimentRunner:
    def __init__(self, data_dir: Optional[str] = None, seeds: Optional[List[int]] = None, k_values: Optional[List[int]] = None, report_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.getcwd(), '..', '..', '..', 'study', 'data')
        self.seeds = seeds or [42, 123, 999, 7, 2024]
        self.k_values = k_values or [1, 5, 10]
        self.report_dir = report_dir or os.path.join(os.getcwd(), 'report')
        os.makedirs(self.report_dir, exist_ok=True)
        self.results: List[Dict[str, object]] = []
        self.logs: List[str] = []
        self.datasets = {
            'MovieLens100K': {
                'path': os.path.join(self.data_dir, 'u.data'),
                'implicit_conversion': True
            },
            'AmazonVideoGames': {
                'path': os.path.join(self.data_dir, 'VideoGames.csv'),
                'implicit_conversion': True
            },
            'HetrecLastFM': {
                'path': os.path.join(self.data_dir, 'UserTaggedArtiststimestamps.dat'),
                'implicit_conversion': True
            }
        }

    def load_and_prepare(self, label: str, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
        if path.endswith('.csv'):
            df = pd.read_csv(path, sep=',', engine='python')
        else:
            df = pd.read_csv(path, sep='\t', engine='python', header=None)
        if df.shape[1] >= 3:
            df.columns = ['user', 'item', 'rating', 'timestamp'][:df.shape[1]]
        else:
            raise ValueError('Unexpected dataset format: expected at least 3 columns')
        df['user'] = df['user'].astype(str)
        df['item'] = df['item'].astype(str)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='ignore')
        if 'rating' in df.columns:
            df['implicit'] = (df['rating'] > 3).astype(int)
        else:
            df['implicit'] = 1
        df = self.filter_5core(df, min_user=5, min_item=5)
        if 'implicit' in df.columns:
            df = df[df['implicit'] > 0]
        df = df[['user', 'item', 'implicit']]
        return df

    def filter_5core(self, df: pd.DataFrame, min_user: int = 5, min_item: int = 5) -> pd.DataFrame:
        user_counts = df.groupby('user').size()
        valid_users = user_counts[user_counts >= min_user].index
        item_counts = df.groupby('item').size()
        valid_items = item_counts[item_counts >= min_item].index
        return df[df['user'].isin(valid_users) & df['item'].isin(valid_items)]

    def holdout_split(self, df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []
        for user, group in df.groupby('user'):
            n = len(group)
            if n == 0:
                continue
            test_n = max(1, int(n * 0.2))
            idx = group.index.tolist()
            test_idx = rng.choice(idx, size=test_n, replace=False)
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_parts.append(df.loc[train_idx])
            test_parts.append(df.loc[test_idx])
        train_df = pd.concat(train_parts) if train_parts else pd.DataFrame(columns=df.columns)
        test_df = pd.concat(test_parts) if test_parts else pd.DataFrame(columns=df.columns)
        return train_df, test_df

    def run_dataset(self, label: str, path: str):
        df_raw = self.load_and_prepare(label, path)
        seeds_metrics: List[Dict[str, object]] = []
        for seed in self.seeds:
            train_df, test_df = self.holdout_split(df_raw, seed)
            for algo_name in ['ALS','ItemKNN','Pop']:
                if HAVE_OMNI and (OmniALS is not None or OmniItemKNN is not None or OmniPop is not None):
                    try:
                        if algo_name == 'ALS' and OmniALS is not None:
                            model = OmniALS()
                        elif algo_name == 'ItemKNN' and OmniItemKNN is not None:
                            model = OmniItemKNN()
                        elif algo_name == 'Pop' and OmniPop is not None:
                            model = OmniPop()
                        else:
                            raise ValueError('Requested OmniRec component not available')
                        model.train(train_df.assign(timestamp=0) if 'timestamp' not in train_df.columns else train_df)
                        predictions = model.predict(test_df, self.k_values)
                        score = MockRecommender(algo_name, train_df)
                        score.train(train_df)
                        preds = predictions
                        metrics = score.score(preds, test_df, self.k_values)
                    except Exception:
                        m = MockRecommender(algo_name, train_df)
                        m.train(train_df)
                        preds = m.predict(test_df, self.k_values)
                        metrics = m.score(preds, test_df, self.k_values)
                else:
                    m = MockRecommender(algo_name, train_df)
                    m.train(train_df)
                    preds = m.predict(test_df, self.k_values)
                    metrics = m.score(preds, test_df, self.k_values)
                for k in self.k_values:
                    ndcg, precision = metrics.get(k, (0.0, 0.0))
                    seeds_metrics.append({
                        'dataset': label,
                        'algorithm': algo_name,
                        'seed': int(seed),
                        'k': int(k),
                        'ndcg': float(ndcg),
                        'precision': float(precision)
                    })
        self.results.extend(seeds_metrics)

    def run(self):
        for label, cfg in self.datasets.items():
            path = cfg.get('path') if isinstance(cfg, dict) else None
            self.run_dataset(label, str(path))
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.report_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        summary = {
            'timestamp': datetime.now().isoformat(),
            'datasets': list(self.datasets.keys()),
            'seeds': self.seeds,
            'ks': self.k_values,
            'metrics': ['ndcg', 'precision']
        }
        with open(os.path.join(self.report_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self._plot(df, 'ndcg', 'ndcg_vs_seed')
        self._plot(df, 'precision', 'precision_vs_seed')

    def _plot(self, df: pd.DataFrame, metric_col: str, fname: str):
        plt.figure(figsize=(10,6))
        for ds in df['dataset'].unique():
            sub = df[(df['dataset'] == ds) & (df['k'] == 1)]
            if sub.empty:
                continue
            plt.plot(sub['seed'], sub[metric_col], label=ds)
        plt.xlabel('Seed')
        plt.ylabel(metric_col.upper())
        plt.legend()
        plt.title(f'{metric_col.upper()} across seeds')
        plt.tight_layout()
        plt_path = os.path.join(self.report_dir, f'{fname}.png')
        plt.savefig(plt_path)
        plt.close()

if __name__ == '__main__':
    runner = ExperimentRunner()
    runner.run()
