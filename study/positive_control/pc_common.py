"""Shared building blocks for the human-authored positive control.

This module deliberately reuses OmniRec's own preprocessing components
(``MakeImplicit``, ``CorePruning``) so that the Tier A (native OmniRec runner)
and Tier B (direct LensKit) pipelines differ *only* in their train/predict/
evaluate backend. Any divergence in results is therefore attributable to the
backend, not to preprocessing.

The task specification being reproduced is ``study/prompts/P0_original.txt``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split

from omnirec.data_variants import RawData, SplitData
from omnirec.preprocess.core_pruning import CorePruning
from omnirec.preprocess.feedback_conversion import MakeImplicit
from omnirec.recsys_data_set import RecSysDataSet

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "study" / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"
CANON_DIR = OUT_DIR / "canon"

# The prompt asks for "5 different random seeds". They are fixed and recorded
# verbatim here so the control is reproducible; nothing is drawn from entropy.
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

# The prompt asks for nDCG@k and Precision@k for k in {1, 5, 10}.
CUTOFFS: tuple[int, ...] = (1, 5, 10)
METRICS: tuple[str, ...] = ("NDCG", "Precision")

# ALS / ItemKNN / Pop as named in the prompt, mapped onto the classes that
# actually exist in the pinned stack.
ALGORITHMS: tuple[str, ...] = ("ALS", "ItemKNN", "Pop")

DATASETS: tuple[str, ...] = ("MovieLens100K", "AmazonVideoGames", "HetrecLastFM")

# 3 datasets x 3 algorithms x 5 seeds x 3 cutoffs x 2 metrics
EXPECTED_ENTRIES = (
    len(DATASETS) * len(ALGORITHMS) * len(SEEDS) * len(CUTOFFS) * len(METRICS)
)

TEST_SIZE = 0.2

# "convert any ratings greater than 3 to implicit interactions".
# MakeImplicit(t) keeps rating >= t, so strictly-greater-than-3 on the integer
# rating scales of MovieLens and Amazon is MakeImplicit(4), NOT MakeImplicit(3).
IMPLICIT_THRESHOLD: dict[str, int | None] = {
    "MovieLens100K": 4,
    "AmazonVideoGames": 4,
    "HetrecLastFM": None,  # already implicit; no rating column to threshold
}

CORE = 5


# --------------------------------------------------------------------------
# raw input handling
# --------------------------------------------------------------------------


@dataclass
class RawSource:
    name: str
    filename: str
    note: str


RAW_SOURCES: dict[str, RawSource] = {
    "MovieLens100K": RawSource(
        "MovieLens100K", "u.data", "tab-separated user/item/rating/timestamp"
    ),
    "AmazonVideoGames": RawSource(
        "AmazonVideoGames",
        "VideoGames.csv",
        "headerless CSV in Amazon-2014 column order: item,user,rating,timestamp",
    ),
    "HetrecLastFM": RawSource(
        "HetrecLastFM",
        "UserTaggedArtiststimestamps.dat",
        "tab-separated user-tagged-artists; rating synthesised as 1 (implicit)",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_raw(name: str) -> pd.DataFrame:
    """Read one of the three raw files supplied in ``study/data``.

    The column conventions mirror OmniRec's own registered loaders for the
    corresponding datasets, so that the only difference from the built-in
    loaders is the file location.
    """
    path = DATA_DIR / RAW_SOURCES[name].filename
    if name == "MovieLens100K":
        return pd.read_csv(
            path, sep="\t", names=["user", "item", "rating", "timestamp"]
        )
    if name == "AmazonVideoGames":
        return pd.read_csv(path, names=["item", "user", "rating", "timestamp"])
    if name == "HetrecLastFM":
        df = pd.read_csv(
            path,
            sep="\t",
            header=0,
            usecols=["userID", "artistID", "timestamp"],
        ).rename(columns={"userID": "user", "artistID": "item"})
        df["rating"] = 1
        return df
    raise ValueError(f"unknown dataset {name}")


def canonicalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replicate ``RecSysDataSet._canonicalize`` transparently.

    Returns the canonical frame plus the counts observed at each stage, so the
    numbers can be reported rather than inferred.
    """
    counts: dict = {"interactions_raw": int(len(df))}

    # 1. drop duplicate (user, item) pairs, keeping the last occurrence
    df = df.drop_duplicates(subset=["user", "item"], keep="last").copy()
    counts["interactions_after_dedup"] = int(len(df))
    counts["duplicates_removed"] = (
        counts["interactions_raw"] - counts["interactions_after_dedup"]
    )

    # 2. normalise identifiers to a dense 0..n-1 range, ordered by first sight
    for col in ("user", "item"):
        mapping = {key: value for value, key in enumerate(df[col].unique())}
        df[col] = df[col].map(mapping)

    # 3. normalise timestamps to integer seconds
    ts_note = "ok"
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        try:
            if pd.api.types.is_numeric_dtype(ts):
                converted = pd.to_datetime(ts, unit="s", errors="coerce", utc=True)
            else:
                converted = pd.to_datetime(ts, errors="coerce", utc=True)
            n_bad = int(converted.isna().sum())
            if n_bad:
                # OmniRec would coerce these to NaT and then to a sentinel int.
                # Timestamps are unused by a random user holdout, so we retain
                # the raw values and record the anomaly instead of propagating
                # a garbage sentinel.
                ts_note = (
                    f"{n_bad} timestamps out of range for unit='s' "
                    "(values look like milliseconds); raw values retained"
                )
            else:
                df["timestamp"] = converted.astype("int64") // 10**9
        except Exception as exc:  # pragma: no cover - defensive
            ts_note = f"timestamp normalisation failed: {exc!r}; raw values retained"
    else:
        ts_note = "no timestamp column"
    counts["timestamp_normalisation"] = ts_note

    counts["users_canonical"] = int(df["user"].nunique())
    counts["items_canonical"] = int(df["item"].nunique())
    return df.reset_index(drop=True), counts


def build_dataset(name: str) -> tuple[RecSysDataSet, dict]:
    """Produce a canonical ``RecSysDataSet`` for ``name`` from the supplied file.

    The canonical frame is written to disk and handed back to OmniRec through
    the public ``use_dataloader(..., canon_path=...)`` entry point, which
    short-circuits download and re-canonicalisation when the file exists.
    """
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    source = RAW_SOURCES[name]
    raw_path = DATA_DIR / source.filename

    df, counts = canonicalize(read_raw(name))
    counts["dataset"] = name
    counts["raw_file"] = source.filename
    counts["raw_sha256"] = sha256(raw_path)
    counts["raw_note"] = source.note

    canon_path = CANON_DIR / f"{name}.csv"
    df.to_csv(canon_path, index=False)
    counts["canon_path"] = str(canon_path)
    counts["canon_sha256"] = sha256(canon_path)

    dataset = RecSysDataSet.use_dataloader(name, canon_path=canon_path)
    return dataset, counts


def preprocess(name: str) -> tuple[pd.DataFrame, dict]:
    """Apply the requested preprocessing and report counts at every stage.

    Order is threshold-then-core, matching the ordering used by OmniRec's own
    documented example pipeline. The count that would result from the reverse
    order is also recorded, because the prompt's wording is ambiguous and the
    difference is worth being explicit about.
    """
    dataset, counts = build_dataset(name)
    threshold = IMPLICIT_THRESHOLD[name]

    counts["implicit_threshold"] = threshold
    counts["implicit_rule"] = (
        f"rating >= {threshold} (i.e. rating > {threshold - 1})"
        if threshold is not None
        else "n/a - dataset is already implicit"
    )

    if threshold is not None:
        dataset = MakeImplicit(threshold).process(dataset)
    counts["interactions_after_threshold"] = int(len(dataset._data.df))

    before_core = dataset._data.df.copy()
    dataset = CorePruning(CORE).process(dataset)
    df = dataset._data.df.reset_index(drop=True)

    counts["interactions_after_core"] = int(len(df))
    counts["users_after_core"] = int(df["user"].nunique())
    counts["items_after_core"] = int(df["item"].nunique())
    counts["core"] = CORE
    counts["preprocessing_order"] = (
        "MakeImplicit -> CorePruning" if threshold is not None else "CorePruning"
    )

    # sensitivity: what the reverse order would have produced
    if threshold is not None:
        alt = RecSysDataSet(RawData(before_core.copy()))
        alt._data.df = read_raw(name)  # re-read: reverse order needs ratings
        alt._data.df, _ = canonicalize(alt._data.df)
        alt = CorePruning(CORE).process(alt)
        alt = MakeImplicit(threshold).process(alt)
        counts["interactions_reverse_order"] = int(len(alt._data.df))
    else:
        counts["interactions_reverse_order"] = counts["interactions_after_core"]

    return df, counts


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def user_holdout_80_20(df: pd.DataFrame, seed: int) -> SplitData:
    """Per-user 80/20 holdout with no validation partition.

    OmniRec's ``UserHoldout`` cannot express this: it always carves a
    validation slice out of the training set, and ``validation_size=0`` makes
    the underlying ``train_test_split`` raise ``InvalidParameterError``. This
    function is the minimal faithful implementation of what the prompt asks
    for. ``val`` is returned as an empty frame with the correct schema.
    """
    df = df.reset_index(drop=True)
    train_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []
    skipped_users = 0

    for _user, items in df.groupby("user").indices.items():
        if len(items) < 2:
            # cannot hold anything out; keep entirely in train
            train_idx.append(items)
            skipped_users += 1
            continue
        tr, te = train_test_split(items, test_size=TEST_SIZE, random_state=seed)
        train_idx.append(tr)
        test_idx.append(te)

    train = df.iloc[np.concatenate(train_idx)] if train_idx else df.iloc[:0]
    test = df.iloc[np.concatenate(test_idx)] if test_idx else df.iloc[:0]
    val = df.iloc[:0]

    split = SplitData(
        train.reset_index(drop=True), val.copy(), test.reset_index(drop=True)
    )
    split.skipped_users = skipped_users  # type: ignore[attr-defined]
    return split


def split_counts(name: str, seed: int, split: SplitData) -> dict:
    train, test = split.train, split.test
    total = len(train) + len(test)
    return {
        "dataset": name,
        "seed": seed,
        "train_interactions": int(len(train)),
        "test_interactions": int(len(test)),
        "val_interactions": int(len(split.val)),
        "total_interactions": int(total),
        "test_fraction": round(len(test) / total, 6) if total else 0.0,
        "train_users": int(train["user"].nunique()),
        "test_users": int(test["user"].nunique()),
        "train_items": int(train["item"].nunique()),
        "test_items": int(test["item"].nunique()),
        "users_without_holdout": int(getattr(split, "skipped_users", 0)),
    }


# --------------------------------------------------------------------------
# resource measurement
# --------------------------------------------------------------------------


@dataclass
class ResourceSample:
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    baseline_rss_bytes: int = 0
    phases: dict = field(default_factory=dict)

    @property
    def peak_rss_mib(self) -> float:
        return self.peak_rss_bytes / 1024**2

    @property
    def delta_rss_mib(self) -> float:
        return (self.peak_rss_bytes - self.baseline_rss_bytes) / 1024**2


@contextmanager
def measure(interval: float = 0.05):
    """Sample process RSS in a background thread for the duration of a block.

    ``tracemalloc`` is not used because it only sees Python-level allocations
    and would miss the NumPy/Numba buffers that dominate here.
    """
    proc = psutil.Process()
    sample = ResourceSample(baseline_rss_bytes=proc.memory_info().rss)
    sample.peak_rss_bytes = sample.baseline_rss_bytes
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
            except psutil.Error:  # pragma: no cover
                return
            if rss > sample.peak_rss_bytes:
                sample.peak_rss_bytes = rss
            stop.wait(interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        yield sample
    finally:
        sample.wall_seconds = time.perf_counter() - start
        stop.set()
        thread.join(timeout=2.0)


# --------------------------------------------------------------------------
# environment fingerprint
# --------------------------------------------------------------------------


_FREEZE_SNIPPET = (
    "import importlib.metadata as m;"
    "print('\\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' "
    "for d in m.distributions() if d.metadata['Name'])))"
)


def _pip_freeze(python: Path | str) -> dict[str, str]:
    """Enumerate installed distributions in the interpreter at ``python``.

    ``pip freeze`` is not usable here: uv-created virtualenvs do not install
    pip, so this goes through ``importlib.metadata`` instead, which reads the
    same installed-distribution metadata.
    """
    try:
        out = subprocess.run(
            [str(python), "-c", _FREEZE_SNIPPET],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover
        return {"__error__": repr(exc)}
    if out.returncode != 0:  # pragma: no cover
        return {"__error__": out.stderr.strip()[:500]}
    packages: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "==" in line:
            pkg, _, version = line.partition("==")
            packages[pkg.strip().lower()] = version.strip()
    return packages


def environment_fingerprint() -> dict:
    """Record exactly what the control ran against, for later Docker parity."""
    import lenskit
    import omnirec

    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"

    runner_env = Path.home() / ".omnirec" / "data" / "envs" / "LensKit_env"
    runner_python = (
        runner_env / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else runner_env / "bin" / "python"
    )

    fingerprint = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "cpu_count": psutil.cpu_count(logical=True),
        "total_memory_bytes": psutil.virtual_memory().total,
        "main_env": {
            "lenskit": lenskit.__version__,
            "omnirec": getattr(omnirec, "__version__", "1.0.0"),
            "packages": _pip_freeze(sys.executable),
        },
        "requested_but_absent": {
            "lenskit": "0.14.4 (requested by P0_original.txt; not installable "
            "in this stack and not present in any environment)"
        },
        "seeds": list(SEEDS),
        "cutoffs": list(CUTOFFS),
        "metrics": list(METRICS),
        "algorithms": list(ALGORITHMS),
        "datasets": list(DATASETS),
        "expected_metric_entries": EXPECTED_ENTRIES,
    }

    if runner_python.exists():
        fingerprint["omnirec_lenskit_runner_env"] = {
            "path": str(runner_env),
            "packages": _pip_freeze(runner_python),
        }
    else:
        fingerprint["omnirec_lenskit_runner_env"] = {
            "path": str(runner_env),
            "packages": {},
            "note": "runner env not materialised",
        }
    return fingerprint


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def entry_key(dataset: str, algorithm: str, seed: int, metric: str, k: int) -> str:
    return f"{dataset}|{algorithm}|seed={seed}|{metric}@{k}"
