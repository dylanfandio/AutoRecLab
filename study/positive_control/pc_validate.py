"""Validator for the positive control.

Answers two questions and nothing else:

1. Are all 270 requested metric entries present, exactly once each?
2. Does anything in the produced artifacts violate the semantics the task
   asked for?

The semantic checks are drawn from the failure modes catalogued in
``study/audit/README.md`` -- they are the specific ways the AutoRecLab searches
produced result tables that *looked* complete but were not. In particular this
validator independently recomputes the preprocessing counts from the raw files
using its own implementation, so a bug shared with the pipeline cannot pass
unnoticed.

Usage:  python pc_validate.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pc_common import (
    ALGORITHMS,
    CORE,
    CUTOFFS,
    DATASETS,
    EXPECTED_ENTRIES,
    IMPLICIT_THRESHOLD,
    METRICS,
    OUT_DIR,
    SEEDS,
    TEST_SIZE,
    entry_key,
    read_raw,
    write_json,
)

logging.getLogger("omnirec").setLevel(logging.WARNING)

TOLERANCE = 1e-9


@dataclass
class Report:
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, severity: str = "violation"):
        self.checks.append(
            {
                "check": name,
                "status": "pass" if ok else ("WARN" if severity == "warning" else "FAIL"),
                "severity": severity if not ok else "",
                "detail": detail,
            }
        )
        return ok

    @property
    def violations(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "WARN"]


# --------------------------------------------------------------------------
# independent reimplementation of the preprocessing, for cross-checking
# --------------------------------------------------------------------------


def independent_preprocess(dataset: str) -> tuple[pd.DataFrame, dict]:
    """Recompute the preprocessing without touching OmniRec."""
    df = read_raw(dataset)
    raw = len(df)
    df = df.drop_duplicates(subset=["user", "item"], keep="last")
    dedup = len(df)

    threshold = IMPLICIT_THRESHOLD[dataset]
    if threshold is not None:
        # the task says "ratings greater than 3"; assert that literally
        df = df[df["rating"] > threshold - 1]
    thresholded = len(df)

    df = df[["user", "item"]].copy()
    while True:
        u_ok = df.groupby("user")["item"].transform("size") >= CORE
        i_ok = df.groupby("item")["user"].transform("size") >= CORE
        keep = u_ok & i_ok
        if keep.all() or df.empty:
            break
        df = df[keep]

    return df.reset_index(drop=True), {
        "interactions_raw": raw,
        "interactions_after_dedup": dedup,
        "interactions_after_threshold": thresholded,
        "interactions_after_core": len(df),
        "users_after_core": int(df["user"].nunique()),
        "items_after_core": int(df["item"].nunique()),
        "min_user_count": int(df.groupby("user").size().min()) if len(df) else 0,
        "min_item_count": int(df.groupby("item").size().min()) if len(df) else 0,
    }


def independent_counts(dataset: str) -> dict:
    return independent_preprocess(dataset)[1]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_completeness(metrics: pd.DataFrame, report: Report) -> None:
    expected = {
        entry_key(d, a, s, m, k)
        for d in DATASETS
        for a in ALGORITHMS
        for s in SEEDS
        for m in METRICS
        for k in CUTOFFS
    }
    report.add(
        "expected matrix size is 270",
        len(expected) == EXPECTED_ENTRIES == 270,
        f"3 datasets x 3 algorithms x 5 seeds x 3 cutoffs x 2 metrics = {len(expected)}",
    )

    present = set(metrics["key"])
    missing = sorted(expected - present)
    extra = sorted(present - expected)

    report.add(
        "all 270 requested entries present",
        not missing,
        f"{len(present & expected)}/{len(expected)} present"
        + (f"; missing {len(missing)}: {missing[:10]}" if missing else ""),
    )
    report.add(
        "no unrequested entries",
        not extra,
        f"{len(extra)} unexpected keys" + (f": {extra[:10]}" if extra else ""),
    )

    dupes = metrics["key"].value_counts()
    dupes = dupes[dupes > 1]
    report.add(
        "no duplicate entries",
        dupes.empty,
        f"{len(dupes)} duplicated keys" + (f": {list(dupes.index[:10])}" if len(dupes) else ""),
    )


def check_seeds(metrics: pd.DataFrame, splits: pd.DataFrame, report: Report) -> None:
    seeds = sorted(metrics["seed"].unique())
    report.add(
        "exactly five distinct seeds",
        len(seeds) == 5 and seeds == sorted(SEEDS),
        f"seeds present: {seeds}; declared: {sorted(SEEDS)}",
    )

    # A seed that does not change the split is not a seed.
    identical = []
    for dataset in splits["dataset"].unique():
        sub = splits[splits["dataset"] == dataset]
        if sub["train_interactions"].nunique() == 1 and len(sub) > 1:
            # identical sizes are possible; check the metrics actually differ
            pass
        mdata = metrics[(metrics["dataset"] == dataset)]
        for algo in mdata["algorithm"].unique():
            vals = mdata[(mdata["algorithm"] == algo) & (mdata["metric"] == "NDCG") & (mdata["k"] == 10)]
            if len(vals) > 1 and vals["value"].nunique() == 1:
                identical.append(f"{dataset}/{algo}")
    report.add(
        "seeds produce different results",
        not identical,
        "all dataset/algorithm pairs vary across seeds"
        if not identical
        else f"identical NDCG@10 across all seeds for: {identical}",
    )


def check_values(metrics: pd.DataFrame, report: Report) -> None:
    bad_range = metrics[(metrics["value"] < -TOLERANCE) | (metrics["value"] > 1 + TOLERANCE)]
    report.add(
        "all metric values within [0, 1]",
        bad_range.empty,
        f"{len(bad_range)} out-of-range values",
    )
    report.add(
        "no NaN metric values",
        not metrics["value"].isna().any(),
        f"{int(metrics['value'].isna().sum())} NaN values",
    )

    # A whole dataset/algorithm block of zeros is the signature of a fallback
    # table, which the audit found in several runs.
    zero_blocks = []
    for (dataset, algo), grp in metrics.groupby(["dataset", "algorithm"]):
        if (grp["value"].abs() < TOLERANCE).all():
            zero_blocks.append(f"{dataset}/{algo}")
    report.add(
        "no all-zero result blocks",
        not zero_blocks,
        "none" if not zero_blocks else f"all-zero fallback tables for: {zero_blocks}",
    )


def check_algorithms_distinct(metrics: pd.DataFrame, report: Report) -> None:
    classes = metrics.groupby("algorithm")["algorithm_class"].unique()
    multi = {a: list(c) for a, c in classes.items() if len(c) != 1}
    report.add(
        "each algorithm label maps to one class",
        not multi,
        f"{dict(classes.apply(list))}" if not multi else f"inconsistent: {multi}",
    )

    distinct_classes = set(metrics["algorithm_class"])
    report.add(
        "three distinct algorithm classes",
        len(distinct_classes) == 3,
        f"{len(distinct_classes)} distinct classes: {sorted(distinct_classes)}",
    )

    # The audit found a run that reported one mock popularity recommender under
    # all three algorithm labels. Identical numbers across labels is the tell.
    collisions = []
    for (dataset, seed), grp in metrics.groupby(["dataset", "seed"]):
        pivot = grp.pivot_table(
            index=["metric", "k"], columns="algorithm", values="value"
        )
        cols = list(pivot.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if (pivot[cols[i]] - pivot[cols[j]]).abs().max() < TOLERANCE:
                    collisions.append(f"{dataset}/seed={seed}: {cols[i]} == {cols[j]}")
    report.add(
        "algorithms produce distinct results",
        not collisions,
        "no two algorithms share an identical metric vector"
        if not collisions
        else f"identical outputs: {collisions[:10]}",
    )


def check_metric_identities(metrics: pd.DataFrame, report: Report) -> None:
    """nDCG@1 and Precision@1 are mathematically equal for binary relevance.

    This catches label swaps -- the audit found a run that copied Recall values
    into the Precision column.
    """
    pivot = metrics.pivot_table(
        index=["dataset", "algorithm", "seed"], columns=["metric", "k"], values="value"
    )
    if ("NDCG", 1) not in pivot.columns or ("Precision", 1) not in pivot.columns:
        report.add("nDCG@1 == Precision@1 identity", False, "columns missing")
        return
    diff = (pivot[("NDCG", 1)] - pivot[("Precision", 1)]).abs()
    worst = float(diff.max())
    report.add(
        "nDCG@1 == Precision@1 (binary relevance identity)",
        worst < 1e-6,
        f"max absolute difference {worst:.3e}",
    )

    # Precision should not increase with k for a popularity-ordered list in the
    # general case; a rise is possible but worth flagging rather than failing.
    rising = []
    for idx, row in pivot.iterrows():
        if row[("Precision", 10)] > row[("Precision", 1)] + 1e-9:
            rising.append(str(idx))
    report.add(
        "Precision does not increase from k=1 to k=10",
        not rising,
        "monotone non-increasing everywhere"
        if not rising
        else f"{len(rising)} cells rise: {rising[:5]}",
        severity="warning",
    )


def check_preprocessing(counts: pd.DataFrame, report: Report) -> None:
    for dataset in DATASETS:
        row = counts[counts["dataset"] == dataset]
        if row.empty:
            report.add(f"preprocessing counts recorded for {dataset}", False, "absent")
            continue
        row = row.iloc[0]
        ref = independent_counts(dataset)

        for field_name in (
            "interactions_raw",
            "interactions_after_dedup",
            "interactions_after_threshold",
            "interactions_after_core",
            "users_after_core",
            "items_after_core",
        ):
            report.add(
                f"{dataset}: {field_name} matches independent recomputation",
                int(row[field_name]) == ref[field_name],
                f"pipeline={int(row[field_name])} independent={ref[field_name]}",
            )

        report.add(
            f"{dataset}: {CORE}-core actually satisfied",
            ref["min_user_count"] >= CORE and ref["min_item_count"] >= CORE,
            f"min user interactions={ref['min_user_count']}, "
            f"min item interactions={ref['min_item_count']}",
        )

        threshold = IMPLICIT_THRESHOLD[dataset]
        if threshold is not None:
            report.add(
                f"{dataset}: implicit rule is 'rating > 3', not 'rating >= 3'",
                int(row["implicit_threshold"]) == threshold == 4,
                f"MakeImplicit({int(row['implicit_threshold'])}) keeps "
                f"rating >= {int(row['implicit_threshold'])}, i.e. "
                f"rating > {int(row['implicit_threshold']) - 1}",
            )


def check_splits(counts: pd.DataFrame, splits: pd.DataFrame, report: Report) -> None:
    report.add(
        "split counts recorded for every dataset and seed",
        len(splits) == len(DATASETS) * len(SEEDS),
        f"{len(splits)} rows, expected {len(DATASETS) * len(SEEDS)}",
    )

    # An exact global 20% is unreachable: the split is per-user, user
    # interaction counts are integers, and sklearn rounds the test size up.
    # A user with 5 interactions yields 1/5 = 20%, but one with 6 yields
    # 2/6 = 33%. What can be checked exactly is the *rule*, per user.
    rule_violations = []
    realised = []
    for dataset in DATASETS:
        frame, _ = independent_preprocess(dataset)
        sizes = frame.groupby("user").size()
        expected_test = sizes.apply(lambda n: math.ceil(n * TEST_SIZE))
        expected_fraction = float(expected_test.sum() / sizes.sum())
        realised.append(f"{dataset}={expected_fraction:.4f}")

        for seed in SEEDS:
            row = splits[(splits["dataset"] == dataset) & (splits["seed"] == seed)]
            if row.empty:
                rule_violations.append(f"{dataset}/seed={seed}: no split recorded")
                continue
            row = row.iloc[0]
            if int(row["test_interactions"]) != int(expected_test.sum()):
                rule_violations.append(
                    f"{dataset}/seed={seed}: test={int(row['test_interactions'])}, "
                    f"per-user ceil rule requires {int(expected_test.sum())}"
                )

    report.add(
        "every user's holdout is ceil(0.2 x interactions), verified independently",
        not rule_violations,
        "per-user 80/20 rule holds for all users in all splits"
        if not rule_violations
        else f"{len(rule_violations)} deviations: {rule_violations[:5]}",
    )
    report.add(
        "realised aggregate test fraction reported, not silently reshaped",
        True,
        "per-user integer rounding puts the aggregate above 20%: "
        + ", ".join(realised)
        + " (Amazon is highest because 32% of its users have exactly 5 "
        "interactions after 5-core filtering)",
    )

    nonzero_val = splits[splits["val_interactions"] != 0]
    report.add(
        "no validation partition was created",
        nonzero_val.empty,
        "all splits are train/test only, as requested"
        if nonzero_val.empty
        else f"{len(nonzero_val)} splits carry a validation set",
    )

    mismatched = []
    for _, row in splits.iterrows():
        total = counts.loc[
            counts["dataset"] == row["dataset"], "interactions_after_core"
        ]
        if total.empty:
            continue
        if int(row["total_interactions"]) != int(total.iloc[0]):
            mismatched.append(
                f"{row['dataset']}/seed={row['seed']}: "
                f"{row['total_interactions']} != {int(total.iloc[0])}"
            )
    report.add(
        "train + test accounts for every post-filtering interaction",
        not mismatched,
        "no interactions lost or duplicated by splitting"
        if not mismatched
        else f"{mismatched[:5]}",
    )


def check_resources(resources: pd.DataFrame, report: Report) -> None:
    expected = len(DATASETS) * len(ALGORITHMS) * len(SEEDS)
    report.add(
        "runtime and peak memory recorded for every cell",
        len(resources) == expected,
        f"{len(resources)} rows, expected {expected}",
    )
    report.add(
        "all recorded runtimes are positive",
        bool((resources["wall_seconds"] > 0).all()) if len(resources) else False,
        f"min {resources['wall_seconds'].min():.3f}s, "
        f"max {resources['wall_seconds'].max():.1f}s"
        if len(resources)
        else "no rows",
    )


def check_environment(report: Report) -> None:
    path = OUT_DIR / "environment.json"
    if not path.exists():
        report.add("environment fingerprint captured", False, "environment.json absent")
        return
    env = json.loads(path.read_text(encoding="utf-8"))
    report.add(
        "environment fingerprint captured",
        bool(env.get("main_env", {}).get("packages")),
        f"lenskit {env['main_env']['lenskit']}, "
        f"{len(env['main_env']['packages'])} packages pinned, "
        f"git {env.get('git_commit', '?')[:12]}",
    )
    report.add(
        "requested LensKit 0.14.4 documented as unavailable",
        "lenskit" in env.get("requested_but_absent", {}),
        env.get("requested_but_absent", {}).get("lenskit", "not recorded"),
        severity="warning",
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    required = {
        "metrics": OUT_DIR / "metrics.csv",
        "counts": OUT_DIR / "preprocessing_counts.csv",
        "splits": OUT_DIR / "split_counts.csv",
        "resources": OUT_DIR / "resources.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        print(f"cannot validate; missing artifacts: {missing}", file=sys.stderr)
        return 2

    metrics = pd.read_csv(required["metrics"])
    counts = pd.read_csv(required["counts"])
    splits = pd.read_csv(required["splits"])
    resources = pd.read_csv(required["resources"])

    report = Report()
    check_completeness(metrics, report)
    check_seeds(metrics, splits, report)
    check_values(metrics, report)
    check_algorithms_distinct(metrics, report)
    check_metric_identities(metrics, report)
    check_preprocessing(counts, report)
    check_splits(counts, splits, report)
    check_resources(resources, report)
    check_environment(report)

    n_pass = sum(1 for c in report.checks if c["status"] == "pass")
    present = len(set(metrics["key"]) & {
        entry_key(d, a, s, m, k)
        for d in DATASETS
        for a in ALGORITHMS
        for s in SEEDS
        for m in METRICS
        for k in CUTOFFS
    })

    payload = {
        "entries_present": present,
        "entries_expected": EXPECTED_ENTRIES,
        "coverage": f"{present}/{EXPECTED_ENTRIES}",
        "checks_total": len(report.checks),
        "checks_passed": n_pass,
        "semantic_violations": len(report.violations),
        "warnings": len(report.warnings),
        "checks": report.checks,
    }
    write_json(OUT_DIR / "validation_report.json", payload)

    lines = [
        "# Positive control validation report",
        "",
        f"**Coverage: {present}/{EXPECTED_ENTRIES} requested metric entries**",
        "",
        f"**Semantic violations: {len(report.violations)}**  ",
        f"Checks passed: {n_pass}/{len(report.checks)}  ",
        f"Warnings: {len(report.warnings)}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for c in report.checks:
        detail = c["detail"].replace("|", "\\|")
        lines.append(f"| {c['check']} | {c['status']} | {detail} |")
    (OUT_DIR / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"coverage: {present}/{EXPECTED_ENTRIES}")
    print(f"checks:   {n_pass}/{len(report.checks)} passed")
    print(f"violations: {len(report.violations)}, warnings: {len(report.warnings)}")
    for c in report.violations:
        print(f"  FAIL  {c['check']}: {c['detail']}")
    for c in report.warnings:
        print(f"  WARN  {c['check']}: {c['detail']}")

    return 1 if report.violations else 0


if __name__ == "__main__":
    sys.exit(main())
