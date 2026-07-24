"""Preregistered-phase audit driver: P2/Mini x5 and P3/Mini x5 native-path baseline.

Reuses the exact inventory logic in ``audit_runs.py`` but scopes it to the ten
new runs and writes to ``study/audit_preregistered/`` so the frozen 2026-07-21
audit in ``study/audit/`` is left untouched. Also emits a condition-level
comparison (P2 vs P3) built only from confound-free signals: API cost, artifact
coverage, reviewer score, and completion markers. Wall-clock and peak memory are
NOT aggregated here because several pairs ran concurrently and shared resources.

Run:  python study/audit_preregistered.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import audit_runs as ar

ROOT = ar.ROOT
RUNS_ROOT = ar.RUNS_ROOT
LOGS_ROOT = ar.LOGS_ROOT
OUT_ROOT = ROOT / "study" / "audit_preregistered"

RUN_LABELS = [f"V100_P{p}_MINI_I05_R0{r}" for p in (2, 3) for r in range(1, 6)]


def condition_of(run: str) -> str:
    if "_P2_" in run:
        return "P2/Mini"
    if "_P3_" in run:
        return "P3/Mini"
    return "other"


def scoped_files() -> list[Path]:
    paths: list[Path] = []
    for label in RUN_LABELS:
        run_dir = RUNS_ROOT / label
        if run_dir.exists():
            paths.extend(p for p in run_dir.rglob("*") if p.is_file())
        log = LOGS_ROOT / f"{label}.console.log"
        if log.is_file():
            paths.append(log)
    return sorted(paths, key=lambda p: str(p).lower())


def build_inventories(files: list[Path]):
    """Reproduces the per-file inventory loop from audit_runs.main(), scoped."""
    manifest: list[dict] = []
    csv_inventory: list[dict] = []
    json_inventory: list[dict] = []
    code_inventory: list[dict] = []
    text_markers: list[dict] = []
    hash_groups: dict[tuple[int, str], list[str]] = defaultdict(list)

    for index, path in enumerate(files, 1):
        root_name = "study_runs" if RUNS_ROOT in path.parents else "study_logs"
        relative = path.relative_to(ROOT).as_posix()
        digest, head, tail = ar.sha256_and_sample(path)
        record = {
            "condition": condition_of(ar.run_name_for(path)),
            "root": root_name,
            "run": ar.run_name_for(path),
            "relative_path": relative,
            "name": path.name,
            "extension": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "readable": True,
        }
        manifest.append(record)
        hash_groups[(path.stat().st_size, digest)].append(relative)

        if path.suffix.lower() == ".csv":
            csv_inventory.append({**record, **ar.inspect_csv(path)})

        if path.suffix.lower() == ".json":
            if path.name in ar.SKIP_FULL_TEXT_NAMES or path.stat().st_size > 64 * 1024 * 1024:
                json_inventory.append({
                    **record,
                    "json_type": "streamed-large",
                    "json_size": "",
                    "json_keys": "",
                    "parse_error": "",
                    "head_sample": ar.decode_sample(head[:2048]).replace("\r", " ").replace("\n", " "),
                    "tail_sample": ar.decode_sample(tail[-2048:]).replace("\r", " ").replace("\n", " "),
                })
            else:
                json_inventory.append({**record, **ar.inspect_small_json(path), "head_sample": "", "tail_sample": ""})

        if path.name == "code.py":
            text = path.read_text(encoding="utf-8", errors="replace")
            code_inventory.append({**record, **ar.inspect_code(text)})

        if path.suffix.lower() in ar.TEXT_EXTENSIONS:
            sample = ar.decode_sample(head + b"\n" + tail).lower()
            text_markers.append({
                **record,
                "contains_traceback": "traceback (most recent call last)" in sample,
                "contains_exception": "exception" in sample,
                "contains_nan": bool(__import__("re").search(r"(?<![a-z])nan(?![a-z])", sample)),
                "contains_placeholder": "placeholder" in sample,
                "contains_synthetic": "synthetic" in sample,
                "contains_download": "download" in sample,
            })

        if index % 100 == 0:
            print(f"audited {index}/{len(files)} files", flush=True)

    return manifest, csv_inventory, json_inventory, code_inventory, text_markers, hash_groups


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = scoped_files()
    print(f"scoped audit over {len(files)} files across {len(RUN_LABELS)} runs")

    manifest, csv_inv, json_inv, code_inv, text_markers, hash_groups = build_inventories(files)

    manifest_by_path = {str((ROOT / row["relative_path"]).resolve()): row for row in manifest}
    runs = []
    for label in RUN_LABELS:
        run_dir = RUNS_ROOT / label
        if run_dir.exists():
            row = ar.parse_run(run_dir, manifest_by_path)
            row = {"condition": condition_of(label), **row}
            runs.append(row)

    duplicates = []
    for (size, digest), paths in hash_groups.items():
        if len(paths) > 1:
            duplicates.append({
                "bytes": size,
                "sha256": digest,
                "copies": len(paths),
                "paths": " | ".join(paths),
            })

    ar.write_csv(OUT_ROOT / "file_manifest.csv", manifest, list(manifest[0]))
    ar.write_csv(OUT_ROOT / "run_summary.csv", runs, list(runs[0]))
    ar.write_csv(OUT_ROOT / "csv_inventory.csv", csv_inv, list(csv_inv[0]))
    ar.write_csv(OUT_ROOT / "json_inventory.csv", json_inv, list(json_inv[0]))
    ar.write_csv(OUT_ROOT / "code_inventory.csv", code_inv, list(code_inv[0]))
    ar.write_csv(OUT_ROOT / "text_marker_inventory.csv", text_markers, list(text_markers[0]))
    ar.write_csv(OUT_ROOT / "duplicate_groups.csv", duplicates, list(duplicates[0]) if duplicates else ["bytes", "sha256", "copies", "paths"])

    # ---- condition-level comparison (confound-free signals only) ----
    def agg(condition: str) -> dict:
        rows = [r for r in runs if r["condition"] == condition]
        def col(name):
            return [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        return {
            "condition": condition,
            "runs": len(rows),
            "total_api_calls": sum(r["api_calls"] for r in rows),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "mean_cost_usd": round(sum(r["cost_usd"] for r in rows) / len(rows), 4) if rows else 0,
            "mean_best_reviewer_score_pct": round(sum(col("best_reviewer_score_pct")) / len(col("best_reviewer_score_pct")), 3) if col("best_reviewer_score_pct") else "",
            "runs_with_satisfactory_node": sum(1 for r in rows if r["satisfactory_true_count"] > 0),
            "total_generated_code_files": sum(r["generated_code_files"] for r in rows),
            "total_prediction_files": sum(r["prediction_files"] for r in rows),
            "total_results_json_files": sum(r["results_json_files"] for r in rows),
            "runs_ended_without_satisfaction": sum(1 for r in rows if r["refinement_without_satisfaction"] or r["found_no_satisfactory_prototype"]),
            "total_traceback_count": sum(r["traceback_count"] for r in rows),
        }

    comparison = [agg("P2/Mini"), agg("P3/Mini")]
    ar.write_csv(OUT_ROOT / "condition_comparison.csv", comparison, list(comparison[0]))

    totals = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "Preregistered native-path baseline (P2/Mini x5, P3/Mini x5)",
        "note_on_confounds": "Wall-clock and peak memory are intentionally omitted from aggregates: several pairs ran concurrently and shared machine resources. API cost, artifact coverage, reviewer score, and completion markers are not affected by concurrency.",
        "runs": len(runs),
        "files": len(manifest),
        "bytes": sum(int(r["bytes"]) for r in manifest),
        "csv_files": len(csv_inv),
        "json_files": len(json_inv),
        "generated_code_files": len(code_inv),
        "duplicate_groups": len(duplicates),
        "duplicate_copies": sum(r["copies"] for r in duplicates),
        "duplicate_bytes_reclaimable": sum(r["bytes"] * (r["copies"] - 1) for r in duplicates),
        "extensions": Counter(r["extension"] or "[none]" for r in manifest),
        "condition_comparison": comparison,
    }
    (OUT_ROOT / "audit_totals.json").write_text(json.dumps(totals, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(totals, indent=2, default=dict), flush=True)


if __name__ == "__main__":
    main()
