from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "study_runs"
OUT = ROOT / "study" / "audit"


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_for(path: Path) -> str:
    return path.relative_to(RUNS).parts[0]


def native_metrics() -> None:
    metrics: list[dict] = []
    result_files: list[dict] = []
    for path in RUNS.rglob("results.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        file_rows: list[dict] = []
        if isinstance(value, dict):
            for dataset, entries in value.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    algorithm_id = str(entry.get("algorithm", ""))
                    algorithm_match = re.search(r"LensKit\.([^-]+)", algorithm_id)
                    seed_match = re.search(r"-([0-9]+)$", algorithm_id)
                    raw_value = entry.get("value")
                    try:
                        numeric = float(raw_value)
                        finite = math.isfinite(numeric)
                    except (TypeError, ValueError):
                        numeric = ""
                        finite = False
                    row = {
                        "run": run_for(path),
                        "relative_path": path.relative_to(ROOT).as_posix(),
                        "dataset_id": dataset,
                        "algorithm_id": algorithm_id,
                        "algorithm": algorithm_match.group(1) if algorithm_match else algorithm_id,
                        "seed": seed_match.group(1) if seed_match else "",
                        "metric": str(entry.get("name", "")),
                        "k": entry.get("k", ""),
                        "value": numeric,
                        "finite": finite,
                    }
                    metrics.append(row)
                    file_rows.append(row)
        result_files.append({
            "run": run_for(path),
            "relative_path": path.relative_to(ROOT).as_posix(),
            "metric_rows": len(file_rows),
            "datasets": "|".join(sorted({str(row["dataset_id"]) for row in file_rows})),
            "algorithms": "|".join(sorted({str(row["algorithm"]) for row in file_rows})),
            "seeds": "|".join(sorted({str(row["seed"]) for row in file_rows})),
            "metrics": "|".join(sorted({str(row["metric"]) for row in file_rows})),
            "ks": "|".join(sorted({str(row["k"]) for row in file_rows})),
            "all_finite": all(bool(row["finite"]) for row in file_rows),
        })
    write_csv("native_metrics.csv", metrics)
    write_csv("native_result_files.csv", result_files)


def generated_code() -> None:
    rows: list[dict] = []
    for path in RUNS.rglob("code.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        out_path = path.with_name("out.log")
        out = out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""
        seed_literals = re.findall(r"(?im)^\s*seeds?\s*(?::[^=]+)?=\s*([^\n#]+)", text)
        rows.append({
            "run": run_for(path),
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "has_out_log": out_path.exists(),
            "out_log_bytes": out_path.stat().st_size if out_path.exists() else 0,
            "out_traceback": "Traceback (most recent call last)" in out,
            "out_program_crashed": "Program crashed" in out,
            "mentions_run_omnirec": "run_omnirec" in lower,
            "imports_omnirec": "omnirec" in lower,
            "uses_mock_or_fallback": bool(re.search(r"(?i)\bmock|fallback|placeholder|synthetic", text)),
            "loads_u_data": "u.data" in lower,
            "loads_video_games": "videogames.csv" in lower,
            "loads_lastfm": "usertaggedartiststimestamps.dat" in lower,
            "mentions_als": bool(re.search(r"(?i)\bals\b|implicitmfscorer", text)),
            "mentions_itemknn": "itemknn" in lower,
            "mentions_pop": bool(re.search(r"(?i)\bpop\b|popscorer|mostpopular", text)),
            "mentions_ndcg": "ndcg" in lower,
            "mentions_precision": "precision" in lower,
            "mentions_recall": "recall" in lower,
            "explicit_download": "download(" in lower or "requests.get" in lower or "urlretrieve" in lower,
            "seed_literals": " || ".join(seed_literals)[:1000],
        })
    write_csv("generated_code_assessment.csv", rows)


if __name__ == "__main__":
    native_metrics()
    generated_code()
