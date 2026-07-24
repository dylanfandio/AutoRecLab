from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "study_runs"
LOGS_ROOT = ROOT / "study_logs"
OUT_ROOT = ROOT / "study" / "audit"
CHUNK = 8 * 1024 * 1024
TEXT_EXTENSIONS = {".csv", ".json", ".log", ".md", ".py", ".toml", ".txt", ".svg"}
SKIP_FULL_TEXT_NAMES = {"predictions.json"}


def sha256_and_sample(path: Path, sample_limit: int = 131072) -> tuple[str, bytes, bytes]:
    digest = hashlib.sha256()
    head = bytearray()
    tail = b""
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
            if len(head) < sample_limit:
                head.extend(block[: sample_limit - len(head)])
            tail = (tail + block)[-sample_limit:]
    return digest.hexdigest(), bytes(head), tail


def decode_sample(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def run_name_for(path: Path) -> str:
    try:
        return path.relative_to(RUNS_ROOT).parts[0]
    except ValueError:
        stem = path.name
        return stem.removesuffix(".console.log")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inspect_csv(path: Path) -> dict:
    result = {"rows": "", "columns": "", "parse_error": ""}
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        result["rows"] = rows
        result["columns"] = "|".join(header)
    except Exception as exc:  # audit must continue past malformed generated files
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_small_json(path: Path) -> dict:
    result = {"json_type": "", "json_size": "", "json_keys": "", "parse_error": ""}
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
            value = json.load(handle)
        result["json_type"] = type(value).__name__
        if isinstance(value, (list, dict)):
            result["json_size"] = len(value)
        if isinstance(value, dict):
            result["json_keys"] = "|".join(map(str, value.keys()))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            result["json_keys"] = "|".join(map(str, value[0].keys()))
    except Exception as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_code(text: str) -> dict:
    lower = text.lower()
    seed_values = sorted(set(re.findall(r"(?i)(?:seeds?|random_state)\s*=\s*\[?([^\]\n#]+)", text)))
    return {
        "mentions_movielens": "movielens" in lower or "u.data" in lower,
        "mentions_amazon": "amazon" in lower or "videogames.csv" in lower,
        "mentions_lastfm": "last.fm" in lower or "lastfm" in lower or "usertaggedartist" in lower,
        "mentions_als": bool(re.search(r"(?i)\bals\b|implicitmf", text)),
        "mentions_itemknn": "itemknn" in lower,
        "mentions_pop": bool(re.search(r"(?i)mostpopular|popscorer|\bpop\b", text)),
        "mentions_ndcg": "ndcg" in lower,
        "mentions_precision": "precision" in lower,
        "mentions_download": "download" in lower,
        "mentions_synthetic": "synthetic" in lower or "placeholder" in lower,
        "seed_expressions": " || ".join(seed_values)[:1000],
    }


def parse_run(run_dir: Path, manifest_by_path: dict[str, dict]) -> dict:
    run = run_dir.name
    config_path = run_dir / "config.toml"
    costs_path = run_dir / "costs_log.csv"
    debug_path = run_dir / "debug.log"
    summary_path = run_dir / "summary.md"
    entered_prompt = run_dir / "entered_prompt.txt"

    model = ""
    iterations = ""
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8", errors="replace")
        model_match = re.search(r'^model\s*=\s*"([^"]+)"', text, re.MULTILINE)
        iteration_match = re.search(r"^max_iterations\s*=\s*(\d+)", text, re.MULTILINE)
        model = model_match.group(1) if model_match else ""
        iterations = iteration_match.group(1) if iteration_match else ""

    total_cost = 0.0
    first_timestamp = None
    last_timestamp = None
    api_calls = 0
    if costs_path.exists():
        with costs_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Position") == "SUMMARIZED":
                    continue
                api_calls += 1
                try:
                    total_cost += float(row.get("Total USD", 0) or 0)
                except ValueError:
                    pass
                try:
                    stamp = datetime.strptime(row.get("Timestamp", ""), "%Y-%m-%d %H:%M:%S")
                    first_timestamp = min(first_timestamp, stamp) if first_timestamp else stamp
                    last_timestamp = max(last_timestamp, stamp) if last_timestamp else stamp
                except ValueError:
                    pass

    debug = debug_path.read_text(encoding="utf-8", errors="replace") if debug_path.exists() else ""
    scores = [float(value) for value in re.findall(r"Scored node:\s+([0-9.]+)%", debug)]
    final_scores = [float(value) * 100 for value in re.findall(r"Final score:\s+([0-9.]+)\s+\(", debug)]
    all_scores = scores + final_scores

    req_proto = run_dir / "code_requirements_prototype.json"
    req_final = run_dir / "code_requirements_final.json"

    def json_len(path: Path) -> str | int:
        if not path.exists():
            return ""
        try:
            value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
            return len(value) if isinstance(value, (list, dict)) else ""
        except Exception:
            return ""

    files = [row for key, row in manifest_by_path.items() if row["run"] == run and row["root"] == "study_runs"]
    csv_files = [row for row in files if row["extension"] == ".csv"]
    result_csvs = [row for row in csv_files if re.search(r"(?i)result|metric|summary", row["name"])]
    generated_code = [row for row in files if row["name"] == "code.py"]
    prediction_files = [row for row in files if row["name"] == "predictions.json"]
    result_jsons = [row for row in files if row["name"] == "results.json"]

    prompt_hash = manifest_by_path.get(str(entered_prompt.resolve()), {}).get("sha256", "") if entered_prompt.exists() else ""
    return {
        "run": run,
        "model": model,
        "max_iterations": iterations,
        "files": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "api_calls": api_calls,
        "cost_usd": round(total_cost, 6),
        "elapsed_minutes_from_cost_log": round((last_timestamp - first_timestamp).total_seconds() / 60, 2) if first_timestamp and last_timestamp else "",
        "has_console_log": (LOGS_ROOT / f"{run}.console.log").exists(),
        "has_summary": summary_path.exists(),
        "prototype_requirements": json_len(req_proto),
        "final_requirements": json_len(req_final),
        "best_reviewer_score_pct": max(all_scores) if all_scores else "",
        "satisfactory_true_count": len(re.findall(r"is_satisfactory=True", debug)),
        "execution_success_count": len(re.findall(r"has_exception=False", debug)),
        "execution_failure_count": len(re.findall(r"has_exception=True", debug)),
        "traceback_count": len(re.findall(r"Traceback \(most recent call last\)", debug)),
        "program_crash_count": len(re.findall(r"Program crashed", debug)),
        "generated_code_files": len(generated_code),
        "prediction_files": len(prediction_files),
        "results_json_files": len(result_jsons),
        "csv_files": len(csv_files),
        "candidate_result_csv_files": len(result_csvs),
        "prompt_sha256": prompt_hash,
        "ended_with_final_response": "Final response:" in debug,
        "found_no_satisfactory_prototype": "Found no satisfactory prototype" in debug,
        "refinement_without_satisfaction": "Refinement loop ended without full satisfaction" in debug,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [*RUNS_ROOT.rglob("*"), *LOGS_ROOT.rglob("*")],
        key=lambda path: str(path).lower(),
    )
    files = [path for path in files if path.is_file()]

    manifest: list[dict] = []
    csv_inventory: list[dict] = []
    json_inventory: list[dict] = []
    code_inventory: list[dict] = []
    text_markers: list[dict] = []
    hash_groups: dict[tuple[int, str], list[str]] = defaultdict(list)

    for index, path in enumerate(files, 1):
        root_name = "study_runs" if RUNS_ROOT in path.parents else "study_logs"
        relative = path.relative_to(ROOT).as_posix()
        digest, head, tail = sha256_and_sample(path)
        record = {
            "root": root_name,
            "run": run_name_for(path),
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
            csv_inventory.append({**record, **inspect_csv(path)})

        if path.suffix.lower() == ".json":
            if path.name in SKIP_FULL_TEXT_NAMES or path.stat().st_size > 64 * 1024 * 1024:
                json_inventory.append({
                    **record,
                    "json_type": "streamed-large",
                    "json_size": "",
                    "json_keys": "",
                    "parse_error": "",
                    "head_sample": decode_sample(head[:2048]).replace("\r", " ").replace("\n", " "),
                    "tail_sample": decode_sample(tail[-2048:]).replace("\r", " ").replace("\n", " "),
                })
            else:
                json_inventory.append({**record, **inspect_small_json(path), "head_sample": "", "tail_sample": ""})

        if path.name == "code.py":
            text = path.read_text(encoding="utf-8", errors="replace")
            code_inventory.append({**record, **inspect_code(text)})

        if path.suffix.lower() in TEXT_EXTENSIONS:
            sample = decode_sample(head + b"\n" + tail).lower()
            text_markers.append({
                **record,
                "contains_traceback": "traceback (most recent call last)" in sample,
                "contains_exception": "exception" in sample,
                "contains_nan": bool(re.search(r"(?<![a-z])nan(?![a-z])", sample)),
                "contains_placeholder": "placeholder" in sample,
                "contains_synthetic": "synthetic" in sample,
                "contains_download": "download" in sample,
            })

        if index % 100 == 0:
            print(f"audited {index}/{len(files)} files", flush=True)

    manifest_by_path = {str((ROOT / row["relative_path"]).resolve()): row for row in manifest}
    runs = [parse_run(path, manifest_by_path) for path in sorted(RUNS_ROOT.iterdir()) if path.is_dir()]

    duplicates = []
    for (size, digest), paths in hash_groups.items():
        if len(paths) > 1:
            duplicates.append({
                "bytes": size,
                "sha256": digest,
                "copies": len(paths),
                "paths": " | ".join(paths),
            })

    write_csv(OUT_ROOT / "file_manifest.csv", manifest, list(manifest[0]))
    write_csv(OUT_ROOT / "run_summary.csv", runs, list(runs[0]))
    write_csv(OUT_ROOT / "csv_inventory.csv", csv_inventory, list(csv_inventory[0]))
    write_csv(OUT_ROOT / "json_inventory.csv", json_inventory, list(json_inventory[0]))
    write_csv(OUT_ROOT / "code_inventory.csv", code_inventory, list(code_inventory[0]))
    write_csv(OUT_ROOT / "text_marker_inventory.csv", text_markers, list(text_markers[0]))
    write_csv(OUT_ROOT / "duplicate_groups.csv", duplicates, list(duplicates[0]) if duplicates else ["bytes", "sha256", "copies", "paths"])

    totals = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": len(manifest),
        "bytes": sum(int(row["bytes"]) for row in manifest),
        "runs": len(runs),
        "csv_files": len(csv_inventory),
        "json_files": len(json_inventory),
        "generated_code_files": len(code_inventory),
        "duplicate_groups": len(duplicates),
        "duplicate_copies": sum(row["copies"] for row in duplicates),
        "duplicate_bytes_reclaimable": sum(row["bytes"] * (row["copies"] - 1) for row in duplicates),
        "extensions": Counter(row["extension"] or "[none]" for row in manifest),
    }
    (OUT_ROOT / "audit_totals.json").write_text(json.dumps(totals, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(totals, indent=2, default=dict), flush=True)


if __name__ == "__main__":
    main()
